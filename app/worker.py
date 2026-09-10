"""Dedicated video workers. Each process claims one Mongo job at a time.

The website API only inserts `status=queued` rows. These processes download
clips from S3, run MediaPipe/OpenCV, and write progress back to the
same job documents.
Closing a browser tab cannot stop a claimed job. The Cancel button is
honored at the next stage boundary; the current CV loop is allowed to finish.
"""

from __future__ import annotations

import asyncio
import logging
import os
import socket
import time
import traceback
from datetime import datetime, timedelta, timezone
from pathlib import Path

from pymongo import ReturnDocument

from app.balltrack import repo as bt_repo
from app.balltrack.runner import run_balltrack_job
from app.config import get_settings
from app.db import quota, repository as repo
from app.logging_config import configure_logging
from app.db.mongo import close_mongo, get_db, ping_mongo
from app.pipeline import clip_spec
from app.pipeline.cancel import JobCancelled, raise_if_cancelled
from app.pipeline.job_progress import JobReporter
from app.pipeline.runner import run_analysis_job
from app.services import cleanup, original_archive, s3_service

log = logging.getLogger("criclab.video-worker")
WORKER_ID = f"{socket.gethostname()}-{os.getpid()}"
_VIDEO_SUFFIXES = {".mp4", ".mov", ".avi", ".mkv", ".webm"}


def _ingest_dest(record_id: str, source_key: str, *, balltrack: bool) -> Path:
    """Write under this process STORAGE_DIR — never Mongo's path (other machines)."""
    suffix = Path(source_key).suffix.lower() or ".mp4"
    if suffix not in _VIDEO_SUFFIXES:
        suffix = ".mp4"
    root = get_settings().storage_path
    folder = (root / "balltrack" / "videos") if balltrack else (root / "videos")
    return folder / f"{record_id}{suffix}"


async def _fetch_source_video(
    source_key: str | None,
    record_id: str,
    job_id: str,
    *,
    balltrack: bool,
    local_path: str | None = None,
) -> Path:
    key = (source_key or "").strip().lstrip("/")
    if key and s3_service.s3_configured():
        if not s3_service.is_our_object_key(key):
            log.debug("fetch reject key not ours job_id=%s key=%s", job_id, key)
            raise ValueError("source_key is not an S3 object key")
        dest = _ingest_dest(record_id, key, balltrack=balltrack)
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.unlink(missing_ok=True)
        progress = JobReporter(job_id, update=bt_repo.update_job if balltrack else None)
        await progress.aset("ingest", 0, "Fetching your clip", force=True)
        log.info("downloading s3://%s → %s", key, dest)

        async def on_dl(written: int, total: int | None) -> None:
            tot = int(total or 0)
            frac = (written / tot) if tot else 0.0
            mb_w = written / (1024 * 1024)
            if tot:
                msg = f"Fetching your clip — {mb_w:.1f} of {tot / (1024 * 1024):.1f} MB"
                detail = {"current": written, "total": tot, "unit": "bytes"}
            else:
                msg = f"Fetching your clip — {mb_w:.1f} MB"
                detail = {"current": written, "total": written, "unit": "bytes"}
            await progress.aset("ingest", frac, msg, detail=detail)

        try:
            await s3_service.download_object(
                key,
                dest,
                max_bytes=clip_spec.MAX_BYTES if not balltrack else s3_service.MAX_BYTES,
                on_progress=on_dl,
            )
            log.debug("download %s -> %s bytes=%s", key, dest, dest.stat().st_size if dest.is_file() else 0)
        except ValueError as exc:
            log.debug("download failed job_id=%s key=%s err=%s", job_id, key, type(exc).__name__)
            msg = str(exc).lower()
            if not balltrack and "too large" in msg:
                raise ValueError(clip_spec.MSG_SIZE) from exc
            if not balltrack and "empty" in msg:
                raise ValueError(clip_spec.MSG_EMPTY) from exc
            raise
        return dest
    if local_path:
        path = Path(local_path)
        if path.is_file():
            log.debug("fetch local path job_id=%s bytes=%s", job_id, path.stat().st_size)
            return path
    log.debug("fetch missing source job_id=%s key=%s local=%s", job_id, key or None, local_path)
    raise FileNotFoundError("Clip has no S3 source_key")


async def _store_compressed(local: Path, record_id: str, *, balltrack: bool) -> str | None:
    if not s3_service.s3_configured():
        log.debug("compressed skip s3_off record_id=%s", record_id)
        return None
    compressed_key = f"compressed/{record_id}.mp4"
    encoded = local.with_name(f"{local.stem}_compressed.mp4")
    try:
        await asyncio.to_thread(s3_service.transcode_playback, local, encoded)
        uploaded = await asyncio.to_thread(
            s3_service.upload_file, encoded, compressed_key, "video/mp4"
        )
    except Exception:
        log.exception("compressed upload failed for %s", record_id)
        return None
    if uploaded:
        if balltrack:
            await bt_repo.update_session(record_id, compressed_key=uploaded)
        else:
            await repo.update_video(record_id, compressed_key=uploaded)
        log.debug("compressed uploaded record_id=%s key=%s", record_id, uploaded)
    return uploaded


async def _run_action(job: dict) -> None:
    job_id = job["_id"]
    await raise_if_cancelled(job_id)
    video = await repo.get_video(job["video_id"])
    if not video:
        log.debug("action video missing job_id=%s video_id=%s", job_id, job.get("video_id"))
        raise ValueError("Video record missing")
    log.debug(
        "action start job_id=%s video_id=%s user_id=%s source_key=%s",
        job_id,
        video["_id"],
        video.get("user_id"),
        video.get("source_key"),
    )
    dest = await _fetch_source_video(
        video.get("source_key"),
        video["_id"],
        job_id,
        balltrack=False,
        local_path=video.get("path"),
    )
    await raise_if_cancelled(job_id)
    await _store_compressed(dest, video["_id"], balltrack=False)
    await raise_if_cancelled(job_id)
    profile = video.get("player_profile") or {}
    await run_analysis_job(
        job_id=job_id,
        video_id=video["_id"],
        video_path=dest,
        player_name=video.get("player_name") or "Bowler",
        meters_per_pixel=profile.get("meters_per_pixel"),
        reference_height_m=profile.get("height_m"),
        player_profile=profile,
        user_id=video.get("user_id"),
    )


async def _run_ballflight(job: dict) -> None:
    job_id = job["_id"]
    await raise_if_cancelled(job_id, get_job=bt_repo.get_job)
    session = await bt_repo.get_session(job["session_id"])
    if not session:
        log.debug("ballflight session missing job_id=%s session_id=%s", job_id, job.get("session_id"))
        raise ValueError("Session record missing")
    log.debug(
        "ballflight start job_id=%s session_id=%s user_id=%s source_key=%s",
        job_id,
        session["_id"],
        session.get("user_id"),
        session.get("source_key"),
    )
    dest = await _fetch_source_video(
        session.get("source_key"),
        session["_id"],
        job_id,
        balltrack=True,
        local_path=session.get("path"),
    )
    await raise_if_cancelled(job_id, get_job=bt_repo.get_job)
    await _store_compressed(dest, session["_id"], balltrack=True)
    await raise_if_cancelled(job_id, get_job=bt_repo.get_job)
    await run_balltrack_job(
        job_id=job_id,
        session_id=session["_id"],
        video_path=dest,
        calibration=session.get("calibration") or {},
    )


async def _requeue_stale_in(collection) -> None:
    cutoff = repo.utcnow() - timedelta(hours=1)
    now = repo.utcnow()
    while True:
        doc = await collection.find_one_and_update(
            {"status": "claimed", "updated_at": {"$lt": cutoff}},
            {
                "$set": {
                    "status": "queued",
                    "message": "Re-queued after a worker stopped",
                    "updated_at": now,
                },
                "$unset": {"quota_day": "", "worker_id": ""},
            },
            return_document=ReturnDocument.BEFORE,
        )
        if not doc:
            break
        await quota.release_lease(doc.get("quota_day"))
        await quota.mark_dirty()


async def _fail_stale_running_in(collection, kind: str) -> None:
    cutoff = repo.utcnow() - timedelta(hours=1)
    now = repo.utcnow()
    filt = {
        "status": {"$in": ["processing", "analyzing"]},
        "updated_at": {"$lt": cutoff},
    }
    docs = await collection.find(
        filt, {"_id": 1, "video_id": 1, "session_id": 1, "quota_day": 1}
    ).to_list(200)
    if not docs:
        return
    await collection.update_many(
        {
            "_id": {"$in": [d["_id"] for d in docs]},
            "status": {"$in": ["processing", "analyzing"]},
        },
        {
            "$set": {
                "status": "failed",
                "stage": "failed",
                "message": "Analysis stopped before this clip finished. Upload it again.",
                "updated_at": now,
            }
        },
    )
    for job in docs:
        # A failed clip gives its daily slot back, exactly as `_finish_slot`
        # does for a job that failed in-process. Without this every stale
        # fail left `quota_days.started` one higher for good, so the cap
        # quietly shrank by one each time a worker died mid-clip.
        try:
            await quota.release_lease(job.get("quota_day"))
        except Exception:
            log.exception("quota release after stale fail %s", job.get("_id"))
        try:
            await original_archive.maybe_archive_for_job(job)
        except Exception:
            log.exception("glacier archive after stale fail %s", job.get("_id"))
        try:
            if kind == "action":
                video_id = job.get("video_id")
                if video_id:
                    video = await repo.get_video(video_id)
                    if video:
                        await asyncio.to_thread(
                            cleanup.cleanup_action_files,
                            str(job["_id"]),
                            video_id,
                            video.get("source_key"),
                            None,
                            None,
                            "failed",
                        )
            else:
                session_id = job.get("session_id")
                if session_id:
                    session = await bt_repo.get_session(session_id)
                    if session:
                        await asyncio.to_thread(
                            cleanup.cleanup_balltrack_files,
                            str(job["_id"]),
                            session_id,
                            session.get("source_key"),
                            None,
                            None,
                            "failed",
                            None,
                        )
        except Exception:
            log.exception("local cleanup after stale fail %s", job.get("_id"))


async def _requeue_stale() -> None:
    db = get_db()
    await _requeue_stale_in(db.jobs)
    await _requeue_stale_in(db["balltrack_jobs"])
    await _fail_stale_running_in(db.jobs, "action")
    await _fail_stale_running_in(db["balltrack_jobs"], "balltrack")
    try:
        await original_archive.sweep_orphan_originals()
    except Exception:
        log.exception("glacier orphan sweep failed")


def _created_at(doc: dict) -> datetime:
    dt = doc.get("created_at")
    if dt is None:
        return datetime.min.replace(tzinfo=timezone.utc)
    if dt.tzinfo is None:
        return dt.replace(tzinfo=timezone.utc)
    return dt


async def _claim_next_fifo(worker_id: str) -> tuple[dict | None, str | None]:
    """Oldest eligible job across both pipelines, after winning a daily slot."""
    now = repo.utcnow()
    action = await repo.peek_oldest_queued(now)
    flight = await bt_repo.peek_oldest_queued(now)
    if action is None and flight is None:
        return None, None
    if action is None:
        pick, kind = flight, "ballflight"
    elif flight is None:
        pick, kind = action, "action"
    elif _created_at(action) <= _created_at(flight):
        pick, kind = action, "action"
    else:
        pick, kind = flight, "ballflight"

    day = await quota.try_lease()
    if day is None:
        return None, "quota_full"
    claim = repo.claim_job if kind == "action" else bt_repo.claim_job
    try:
        job = await claim(pick["_id"], worker_id, day)
    except Exception:
        # The slot was taken but no job carries it — give it back before the
        # error propagates, or the day's cap shrinks by one per Mongo hiccup.
        await quota.release_lease(day)
        raise
    if job is None:
        await quota.release_lease(day)
        log.debug("claim lost job_id=%s kind=%s", pick["_id"], kind)
        return None, None
    return job, kind


async def _finish_slot(job: dict, kind: str) -> None:
    job_id = job["_id"]
    getter = repo.get_job if kind == "action" else bt_repo.get_job
    latest = await getter(job_id)
    status = (latest or {}).get("status")
    day = (latest or {}).get("quota_day") or job.get("quota_day")
    if status == "failed":
        await quota.release_lease(day)
    if status in ("completed", "failed", "cancelled"):
        try:
            await original_archive.maybe_archive_for_job(latest or job)
        except Exception:
            log.exception("glacier archive after %s %s", status, job_id)

        try:
            if kind == "action":
                video_id = (latest or job).get("video_id")
                if video_id:
                    video = await repo.get_video(video_id)
                    result = (latest or job).get("result") or {}
                    await asyncio.to_thread(
                        cleanup.cleanup_action_files,
                        job_id,
                        video_id,
                        (video or {}).get("source_key"),
                        result.get("overlay_key"),
                        result.get("pdf_key"),
                        status,
                    )
            else:
                session_id = (latest or job).get("session_id")
                if session_id:
                    session = await bt_repo.get_session(session_id)
                    artifacts = (session or {}).get("artifacts") or {}
                    deliveries = await bt_repo.list_deliveries_for_session(session_id)
                    clip_keys = [
                        (d.get("artifacts") or {}).get("clip_key")
                        for d in deliveries
                    ]
                    await asyncio.to_thread(
                        cleanup.cleanup_balltrack_files,
                        job_id,
                        session_id,
                        (session or {}).get("source_key"),
                        artifacts.get("overlay_key"),
                        artifacts.get("pitch_map_key"),
                        status,
                        clip_keys,
                    )
        except Exception:
            log.exception("local cleanup after %s %s", status, job_id)
    await quota.mark_dirty()


_IN_FLIGHT = ("claimed", "processing", "analyzing")


async def _has_in_flight() -> bool:
    db = get_db()
    filt = {"status": {"$in": list(_IN_FLIGHT)}}
    if await db.jobs.find_one(filt, {"_id": 1}):
        return True
    if await db["balltrack_jobs"].find_one(filt, {"_id": 1}):
        return True
    return False


async def _has_claimable_queued() -> bool:
    filt = quota.queued_eligible_filter()
    db = get_db()
    if await db.jobs.find_one(filt, {"_id": 1}):
        return True
    if await db["balltrack_jobs"].find_one(filt, {"_id": 1}):
        return True
    return False


async def _idle_stop_reason(kind: str | None) -> str | None:
    return quota.idle_stop_reason(
        quota_full=kind == "quota_full",
        has_in_flight=await _has_in_flight(),
        has_claimable=await _has_claimable_queued(),
    )


def _stop_ec2_instance(instance_id: str, region: str) -> None:
    import boto3

    boto3.client("ec2", region_name=region).stop_instances(InstanceIds=[instance_id])


async def _stop_idle_instance(instance_id: str, region: str) -> bool:
    try:
        await asyncio.to_thread(_stop_ec2_instance, instance_id, region)
        return True
    except Exception:
        log.exception("failed to stop EC2 instance %s in %s", instance_id, region)
        return False


async def worker_loop() -> None:
    settings = get_settings()
    configure_logging(is_production=settings.is_production)
    log.debug("video worker %s boot db=%s", WORKER_ID, settings.mongodb_db)
    for sub in ("videos", "artifacts", "frames", "balltrack"):
        (settings.storage_path / sub).mkdir(parents=True, exist_ok=True)
    if not await ping_mongo():
        raise SystemExit("MongoDB is not reachable")
    log.info("video worker %s waiting for jobs (db=%s)", WORKER_ID, settings.mongodb_db)
    ticks = 0
    idle_since: float | None = None
    idle_stop_s = max(1, int(settings.ec2_idle_stop_seconds))
    instance_id = settings.ec2_instance_id if settings.is_production else None
    while True:
        try:
            ticks += 1
            if ticks % 40 == 1:
                await _requeue_stale()
            job, kind = await _claim_next_fifo(WORKER_ID)
        except asyncio.CancelledError:
            log.info("video worker %s stopped", WORKER_ID)
            return
        except Exception:
            # A transient Mongo error while sweeping or claiming used to end
            # the process (PM2 restarts it, but any in-flight lease was lost
            # with it). Log it and try again after a short pause instead.
            log.exception("worker tick failed; retrying")
            try:
                await asyncio.sleep(3.0)
            except asyncio.CancelledError:
                log.info("video worker %s stopped", WORKER_ID)
                return
            continue
        if kind == "quota_full" or job is None:
            now = time.monotonic()
            if idle_since is None:
                idle_since = now
            if instance_id and (now - idle_since) >= idle_stop_s:
                reason = await _idle_stop_reason(kind)
                if reason is None:
                    idle_since = None
                else:
                    log.info(
                        "idle %.0fs (%s); stopping %s (%s)",
                        idle_stop_s,
                        reason,
                        instance_id,
                        settings.ec2_stop_region,
                    )
                    if await _stop_idle_instance(instance_id, settings.ec2_stop_region):
                        log.info("stop requested; waiting for instance halt")
                        await asyncio.Event().wait()
                    idle_since = now
            wait = 0.8
            if kind == "quota_full":
                wait = min(5.0, quota.seconds_until_utc_midnight())
                if ticks % 12 == 1:
                    log.info(
                        "daily quota full; will idle-stop if still blocked (%.0fs until UTC midnight)",
                        quota.seconds_until_utc_midnight(),
                    )
            try:
                await asyncio.sleep(wait)
            except asyncio.CancelledError:
                log.info("video worker %s stopped", WORKER_ID)
                return
            continue
        idle_since = None
        job_id = job["_id"]
        log.debug("claimed %s job_id=%s", kind, job_id)
        log.info("claimed %s job %s", kind, job_id)
        try:
            if kind == "action":
                await _run_action(job)
            else:
                await _run_ballflight(job)
            log.debug("finished job_id=%s", job_id)
            log.info("finished %s", job_id)
        except JobCancelled:
            log.debug("job cancelled job_id=%s kind=%s", job_id, kind)
            log.info("job %s cancelled; not persisting", job_id)
        except Exception as exc:
            log.exception("job %s failed", job_id)
            fail = dict(
                status="failed",
                stage="failed",
                message=_user_failure_message(exc),
                error=traceback.format_exc(),
            )
            try:
                if kind == "action":
                    await repo.update_job(job_id, **fail)
                else:
                    await bt_repo.update_job(job_id, **fail)
            except Exception:
                log.exception("could not record failure for job %s", job_id)
        try:
            await _finish_slot(job, kind)
        except Exception:
            log.exception("finish-slot failed for job %s", job_id)


def _user_failure_message(exc: BaseException) -> str:
    """The sentence the results page shows for a failed clip.

    Ingest deliberately raises `ValueError` with the same wording as the
    uploader (too large, empty, wrong container) and `ArchivedOriginalError`
    says the clip is no longer available. Those are for the user; anything
    else is an internal fault and gets the generic line so a traceback's
    text never reaches the screen.
    """
    if isinstance(exc, (ValueError, FileNotFoundError, s3_service.ArchivedOriginalError)):
        text = str(exc).strip()
        if text and len(text) <= 300:
            return text
    return "Video worker failed"


def main() -> None:
    try:
        asyncio.run(worker_loop())
    except KeyboardInterrupt:
        pass
    finally:
        try:
            asyncio.run(close_mongo())
        except Exception:
            pass


if __name__ == "__main__":
    main()
