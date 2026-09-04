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
from app.db.mongo import close_mongo, get_db, ping_mongo
from app.pipeline import clip_spec
from app.pipeline.cancel import JobCancelled, raise_if_cancelled
from app.pipeline.job_progress import JobReporter
from app.pipeline.runner import run_analysis_job
from app.services import original_archive, s3_service

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
        except ValueError as exc:
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
            return path
    raise FileNotFoundError("Clip has no S3 source_key")


async def _store_compressed(local: Path, record_id: str, *, balltrack: bool) -> str | None:
    if not s3_service.s3_configured():
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
    return uploaded


async def _run_action(job: dict) -> None:
    job_id = job["_id"]
    await raise_if_cancelled(job_id)
    video = await repo.get_video(job["video_id"])
    if not video:
        raise ValueError("Video record missing")
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
        raise ValueError("Session record missing")
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


async def _fail_stale_running_in(collection) -> None:
    cutoff = repo.utcnow() - timedelta(hours=1)
    now = repo.utcnow()
    filt = {
        "status": {"$in": ["processing", "analyzing"]},
        "updated_at": {"$lt": cutoff},
    }
    docs = await collection.find(filt, {"_id": 1, "video_id": 1, "session_id": 1}).to_list(200)
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
        try:
            await original_archive.maybe_archive_for_job(job)
        except Exception:
            log.exception("glacier archive after stale fail %s", job.get("_id"))


async def _requeue_stale() -> None:
    db = get_db()
    await _requeue_stale_in(db.jobs)
    await _requeue_stale_in(db["balltrack_jobs"])
    await _fail_stale_running_in(db.jobs)
    await _fail_stale_running_in(db["balltrack_jobs"])
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
    job = await claim(pick["_id"], worker_id, day)
    if job is None:
        await quota.release_lease(day)
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
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    settings = get_settings()
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
        ticks += 1
        if ticks % 40 == 1:
            await _requeue_stale()
        job, kind = await _claim_next_fifo(WORKER_ID)
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
            await asyncio.sleep(wait)
            continue
        idle_since = None
        job_id = job["_id"]
        log.info("claimed %s job %s", kind, job_id)
        try:
            if kind == "action":
                await _run_action(job)
            else:
                await _run_ballflight(job)
            log.info("finished %s", job_id)
        except JobCancelled:
            log.info("job %s cancelled; not persisting", job_id)
        except Exception:
            log.exception("job %s failed", job_id)
            fail = dict(
                status="failed",
                stage="failed",
                message="Video worker failed",
                error=traceback.format_exc(),
            )
            if kind == "action":
                await repo.update_job(job_id, **fail)
            else:
                await bt_repo.update_job(job_id, **fail)
        await _finish_slot(job, kind)


def main() -> None:
    try:
        asyncio.run(worker_loop())
    finally:
        try:
            asyncio.run(close_mongo())
        except Exception:
            pass


if __name__ == "__main__":
    main()
