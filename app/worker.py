"""Dedicated video workers. Each process claims one Mongo job at a time.

The website API only inserts `status=queued` rows. These processes download
clips, run MediaPipe/OpenCV, and write progress back to the same job documents.
Closing a browser tab cannot stop a claimed job.
"""

from __future__ import annotations

import asyncio
import logging
import os
import socket
import traceback
from datetime import timedelta
from pathlib import Path

from app.balltrack import repo as bt_repo
from app.balltrack.runner import run_balltrack_job
from app.config import get_settings
from app.db import repository as repo
from app.db.mongo import close_mongo, get_db, ping_mongo
from app.pipeline.job_progress import JobReporter
from app.pipeline.runner import run_analysis_job
from app.services import cloudinary_service

log = logging.getLogger("criclab.video-worker")
WORKER_ID = f"{socket.gethostname()}-{os.getpid()}"


async def _ensure_local_video(dest: Path, source_url: str | None, job_id: str, *, balltrack: bool) -> Path:
    if dest.exists() and dest.stat().st_size > 0:
        return dest
    if not source_url:
        raise FileNotFoundError(f"Video missing at {dest} and no source_url")
    dest.parent.mkdir(parents=True, exist_ok=True)
    progress = JobReporter(job_id, update=bt_repo.update_job if balltrack else None)
    await progress.aset("ingest", 0, "Fetching your clip", force=True)

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

    await cloudinary_service.download_to_path(source_url, dest, on_progress=on_dl)
    return dest


async def _run_action(job: dict) -> None:
    job_id = job["_id"]
    video = await repo.get_video(job["video_id"])
    if not video:
        raise ValueError("Video record missing")
    dest = Path(video["path"])
    dest = await _ensure_local_video(dest, video.get("source_url"), job_id, balltrack=False)
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
    session = await bt_repo.get_session(job["session_id"])
    if not session:
        raise ValueError("Session record missing")
    dest = Path(session["path"])
    dest = await _ensure_local_video(dest, session.get("source_url"), job_id, balltrack=True)
    await run_balltrack_job(
        job_id=job_id,
        session_id=session["_id"],
        video_path=dest,
        calibration=session.get("calibration") or {},
    )


async def _requeue_stale() -> None:
    cutoff = repo.utcnow()
    cutoff = cutoff - timedelta(minutes=45)
    now = repo.utcnow()
    stale = {"status": "claimed", "updated_at": {"$lt": cutoff}}
    reset = {
        "$set": {
            "status": "queued",
            "message": "Re-queued after a worker stopped",
            "updated_at": now,
        }
    }
    await get_db().jobs.update_many(stale, reset)
    await get_db()["balltrack_jobs"].update_many(stale, reset)


async def worker_loop() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    settings = get_settings()
    for sub in ("videos", "artifacts", "frames", "balltrack"):
        (settings.storage_path / sub).mkdir(parents=True, exist_ok=True)
    if not await ping_mongo():
        raise SystemExit("MongoDB is not reachable")
    log.info("video worker %s waiting for jobs (db=%s)", WORKER_ID, settings.mongodb_db)
    ticks = 0
    while True:
        ticks += 1
        if ticks % 40 == 1:
            await _requeue_stale()
        job = await repo.claim_next_job(WORKER_ID)
        kind = "action"
        if job is None:
            job = await bt_repo.claim_next_job(WORKER_ID)
            kind = "ballflight"
        if job is None:
            await asyncio.sleep(0.8)
            continue
        job_id = job["_id"]
        log.info("claimed %s job %s", kind, job_id)
        try:
            if kind == "action":
                await _run_action(job)
            else:
                await _run_ballflight(job)
            log.info("finished %s", job_id)
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
