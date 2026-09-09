"""Archive finished originals to Glacier Flexible Retrieval.

S3 does not see Mongo job status. This module checks live jobs, then calls
``archive_original``. Failures here must not fail the analysis job.
"""

from __future__ import annotations

import asyncio
import logging
from datetime import datetime, timezone
from typing import Any

from app.db.mongo import get_db
from app.services import s3_service

log = logging.getLogger("criclab.archive")

LIVE_STATUSES = ("queued", "claimed", "processing", "analyzing")
_SWEEP_LIMIT = 40
_UNARCHIVED = {
    "source_key": {"$regex": r"^original/"},
    "$or": [
        {"source_storage_class": {"$exists": False}},
        {"source_storage_class": None},
        {"source_storage_class": {"$nin": list(s3_service.ARCHIVED_STORAGE_CLASSES)}},
    ],
}


def _now() -> datetime:
    return datetime.now(timezone.utc)


async def source_key_has_live_jobs(
    key: str, *, except_job_id: str | None = None
) -> bool:
    """True if any Action or Ball-flight job still needs GetObject on this key."""
    db = get_db()
    videos = await db.videos.find({"source_key": key}, {"_id": 1}).to_list(200)
    sessions = await db.balltrack_sessions.find({"source_key": key}, {"_id": 1}).to_list(200)
    live: dict[str, Any] = {"status": {"$in": list(LIVE_STATUSES)}}
    if except_job_id:
        live["_id"] = {"$ne": except_job_id}
    if videos:
        found = await db.jobs.find_one(
            {**live, "video_id": {"$in": [v["_id"] for v in videos]}},
            {"_id": 1},
        )
        if found:
            return True
    if sessions:
        found = await db["balltrack_jobs"].find_one(
            {**live, "session_id": {"$in": [s["_id"] for s in sessions]}},
            {"_id": 1},
        )
        if found:
            return True
    return False


async def _source_key_for_job(job: dict[str, Any]) -> str | None:
    db = get_db()
    video_id = job.get("video_id")
    if video_id:
        video = await db.videos.find_one({"_id": video_id}, {"source_key": 1})
        return (video or {}).get("source_key")
    session_id = job.get("session_id")
    if session_id:
        session = await db.balltrack_sessions.find_one(
            {"_id": session_id}, {"source_key": 1}
        )
        return (session or {}).get("source_key")
    return None


async def _mark_archived(key: str, storage_class: str) -> None:
    fields = {"source_storage_class": storage_class, "source_archived_at": _now()}
    db = get_db()
    await db.videos.update_many({"source_key": key}, {"$set": fields})
    await db.balltrack_sessions.update_many({"source_key": key}, {"$set": fields})


async def maybe_archive_original(
    key: str | None, *, except_job_id: str | None = None
) -> bool:
    """Transition ``original/…`` to Glacier if no live job still needs it."""
    k = s3_service.original_object_key(key)
    if not k or not s3_service.s3_configured():
        return False
    if await source_key_has_live_jobs(k, except_job_id=except_job_id):
        log.info("skip glacier; a live job still needs %s", k)
        return False
    try:
        stored = await asyncio.to_thread(s3_service.archive_original, k)
    except Exception:
        log.exception("glacier archive failed for %s", k)
        return False
    if not stored:
        return False
    await _mark_archived(k, stored)
    return True


async def maybe_archive_for_job(job: dict[str, Any] | None) -> bool:
    if not job:
        return False
    key = await _source_key_for_job(job)
    return await maybe_archive_original(key, except_job_id=job.get("_id"))


async def sweep_orphan_originals(*, limit: int = _SWEEP_LIMIT) -> int:
    """Archive originals whose jobs are all terminal but still Standard."""
    if not s3_service.s3_configured():
        return 0
    cap = max(1, int(limit))
    db = get_db()
    videos = await db.videos.find(_UNARCHIVED, {"source_key": 1}).to_list(cap)
    sessions = await db.balltrack_sessions.find(_UNARCHIVED, {"source_key": 1}).to_list(cap)
    seen: set[str] = set()
    keys: list[str] = []
    for doc in videos + sessions:
        raw = (doc.get("source_key") or "").strip()
        if raw and raw not in seen:
            seen.add(raw)
            keys.append(raw)
    archived = 0
    for key in keys:
        if await maybe_archive_original(key):
            archived += 1
    return archived
