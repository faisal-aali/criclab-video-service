"""Daily processing cap. Counter only — no email, no schedule math.

`quota_days.started` is incremented on claim and decremented on fail / stale
re-queue. The website API reads it to compute expected start times.
"""

from __future__ import annotations

import logging
from datetime import datetime, timedelta, timezone

from pymongo.errors import DuplicateKeyError

from app.config import get_settings
from app.db.mongo import get_db

log = logging.getLogger("criclab.quota")

QUOTA_STATE_ID = "global"


def utcnow() -> datetime:
    return datetime.now(timezone.utc)


def utc_day_id(dt: datetime | None = None) -> str:
    now = dt or utcnow()
    if now.tzinfo is None:
        now = now.replace(tzinfo=timezone.utc)
    return now.astimezone(timezone.utc).strftime("%Y-%m-%d")


def queued_eligible_filter(now: datetime | None = None) -> dict:
    """Queued jobs whose scheduled day has opened (or pre-quota rows with no date)."""
    when = now or utcnow()
    return {
        "status": "queued",
        "$or": [
            {"available_at": {"$lte": when}},
            {"available_at": {"$exists": False}},
            {"available_at": None},
        ],
    }


def seconds_until_utc_midnight(now: datetime | None = None) -> float:
    when = now or utcnow()
    if when.tzinfo is None:
        when = when.replace(tzinfo=timezone.utc)
    when = when.astimezone(timezone.utc)
    nxt = when.replace(hour=0, minute=0, second=0, microsecond=0) + timedelta(days=1)
    return max(1.0, (nxt - when).total_seconds())


def idle_stop_reason(*, quota_full: bool, has_in_flight: bool, has_claimable: bool) -> str | None:
    """Why the worker EC2 may stop, or None to keep running.

    Tomorrow's queued clips are not a reason to stay up — the website API
    starts this instance at 00:00 UTC when they become claimable.
    """
    if has_in_flight:
        return None
    if quota_full:
        return "daily quota full"
    if has_claimable:
        return None
    return "nothing claimable"


async def try_lease() -> str | None:
    """Atomically take one of today's slots. Returns the UTC day id, or None if full."""
    settings = get_settings()
    quota = int(settings.daily_video_quota)
    day = utc_day_id()
    now = utcnow()
    col = get_db().quota_days
    doc = await col.find_one_and_update(
        {"_id": day, "started": {"$lt": quota}},
        {"$inc": {"started": 1}, "$set": {"updated_at": now}},
        return_document=True,
    )
    if doc is not None:
        return day
    existing = await col.find_one({"_id": day})
    if existing is not None:
        return None
    try:
        await col.insert_one({"_id": day, "started": 1, "updated_at": now})
        return day
    except DuplicateKeyError:
        doc = await col.find_one_and_update(
            {"_id": day, "started": {"$lt": quota}},
            {"$inc": {"started": 1}, "$set": {"updated_at": now}},
            return_document=True,
        )
        return day if doc is not None else None


async def release_lease(day: str | None) -> None:
    """Give back a slot taken today. Past days are left as an audit count."""
    if not day:
        return
    today = utc_day_id()
    if day != today:
        return
    result = await get_db().quota_days.find_one_and_update(
        {"_id": day, "started": {"$gt": 0}},
        {"$inc": {"started": -1}, "$set": {"updated_at": utcnow()}},
    )
    if result is None:
        log.warning("quota release found no started slots for %s", day)


async def mark_dirty() -> None:
    await get_db().quota_state.update_one(
        {"_id": QUOTA_STATE_ID},
        {"$set": {"dirty_at": utcnow()}},
        upsert=True,
    )
