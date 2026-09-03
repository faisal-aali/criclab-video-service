from datetime import datetime, timezone
from typing import Any
from uuid import uuid4

from app.db.mongo import get_db
from app.db.quota import queued_eligible_filter


def new_id(prefix: str) -> str:
    return f"{prefix}_{uuid4().hex[:12]}"


def utcnow() -> datetime:
    return datetime.now(timezone.utc)


async def insert_video(doc: dict[str, Any]) -> str:
    db = get_db()
    await db.videos.insert_one(doc)
    return doc["_id"]


async def insert_job(doc: dict[str, Any]) -> str:
    db = get_db()
    await db.jobs.insert_one(doc)
    return doc["_id"]


async def update_job(job_id: str, **fields: Any) -> None:
    db = get_db()
    now = utcnow()
    fields["updated_at"] = now
    update: dict[str, Any] = {"$set": fields}
    # First processing write stamps started_at; later writes keep the earlier
    # value ($min). ETA then measures analysis, not time spent in the queue.
    if fields.get("status") in (None, "processing", "analyzing"):
        update["$min"] = {"started_at": now}
    await db.jobs.update_one({"_id": job_id}, update)


async def get_job(job_id: str) -> dict[str, Any] | None:
    return await get_db().jobs.find_one({"_id": job_id})


async def get_delivery(delivery_id: str) -> dict[str, Any] | None:
    return await get_db().deliveries.find_one({"_id": delivery_id})


async def insert_delivery(doc: dict[str, Any]) -> str:
    await get_db().deliveries.insert_one(doc)
    return doc["_id"]


async def list_deliveries(
    limit: int = 50, player_name: str | None = None, user_id: str | None = None
) -> list[dict[str, Any]]:
    query: dict[str, Any] = {}
    if player_name:
        query["player_name"] = player_name
    if user_id:
        query["user_id"] = user_id
    cursor = get_db().deliveries.find(query).sort("created_at", -1).limit(limit)
    return await cursor.to_list(length=limit)


async def top_throws(limit: int = 20) -> list[dict[str, Any]]:
    """Fastest Action deliveries with a measured (not estimated) ball speed.

    The public board caps at 20; admin can pass a much larger limit to list
    every ranked throw. `limit` is always applied so a missing cap cannot
    dump the whole collection.
    """
    cap = max(1, int(limit))
    pipeline: list[dict[str, Any]] = [
        {
            "$match": {
                "metrics.ball_speed_kmh.status": "ok",
                "metrics.ball_speed_kmh.value": {"$gt": 0},
            }
        },
        {"$sort": {"metrics.ball_speed_kmh.value": -1, "created_at": 1}},
        {"$limit": cap},
        {
            "$project": {
                "_id": 1,
                "user_id": 1,
                "player_name": 1,
                "created_at": 1,
                "metrics.ball_speed_kmh": 1,
                "metrics.arm_speed_kmh": 1,
                "metrics.delivery_type": 1,
                "metrics.player_profile": 1,
            }
        },
    ]
    return await get_db().deliveries.aggregate(pipeline).to_list(cap)


async def update_video(video_id: str, **fields: Any) -> None:
    await get_db().videos.update_one({"_id": video_id}, {"$set": fields})


async def get_video(video_id: str) -> dict[str, Any] | None:
    return await get_db().videos.find_one({"_id": video_id})


_ACTIVE = ("queued", "claimed", "processing", "analyzing")


async def peek_oldest_queued(now: datetime | None = None) -> dict[str, Any] | None:
    return await get_db().jobs.find_one(queued_eligible_filter(now), sort=[("created_at", 1)])


async def claim_job(job_id: str, worker_id: str, quota_day: str) -> dict[str, Any] | None:
    """Take a specific queued Action job after a quota lease has been won."""
    now = utcnow()
    return await get_db().jobs.find_one_and_update(
        {"_id": job_id, "status": "queued"},
        {
            "$set": {
                "status": "claimed",
                "worker_id": worker_id,
                "quota_day": quota_day,
                "message": "Assigned to a video worker",
                "updated_at": now,
            }
        },
        return_document=True,
    )


async def claim_next_job(worker_id: str) -> dict[str, Any] | None:
    """Atomically take the oldest eligible queued Action job. Safe across worker processes."""
    now = utcnow()
    return await get_db().jobs.find_one_and_update(
        queued_eligible_filter(now),
        {
            "$set": {
                "status": "claimed",
                "worker_id": worker_id,
                "message": "Assigned to a video worker",
                "updated_at": now,
            }
        },
        sort=[("created_at", 1)],
        return_document=True,
    )


async def list_active_jobs(user_id: str, limit: int = 20) -> list[dict[str, Any]]:
    cursor = (
        get_db()
        .jobs.find({"user_id": user_id, "status": {"$in": list(_ACTIVE)}})
        .sort("created_at", -1)
        .limit(limit)
    )
    return await cursor.to_list(length=limit)
