from __future__ import annotations

from datetime import datetime, timezone
from typing import Any
from uuid import uuid4

from app.db.mongo import get_db


def new_id(prefix: str) -> str:
    return f"{prefix}_{uuid4().hex[:12]}"


def utcnow() -> datetime:
    return datetime.now(timezone.utc)


def _col_sessions():
    return get_db()["balltrack_sessions"]


def _col_deliveries():
    return get_db()["balltrack_deliveries"]


def _col_jobs():
    return get_db()["balltrack_jobs"]


async def insert_session(doc: dict[str, Any]) -> str:
    await _col_sessions().insert_one(doc)
    return doc["_id"]


async def get_session(session_id: str) -> dict[str, Any] | None:
    return await _col_sessions().find_one({"_id": session_id})


async def update_session(session_id: str, **fields: Any) -> None:
    fields["updated_at"] = utcnow()
    await _col_sessions().update_one({"_id": session_id}, {"$set": fields})


async def list_sessions(limit: int = 50, user_id: str | None = None) -> list[dict[str, Any]]:
    query: dict[str, Any] = {"user_id": user_id} if user_id else {}
    cursor = _col_sessions().find(query).sort("created_at", -1).limit(limit)
    return await cursor.to_list(length=limit)


async def insert_delivery(doc: dict[str, Any]) -> str:
    await _col_deliveries().insert_one(doc)
    return doc["_id"]


async def get_delivery(delivery_id: str) -> dict[str, Any] | None:
    return await _col_deliveries().find_one({"_id": delivery_id})


async def list_deliveries_for_session(session_id: str) -> list[dict[str, Any]]:
    cursor = _col_deliveries().find({"session_id": session_id}).sort("index", 1)
    return await cursor.to_list(length=200)


async def insert_job(doc: dict[str, Any]) -> str:
    await _col_jobs().insert_one(doc)
    return doc["_id"]


async def update_job(job_id: str, **fields: Any) -> None:
    now = utcnow()
    fields["updated_at"] = now
    update: dict[str, Any] = {"$set": fields}
    if fields.get("status") in (None, "processing", "analyzing"):
        update["$min"] = {"started_at": now}
    await _col_jobs().update_one({"_id": job_id}, update)


async def get_job(job_id: str) -> dict[str, Any] | None:
    return await _col_jobs().find_one({"_id": job_id})


_ACTIVE = ("queued", "claimed", "processing", "analyzing")


async def claim_next_job(worker_id: str) -> dict[str, Any] | None:
    now = utcnow()
    return await _col_jobs().find_one_and_update(
        {"status": "queued"},
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
        _col_jobs()
        .find({"user_id": user_id, "status": {"$in": list(_ACTIVE)}})
        .sort("created_at", -1)
        .limit(limit)
    )
    return await cursor.to_list(length=limit)
