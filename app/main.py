"""HTTP face of the video service — health and worker status only.

Job rows live in the same Mongo database as the website API. Workers claim
`queued` jobs; this process does not run OpenCV.
"""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI

from app.config import get_settings
from app.db.mongo import close_mongo, get_db, ping_mongo

log = logging.getLogger("criclab.video-service")


@asynccontextmanager
async def lifespan(_: FastAPI):
    settings = get_settings()
    for sub in ("videos", "artifacts", "frames", "balltrack"):
        (settings.storage_path / sub).mkdir(parents=True, exist_ok=True)
    if await ping_mongo():
        log.info("video-service mongo ok db=%s", settings.mongodb_db)
    else:
        log.warning("video-service mongo unreachable")
    yield
    await close_mongo()


app = FastAPI(title="CricLab Video Service", version="1.0.0", lifespan=lifespan)


@app.get("/health")
async def health():
    return {"ok": True, "service": "criclab-video-service"}


@app.get("/health/mongo")
async def health_mongo():
    return {"ok": await ping_mongo(), "db": get_settings().mongodb_db}


@app.get("/health/queue")
async def health_queue():
    db = get_db()
    action = await db.jobs.count_documents({"status": "queued"})
    flight = await db["balltrack_jobs"].count_documents({"status": "queued"})
    busy_a = await db.jobs.count_documents({"status": {"$in": ["claimed", "processing", "analyzing"]}})
    busy_f = await db["balltrack_jobs"].count_documents(
        {"status": {"$in": ["claimed", "processing", "analyzing"]}}
    )
    return {
        "queued": {"action": action, "ballflight": flight},
        "running": {"action": busy_a, "ballflight": busy_f},
    }
