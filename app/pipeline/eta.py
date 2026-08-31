"""Estimated time remaining for a running analysis job.

Blends two signals, because neither is trustworthy alone:

* **This job's own pace**, measured from how much *clock* it has burned — not
  from the progress bar. Pose is a small slice of the bar and most of the
  minutes; ingest is the reverse. `elapsed / (progress/100)` therefore says
  "~2 minutes" at the start of pose and "~20 minutes" a few seconds later.
* **How long recent jobs of the same kind actually took.** Video length and
  machine load, captured from the last N completed jobs rather than guessed.

Elapsed is counted from `started_at` (first processing write), not `created_at`,
so time spent in the queue does not inflate the remaining estimate.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from app.db.mongo import get_db
from app.pipeline.job_progress import (
    ACTION_BANDS,
    ACTION_STAGE_ORDER,
    ACTION_TIME_SHARE,
    BALLTRACK_BANDS,
    BALLTRACK_STAGE_ORDER,
    BALLTRACK_TIME_SHARE,
    work_done_fraction,
)

# Cold-cache defaults. A 4K/120 fps Action clip is a few minutes of pose, not
# the ~90 s these used to claim.
DEFAULT_TOTAL_SECONDS: dict[str, float] = {
    "action": 180.0,
    "ballflight": 120.0,
}

HISTORY_SAMPLE = 20
MIN_WORK_FOR_LIVE_SIGNAL = 0.06
FLOOR_SECONDS = 8.0
CEILING_SECONDS = 20 * 60


def _aware(dt: datetime | None) -> datetime | None:
    if dt is None:
        return None
    return dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)


def _clock_start(doc: dict[str, Any]) -> datetime | None:
    return _aware(doc.get("started_at")) or _aware(doc.get("created_at"))


async def _historical_average_seconds(collection: str, pipeline: str) -> float | None:
    """Mean processing duration of the last `HISTORY_SAMPLE` completed jobs."""
    db = get_db()
    cursor = (
        db[collection]
        .find(
            {"status": "completed", "updated_at": {"$exists": True}},
            {"created_at": 1, "started_at": 1, "updated_at": 1},
        )
        .sort("updated_at", -1)
        .limit(HISTORY_SAMPLE)
    )
    durations: list[float] = []
    async for doc in cursor:
        start, end = _clock_start(doc), _aware(doc.get("updated_at"))
        if not start or not end:
            continue
        seconds = (end - start).total_seconds()
        if 1.0 <= seconds <= CEILING_SECONDS:
            durations.append(seconds)
    if not durations:
        return None
    return sum(durations) / len(durations)


def _work_done(pipeline: str, job: dict[str, Any]) -> float:
    progress = int(job.get("progress") or 0)
    stage = job.get("stage")
    if pipeline == "ballflight":
        return work_done_fraction(
            stage=stage,
            progress=progress,
            bands=BALLTRACK_BANDS,
            shares=BALLTRACK_TIME_SHARE,
            order=BALLTRACK_STAGE_ORDER,
        )
    return work_done_fraction(
        stage=stage,
        progress=progress,
        bands=ACTION_BANDS,
        shares=ACTION_TIME_SHARE,
        order=ACTION_STAGE_ORDER,
    )


async def estimate_eta_seconds(
    *, collection: str, pipeline: str, job: dict[str, Any]
) -> int | None:
    """Seconds remaining for a processing job, or None while too little is known.

    `collection` is the Mongo collection the job lives in ("jobs" or
    "balltrack_jobs"); `pipeline` picks the default baseline when history is
    thin.
    """
    status = job.get("status")
    if status in ("completed", "failed"):
        return None

    progress = int(job.get("progress") or 0)
    started = _clock_start(job)
    if not started or progress <= 0:
        return None

    now = datetime.now(timezone.utc)
    elapsed = (now - started).total_seconds()
    if elapsed <= 0:
        return None

    historical_total = await _historical_average_seconds(collection, pipeline)
    baseline_total = historical_total or DEFAULT_TOTAL_SECONDS.get(pipeline, 150.0)
    work = _work_done(pipeline, job)

    if work < MIN_WORK_FOR_LIVE_SIGNAL:
        remaining = baseline_total - elapsed
    else:
        live_total_estimate = elapsed / work
        # Trust this job's own pace once a meaningful slice of clock-work is
        # done (mid-pose), not when the bar happens to read 60%.
        live_weight = min(1.0, max(0.0, (work - MIN_WORK_FOR_LIVE_SIGNAL) / 0.40))
        blended_total = baseline_total * (1.0 - live_weight) + live_total_estimate * live_weight
        remaining = blended_total - elapsed

    remaining = max(FLOOR_SECONDS, min(remaining, CEILING_SECONDS))
    return int(round(remaining))
