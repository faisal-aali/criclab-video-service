"""Live job progress that GET /jobs can read while a long stage is running.

Pose, ball tracking and overlay render are CPU-bound OpenCV/MediaPipe loops.
If they run on the event loop, Mongo updates (and the poll the UI is making)
sit behind them — which is why the bar sat at 0% then jumped to 60%.

Callers run those loops in a worker thread and emit through `JobReporter`.
Writes are throttled so a 120 fps clip does not hammer Mongo once per frame.
"""

from __future__ import annotations

import asyncio
import threading
import time
from collections.abc import Awaitable, Callable
from typing import Any

from app.db import repository as repo

UpdateFn = Callable[..., Awaitable[None]]

# Inclusive overall-percent bands. Intra-stage work interpolates inside them.
ACTION_BANDS: dict[str, tuple[int, int]] = {
    "queued": (0, 0),
    "claimed": (0, 0),
    "ingest": (1, 8),
    "extract": (8, 12),
    "pose": (12, 48),
    "action": (48, 54),
    "ball": (54, 66),
    "metrics": (66, 70),
    "render": (70, 84),
    "upload": (84, 88),
    "agent": (88, 94),
    "pdf": (94, 99),
    "done": (100, 100),
}

BALLTRACK_BANDS: dict[str, tuple[int, int]] = {
    "queued": (0, 0),
    "claimed": (0, 0),
    "ingest": (1, 8),
    "calibrate": (8, 14),
    "detect": (14, 42),
    "track": (42, 58),
    "metrics": (58, 78),
    "render": (78, 90),
    "agent": (90, 99),
    "done": (100, 100),
}

# Wall-clock share of a typical run — not the same as bar width. Pose is ~12–48%
# of the bar but about half the minutes; a naive elapsed/progress% ETA therefore
# says "2 minutes" at the start of pose and "20 minutes" a few seconds later.
ACTION_STAGE_ORDER = [
    "ingest",
    "extract",
    "pose",
    "action",
    "ball",
    "metrics",
    "render",
    "upload",
    "agent",
    "pdf",
]
ACTION_TIME_SHARE: dict[str, float] = {
    "ingest": 0.08,
    "extract": 0.02,
    "pose": 0.48,
    "action": 0.03,
    "ball": 0.12,
    "metrics": 0.02,
    "render": 0.15,
    "upload": 0.04,
    "agent": 0.04,
    "pdf": 0.02,
}

BALLTRACK_STAGE_ORDER = [
    "ingest",
    "calibrate",
    "detect",
    "track",
    "metrics",
    "render",
    "agent",
]
BALLTRACK_TIME_SHARE: dict[str, float] = {
    "ingest": 0.08,
    "calibrate": 0.04,
    "detect": 0.42,
    "track": 0.12,
    "metrics": 0.08,
    "render": 0.16,
    "agent": 0.10,
}


def clamp_counts(current: int, total: int) -> tuple[int, int]:
    """Inclusive frame ranges must never display as '69 of 68'."""
    total = max(1, int(total or 0))
    current = max(0, min(int(current or 0), total))
    return current, total


def work_done_fraction(
    *,
    stage: str | None,
    progress: int,
    bands: dict[str, tuple[int, int]],
    shares: dict[str, float],
    order: list[str],
) -> float:
    """How much of the *clock* this job has burned, 0–1.

    Interpolates inside the current stage using the live overall percent, so a
    frame-level pose update still moves the ETA instead of sitting on the stage
    start until the next stage begins.
    """
    progress = max(0, min(100, int(progress or 0)))
    if progress >= 100 or stage in ("done", "completed"):
        return 1.0
    done = 0.0
    for name in order:
        share = float(shares.get(name) or 0.0)
        if name == stage:
            lo, hi = bands.get(name, (0, 0))
            span = max(1, int(hi) - int(lo))
            frac = (progress - int(lo)) / span
            return min(1.0, max(0.0, done + share * min(1.0, max(0.0, frac))))
        done += share
    return min(1.0, progress / 100.0)


def lerp(lo: int, hi: int, fraction: float) -> int:
    t = min(1.0, max(0.0, float(fraction)))
    return int(round(lo + (hi - lo) * t))


def in_band(bands: dict[str, tuple[int, int]], stage: str, fraction: float = 0.0) -> int:
    lo, hi = bands.get(stage, (0, 0))
    return lerp(lo, hi, fraction)


class JobReporter:
    """Thread-safe, throttled writer for a job's progress fields."""

    def __init__(
        self,
        job_id: str,
        *,
        bands: dict[str, tuple[int, int]] | None = None,
        update: UpdateFn | None = None,
        min_interval_s: float = 0.35,
    ) -> None:
        self.job_id = job_id
        self.bands = bands or ACTION_BANDS
        self._update = update or repo.update_job
        self._min_interval = min_interval_s
        self._lock = threading.Lock()
        self._last_pct = -1
        self._last_t = 0.0
        try:
            self._loop = asyncio.get_running_loop()
        except RuntimeError:
            self._loop = None

    def pct(self, stage: str, fraction: float = 0.0) -> int:
        return in_band(self.bands, stage, fraction)

    def _should_write(self, progress: int, *, force: bool) -> bool:
        now = time.monotonic()
        with self._lock:
            if not force:
                if progress < self._last_pct:
                    return False
                if progress == self._last_pct and (now - self._last_t) < self._min_interval:
                    return False
            self._last_pct = max(self._last_pct, progress)
            self._last_t = now
            return True

    def _fields(
        self,
        progress: int,
        stage: str,
        message: str,
        *,
        status: str | None,
        detail: dict[str, Any] | None,
    ) -> dict[str, Any]:
        fields: dict[str, Any] = {
            "progress": max(0, min(100, int(progress))),
            "stage": stage,
            "message": message,
            "stage_detail": detail,
        }
        if status:
            fields["status"] = status
        return fields

    async def aset(
        self,
        stage: str,
        fraction: float = 0.0,
        message: str = "",
        *,
        status: str | None = "processing",
        detail: dict[str, Any] | None = None,
        force: bool = False,
        progress: int | None = None,
    ) -> None:
        pct = int(progress) if progress is not None else self.pct(stage, fraction)
        if not self._should_write(pct, force=force):
            return
        await self._update(self.job_id, **self._fields(pct, stage, message, status=status, detail=detail))

    def emit(
        self,
        stage: str,
        fraction: float = 0.0,
        message: str = "",
        *,
        status: str | None = "processing",
        detail: dict[str, Any] | None = None,
        force: bool = False,
        progress: int | None = None,
    ) -> None:
        """Safe from a worker thread. Never waits on the event loop."""
        pct = int(progress) if progress is not None else self.pct(stage, fraction)
        if not self._should_write(pct, force=force):
            return
        if self._loop is None or not self._loop.is_running():
            return
        asyncio.run_coroutine_threadsafe(
            self._update(self.job_id, **self._fields(pct, stage, message, status=status, detail=detail)),
            self._loop,
        )
