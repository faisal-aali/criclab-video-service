"""Honor an in-flight cancel at the next pipeline stage, not mid-CV loop.

The website API sets `status=cancelled`. Progress writes must not overwrite
that, and the next stage must not start. The current pose/ball/render loop
is allowed to finish.
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from typing import Any

from app.db import repository as repo

GetJob = Callable[[str], Awaitable[dict[str, Any] | None]]


class JobCancelled(Exception):
    """User cancelled; do not write failed or persist a delivery."""


async def raise_if_cancelled(job_id: str, *, get_job: GetJob | None = None) -> None:
    getter = get_job or repo.get_job
    job = await getter(job_id)
    if job and job.get("status") == "cancelled":
        raise JobCancelled()
