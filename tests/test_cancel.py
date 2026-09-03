"""In-flight cancel: progress must not overwrite cancelled; stop at the next stage."""

from __future__ import annotations

import unittest
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

from app.pipeline.cancel import JobCancelled, raise_if_cancelled
from app.pipeline.runner import run_analysis_job
from app.worker import _finish_slot


class RaiseIfCancelledTests(unittest.IsolatedAsyncioTestCase):
    async def test_raises_when_cancelled(self) -> None:
        get_job = AsyncMock(return_value={"_id": "job_1", "status": "cancelled"})
        with self.assertRaises(JobCancelled):
            await raise_if_cancelled("job_1", get_job=get_job)

    async def test_passes_when_processing(self) -> None:
        get_job = AsyncMock(return_value={"_id": "job_1", "status": "processing"})
        await raise_if_cancelled("job_1", get_job=get_job)

    async def test_passes_when_missing(self) -> None:
        await raise_if_cancelled("job_1", get_job=AsyncMock(return_value=None))


class UpdateJobGuardTests(unittest.IsolatedAsyncioTestCase):
    async def test_action_requires_in_flight_status(self) -> None:
        from app.db import repository as repo

        db = MagicMock()
        db.jobs.update_one = AsyncMock()
        with patch("app.db.repository.get_db", return_value=db):
            await repo.update_job("job_1", status="processing", progress=58)
        filt = db.jobs.update_one.await_args.args[0]
        self.assertEqual(filt["_id"], "job_1")
        self.assertEqual(set(filt["status"]["$in"]), {"claimed", "processing", "analyzing"})

    async def test_balltrack_requires_in_flight_status(self) -> None:
        from app.balltrack import repo

        col = MagicMock()
        col.update_one = AsyncMock()
        with patch("app.balltrack.repo._col_jobs", return_value=col):
            await repo.update_job("job_1", status="processing", progress=20)
        filt = col.update_one.await_args.args[0]
        self.assertEqual(set(filt["status"]["$in"]), {"claimed", "processing", "analyzing"})


class FinishSlotTests(unittest.IsolatedAsyncioTestCase):
    async def test_cancelled_archives_without_releasing_slot(self) -> None:
        job = {"_id": "job_1", "quota_day": "2026-09-03", "status": "cancelled"}
        with (
            patch("app.worker.repo.get_job", new=AsyncMock(return_value=job)),
            patch("app.worker.quota.release_lease", new=AsyncMock()) as release,
            patch("app.worker.quota.mark_dirty", new=AsyncMock()),
            patch(
                "app.worker.original_archive.maybe_archive_for_job",
                new=AsyncMock(),
            ) as archive,
        ):
            await _finish_slot(job, "action")
        release.assert_not_called()
        archive.assert_awaited_once()

    async def test_failed_releases_slot(self) -> None:
        job = {"_id": "job_1", "quota_day": "2026-09-03", "status": "failed"}
        with (
            patch("app.worker.repo.get_job", new=AsyncMock(return_value=job)),
            patch("app.worker.quota.release_lease", new=AsyncMock()) as release,
            patch("app.worker.quota.mark_dirty", new=AsyncMock()),
            patch(
                "app.worker.original_archive.maybe_archive_for_job",
                new=AsyncMock(),
            ),
        ):
            await _finish_slot(job, "action")
        release.assert_awaited_once_with("2026-09-03")


class ActionRunnerCancelTests(unittest.IsolatedAsyncioTestCase):
    async def test_cancelled_before_extract_does_not_fail_or_persist(self) -> None:
        with (
            patch(
                "app.pipeline.runner.raise_if_cancelled",
                new=AsyncMock(side_effect=JobCancelled),
            ),
            patch("app.pipeline.runner.repo.update_job", new=AsyncMock()) as update,
            patch("app.pipeline.runner.repo.insert_delivery", new=AsyncMock()) as insert,
        ):
            with self.assertRaises(JobCancelled):
                await run_analysis_job(
                    job_id="job_1",
                    video_id="vid_1",
                    video_path=Path("clip.mp4"),
                )
        update.assert_not_called()
        insert.assert_not_called()


class BalltrackCancelTests(unittest.IsolatedAsyncioTestCase):
    async def test_cancelled_deletes_partial_deliveries_and_does_not_fail(self) -> None:
        from app.balltrack.runner import run_balltrack_job

        with (
            patch(
                "app.balltrack.runner.raise_if_cancelled",
                new=AsyncMock(side_effect=JobCancelled),
            ),
            patch("app.balltrack.runner.repo.update_job", new=AsyncMock()) as update,
            patch("app.balltrack.runner.repo.update_session", new=AsyncMock()) as session,
            patch(
                "app.balltrack.runner.repo.delete_deliveries_for_job",
                new=AsyncMock(return_value=2),
            ) as delete,
            patch("app.balltrack.runner.get_settings") as settings,
        ):
            settings.return_value.storage_path = Path(".")
            with self.assertRaises(JobCancelled):
                await run_balltrack_job(
                    job_id="job_1",
                    session_id="ses_1",
                    video_path=Path("clip.mp4"),
                    calibration={},
                )
        delete.assert_awaited_once_with("job_1")
        update.assert_not_called()
        session.assert_not_called()

    async def test_delete_deliveries_for_job_filters_job_id(self) -> None:
        from app.balltrack import repo

        col = MagicMock()
        result = MagicMock()
        result.deleted_count = 2
        col.delete_many = AsyncMock(return_value=result)
        with patch("app.balltrack.repo._col_deliveries", return_value=col):
            n = await repo.delete_deliveries_for_job("job_1")
        self.assertEqual(n, 2)
        col.delete_many.assert_awaited_once_with({"job_id": "job_1"})


if __name__ == "__main__":
    unittest.main()
