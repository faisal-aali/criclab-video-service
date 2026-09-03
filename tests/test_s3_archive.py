"""Glacier archive helpers and GetObject InvalidObjectState handling."""

from __future__ import annotations

import asyncio
import tempfile
import unittest
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

from app.services import original_archive, s3_service


class _Settings:
    s3_bucket = "criclab-s3-bucket"
    s3_region = "ap-south-1"
    aws_region = "us-east-1"
    aws_access_key_id = None
    aws_secret_access_key = None


class ArchiveOriginalTests(unittest.TestCase):
    def test_rejects_playback_prefixes(self) -> None:
        self.assertIsNone(s3_service.original_object_key("compressed/vid.mp4"))
        self.assertEqual(s3_service.original_object_key("original/u/a.mov"), "original/u/a.mov")

    def test_already_glacier_skips_copy(self) -> None:
        client = MagicMock()
        client.head_object.return_value = {"StorageClass": "GLACIER"}
        with (
            patch("app.services.s3_service.get_settings", return_value=_Settings()),
            patch("app.services.s3_service._s3_client", return_value=client),
        ):
            self.assertEqual(s3_service.archive_original("original/u/a.mov"), "GLACIER")
        client.copy_object.assert_not_called()

    def test_copy_to_glacier_flexible(self) -> None:
        client = MagicMock()
        client.head_object.return_value = {"StorageClass": "STANDARD"}
        with (
            patch("app.services.s3_service.get_settings", return_value=_Settings()),
            patch("app.services.s3_service._s3_client", return_value=client),
        ):
            self.assertEqual(s3_service.archive_original("original/u/a.mov"), "GLACIER")
        self.assertEqual(client.copy_object.call_args.kwargs["StorageClass"], "GLACIER")


class DownloadArchivedTests(unittest.TestCase):
    def test_invalid_object_state_is_archived_error(self) -> None:
        class _ClientError(Exception):
            response = {"Error": {"Code": "InvalidObjectState"}}

        client = MagicMock()
        client.get_object.side_effect = _ClientError("glacier")
        with tempfile.TemporaryDirectory() as tmp:
            dest = Path(tmp) / "clip.mov"
            with (
                patch("app.services.s3_service.get_settings", return_value=_Settings()),
                patch("app.services.s3_service.s3_configured", return_value=True),
                patch("app.services.s3_service._s3_client", return_value=client),
            ):
                with self.assertRaises(s3_service.ArchivedOriginalError):
                    asyncio.run(s3_service.download_object("original/u/a.mov", dest))


class MaybeArchiveTests(unittest.IsolatedAsyncioTestCase):
    async def test_skips_when_live_job(self) -> None:
        with (
            patch("app.services.original_archive.s3_service.s3_configured", return_value=True),
            patch(
                "app.services.original_archive.source_key_has_live_jobs",
                new=AsyncMock(return_value=True),
            ),
            patch("app.services.original_archive.s3_service.archive_original") as copy,
        ):
            self.assertFalse(await original_archive.maybe_archive_original("original/u/a.mov"))
        copy.assert_not_called()


if __name__ == "__main__":
    unittest.main()
