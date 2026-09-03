"""S3 key allowlist and ffmpeg playback argv."""

from __future__ import annotations

import unittest
from pathlib import Path
from unittest.mock import patch

from app.services import s3_service


class ObjectKeyTests(unittest.TestCase):
    def test_prefixes(self) -> None:
        self.assertTrue(s3_service.is_our_object_key("original/u/a.mov"))
        self.assertTrue(s3_service.is_our_object_key("files/job_report.pdf"))
        self.assertFalse(s3_service.is_our_object_key("tmp/a.mp4"))
        self.assertEqual(
            s3_service.s3_endpoint_url("ap-south-1"),
            "https://s3.ap-south-1.amazonaws.com",
        )


class ResolveContentTypeTests(unittest.TestCase):
    def test_from_extension(self) -> None:
        self.assertEqual(
            s3_service.resolve_content_type("overlays/job_overlay.mp4"),
            "video/mp4",
        )
        self.assertEqual(
            s3_service.resolve_content_type("files/job_report.pdf"),
            "application/pdf",
        )
        self.assertEqual(
            s3_service.resolve_content_type("files/job_pitchmap.png"),
            "image/png",
        )
        self.assertEqual(
            s3_service.resolve_content_type("original/u/a.mov", "video/mp4"),
            "video/mp4",
        )


class FfmpegArgvTests(unittest.TestCase):
    def test_playback_flags(self) -> None:
        with patch("app.services.s3_service._ffmpeg_exe", return_value="ffmpeg"):
            argv = s3_service.ffmpeg_argv(Path("in.mov"), Path("out.mp4"))
        joined = " ".join(argv)
        self.assertEqual(argv[0], "ffmpeg")
        self.assertIn("-r", argv)
        self.assertIn("30", argv)
        self.assertIn("1000k", argv)
        self.assertIn("-minrate", argv)
        self.assertIn("nal-hrd=cbr", argv)
        self.assertIn("libx264", argv)
        self.assertIn("1280:720", joined)
        self.assertIn("+faststart", joined)
        self.assertIn("-an", argv)


if __name__ == "__main__":
    unittest.main()
