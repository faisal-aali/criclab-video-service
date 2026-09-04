"""Action clip_spec gates — KEEP IN SYNC with criclab-web-backend tests."""

from __future__ import annotations

import unittest

from app.pipeline import clip_spec as spec


class TaggedFpsTests(unittest.TestCase):
    def test_pass_120_and_240(self) -> None:
        self.assertTrue(spec.tagged_fps_ok(120))
        self.assertTrue(spec.tagged_fps_ok(240))
        self.assertTrue(spec.tagged_fps_ok(119.88))
        self.assertTrue(spec.tagged_fps_ok(239.76))
        self.assertTrue(spec.tagged_fps_ok(117))
        self.assertTrue(spec.tagged_fps_ok(123))
        self.assertFalse(spec.tagged_fps_ok(116))
        self.assertFalse(spec.tagged_fps_ok(124))

    def test_reject_common_and_high_speed(self) -> None:
        for fps in (24, 25, 30, 50, 60, 90, 100, 144, 200, 480, 960):
            self.assertFalse(spec.tagged_fps_ok(fps), fps)


class DisplaySizeTests(unittest.TestCase):
    def test_identity(self) -> None:
        self.assertEqual(spec.display_size(1920, 1080, 0), (1920, 1080))

    def test_rotates_90(self) -> None:
        self.assertEqual(spec.display_size(1920, 1080, 90), (1080, 1920))
        self.assertEqual(spec.display_size(1920, 1080, -90), (1080, 1920))


class EvaluateClipTests(unittest.TestCase):
    def _ok(self, **overrides):
        base = dict(
            filename="clip.mp4",
            size_bytes=5_000_000,
            duration_s=6.0,
            width=1920,
            height=1080,
            fps=120.0,
        )
        base.update(overrides)
        return spec.evaluate_clip(**base)

    def test_happy_path(self) -> None:
        self.assertEqual(self._ok(), [])

    def test_duration_boundary(self) -> None:
        self.assertEqual(self._ok(duration_s=10.0), [])
        self.assertIn(spec.MSG_DURATION, self._ok(duration_s=10.01))
        self.assertIn(spec.MSG_DURATION, self._ok(duration_s=0))
        self.assertIn(spec.MSG_DURATION, self._ok(duration_s=None))

    def test_resolution_and_landscape(self) -> None:
        self.assertEqual(self._ok(width=1919, height=1080), [])
        self.assertEqual(self._ok(width=1920, height=1080), [])
        self.assertIn(spec.MSG_RESOLUTION, self._ok(width=1920, height=800))
        self.assertIn(spec.MSG_RESOLUTION, self._ok(width=1280, height=720))
        errors = self._ok(width=1080, height=1920)
        self.assertIn(spec.MSG_LANDSCAPE, errors)

    def test_size_and_suffix(self) -> None:
        self.assertIn(spec.MSG_SIZE, self._ok(size_bytes=spec.MAX_BYTES + 1))
        self.assertIn(spec.MSG_EXTENSION, self._ok(filename="clip.webm"))
        self.assertIn(spec.MSG_EXTENSION, self._ok(filename="clip.mp4.exe"))
        self.assertEqual(self._ok(filename="clip.MOV"), [])
        self.assertIn(spec.MSG_EMPTY, self._ok(size_bytes=10))

    def test_fps_unreadable_and_vfr(self) -> None:
        self.assertIn(spec.MSG_FPS_UNREADABLE, self._ok(fps=None, fps_unreadable=True))
        self.assertIn(spec.MSG_VFR, self._ok(variable_frame_rate=True))
        self.assertIn(spec.MSG_FPS, self._ok(fps=30))
        self.assertIn(spec.MSG_FPS, self._ok(fps=480))

    def test_collects_multiple_errors(self) -> None:
        errors = spec.evaluate_clip(
            filename="over.avi",
            size_bytes=spec.MAX_BYTES + 50,
            duration_s=24.0,
            width=1280,
            height=720,
            fps=30.0,
        )
        self.assertGreaterEqual(len(errors), 4)
        self.assertIn(spec.MSG_EXTENSION, errors)
        self.assertIn(spec.MSG_SIZE, errors)
        self.assertIn(spec.MSG_DURATION, errors)
        self.assertIn(spec.MSG_FPS, errors)


if __name__ == "__main__":
    unittest.main()
