"""ffprobe slow-mo tags: presentation 30 fps vs media 120/240."""

from __future__ import annotations

import unittest

from app.pipeline.clip_probe import _meta_from_ffprobe_stream


def _iphone_slomo(*, r: str, avg: str, frames: str, media_s: str, play_s: str):
    stream = {
        "width": 1920,
        "height": 1080,
        "r_frame_rate": r,
        "avg_frame_rate": avg,
        "nb_frames": frames,
        "duration": media_s,
        "time_base": "1/600",
    }
    fmt = {"duration": play_s}
    return _meta_from_ffprobe_stream(stream, fmt)


class IphoneSlowMoFfprobeTests(unittest.TestCase):
    def test_playback_30_media_120_from_frame_count(self) -> None:
        meta = _iphone_slomo(
            r="30/1", avg="30/1", frames="240", media_s="2.0", play_s="8.0"
        )
        self.assertAlmostEqual(meta["fps"], 120.0)
        self.assertFalse(meta["variable_frame_rate"])
        self.assertAlmostEqual(meta["duration_s"], 2.0)

    def test_r_120_avg_30_is_slowmo_not_vfr(self) -> None:
        meta = _iphone_slomo(
            r="120/1", avg="30/1", frames="N/A", media_s="2.0", play_s="8.0"
        )
        self.assertAlmostEqual(meta["fps"], 120.0)
        self.assertFalse(meta["variable_frame_rate"])

    def test_packet_count_when_nb_frames_missing(self) -> None:
        stream = {
            "width": 1920,
            "height": 1080,
            "r_frame_rate": "30/1",
            "avg_frame_rate": "30/1",
            "nb_read_packets": "480",
            "duration_ts": "1200",
            "time_base": "1/600",
        }
        meta = _meta_from_ffprobe_stream(stream, {"duration": "8.0"})
        self.assertAlmostEqual(meta["fps"], 240.0)
        self.assertFalse(meta["variable_frame_rate"])

    def test_plain_30fps_stays_30(self) -> None:
        meta = _iphone_slomo(
            r="30/1", avg="30/1", frames="180", media_s="6.0", play_s="6.0"
        )
        self.assertAlmostEqual(meta["fps"], 30.0)
        self.assertFalse(meta["variable_frame_rate"])


if __name__ == "__main__":
    unittest.main()
