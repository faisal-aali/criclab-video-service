"""Read tagged container metadata for Action clip gates.

Prefers ffprobe (sibling of imageio-ffmpeg's ffmpeg), then `ffmpeg -i`,
then OpenCV. Does not use timebase / gravity recovery.
"""

from __future__ import annotations

import json
import logging
import re
import subprocess
from pathlib import Path
from typing import Any

from app.pipeline import clip_spec, extract

log = logging.getLogger("criclab.clip_probe")

_DURATION = re.compile(r"Duration:\s*(\d+):(\d+):(\d+(?:\.\d+)?)")
_VIDEO_STREAM = re.compile(
    r"Video:.*?(\d{2,5})x(\d{2,5}).*?(\d+(?:\.\d+)?)\s*(?:fps|tbr)",
    re.IGNORECASE | re.DOTALL,
)
_ROT_SIDE = re.compile(r"rotation of\s*(-?\d+(?:\.\d+)?)", re.IGNORECASE)
_ROT_TAG = re.compile(r"rotate\s*:\s*(-?\d+(?:\.\d+)?)", re.IGNORECASE)


def probe_action_file(path: Path, *, filename: str | None = None) -> dict[str, Any]:
    """Kwargs for `clip_spec.evaluate_clip` plus display size fields."""
    size = path.stat().st_size if path.is_file() else 0
    name = filename or path.name
    meta = _ffprobe_json(path) or _ffmpeg_banner(path) or _opencv_meta(path)
    if meta is None:
        return {
            "filename": name,
            "size_bytes": size,
            "duration_s": None,
            "width": 0,
            "height": 0,
            "fps": None,
            "fps_unreadable": True,
            "variable_frame_rate": False,
            "has_video_track": False,
            "rotation_deg": None,
        }
    return {
        "filename": name,
        "size_bytes": size,
        "duration_s": meta.get("duration_s"),
        "width": int(meta.get("width") or 0),
        "height": int(meta.get("height") or 0),
        "fps": meta.get("fps"),
        "fps_unreadable": meta.get("fps") is None,
        "variable_frame_rate": bool(meta.get("variable_frame_rate")),
        "has_video_track": bool(meta.get("has_video_track", True)),
        "rotation_deg": meta.get("rotation_deg"),
    }


def assert_action_clip(path: Path, *, filename: str | None = None) -> None:
    errors = clip_spec.evaluate_clip(**probe_action_file(path, filename=filename))
    if errors:
        raise ValueError(clip_spec.join_errors(errors))


def _ffmpeg_exe() -> str | None:
    try:
        import imageio_ffmpeg

        return imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:
        return None


def _ffprobe_exe(ffmpeg: str) -> str | None:
    probe = Path(ffmpeg).with_name(Path(ffmpeg).name.replace("ffmpeg", "ffprobe"))
    return str(probe) if probe.is_file() else None


def _ratio_fps(value: str | None) -> float | None:
    raw = (value or "").strip()
    if not raw or raw in {"0/0", "N/A"}:
        return None
    if "/" in raw:
        num, den = raw.split("/", 1)
        try:
            n, d = float(num), float(den)
        except ValueError:
            return None
        if d == 0:
            return None
        return n / d
    try:
        fps = float(raw)
    except ValueError:
        return None
    return fps if fps > 0 else None


def _ffprobe_json(path: Path) -> dict[str, Any] | None:
    ffmpeg = _ffmpeg_exe()
    if not ffmpeg:
        return None
    probe = _ffprobe_exe(ffmpeg)
    if not probe:
        return None
    try:
        result = subprocess.run(
            [
                probe,
                "-v",
                "error",
                "-select_streams",
                "v:0",
                "-show_entries",
                "stream=width,height,duration,avg_frame_rate,r_frame_rate,nb_frames,codec_type:stream_tags=rotate:side_data=rotation:format=duration",
                "-of",
                "json",
                str(path),
            ],
            capture_output=True,
            text=True,
            timeout=30,
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    if result.returncode != 0 or not (result.stdout or "").strip():
        return None
    try:
        data = json.loads(result.stdout)
    except json.JSONDecodeError:
        return None
    streams = data.get("streams") or []
    if not streams:
        return {
            "has_video_track": False,
            "fps": None,
            "width": 0,
            "height": 0,
            "duration_s": None,
            "rotation_deg": None,
            "variable_frame_rate": False,
        }
    stream = streams[0]
    r_fps = _ratio_fps(stream.get("r_frame_rate"))
    avg_fps = _ratio_fps(stream.get("avg_frame_rate"))
    fps = r_fps or avg_fps
    vfr = bool(
        r_fps and avg_fps and abs(r_fps - avg_fps) > clip_spec.FPS_TOLERANCE
    )
    duration = _float_or_none(stream.get("duration"))
    if duration is None:
        duration = _float_or_none((data.get("format") or {}).get("duration"))
    rotation = _float_or_none((stream.get("tags") or {}).get("rotate"))
    if rotation is None:
        for side in stream.get("side_data_list") or []:
            rotation = _float_or_none(side.get("rotation"))
            if rotation is not None:
                break
    return {
        "has_video_track": True,
        "fps": fps,
        "width": int(stream.get("width") or 0),
        "height": int(stream.get("height") or 0),
        "duration_s": duration,
        "rotation_deg": rotation,
        "variable_frame_rate": vfr,
    }


def _ffmpeg_banner(path: Path) -> dict[str, Any] | None:
    ffmpeg = _ffmpeg_exe()
    if not ffmpeg:
        return None
    try:
        result = subprocess.run(
            [ffmpeg, "-hide_banner", "-i", str(path)],
            capture_output=True,
            text=True,
            timeout=30,
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    banner = (result.stderr or result.stdout or "")
    if "Video:" not in banner:
        if "Duration:" in banner or "Audio:" in banner:
            return {
                "has_video_track": False,
                "fps": None,
                "width": 0,
                "height": 0,
                "duration_s": None,
                "rotation_deg": None,
                "variable_frame_rate": False,
            }
        return None
    dur_m = _DURATION.search(banner)
    duration_s = None
    if dur_m:
        h, m, s = dur_m.group(1), dur_m.group(2), dur_m.group(3)
        duration_s = int(h) * 3600 + int(m) * 60 + float(s)
    vid = _VIDEO_STREAM.search(banner)
    width = height = 0
    fps = None
    if vid:
        width, height = int(vid.group(1)), int(vid.group(2))
        fps = float(vid.group(3))
    rot_m = _ROT_SIDE.search(banner) or _ROT_TAG.search(banner)
    rotation = float(rot_m.group(1)) if rot_m else None
    return {
        "has_video_track": True,
        "fps": fps,
        "width": width,
        "height": height,
        "duration_s": duration_s,
        "rotation_deg": rotation,
        "variable_frame_rate": False,
    }


def _opencv_meta(path: Path) -> dict[str, Any] | None:
    try:
        meta = extract.extract_video_meta(path)
    except Exception:
        log.info("OpenCV could not open %s", path)
        return None
    fps = float(meta.get("fps") or 0) or None
    return {
        "has_video_track": True,
        "fps": fps if fps and fps > 0 else None,
        "width": int(meta.get("width") or 0),
        "height": int(meta.get("height") or 0),
        "duration_s": float(meta.get("duration_s") or 0) or None,
        "rotation_deg": None,
        "variable_frame_rate": False,
    }


def _float_or_none(value: Any) -> float | None:
    if value is None or value == "":
        return None
    try:
        n = float(value)
    except (TypeError, ValueError):
        return None
    if n != n or n == float("inf") or n == float("-inf"):
        return None
    return n
