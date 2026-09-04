"""Action upload gates. KEEP IN SYNC with:

- criclab-web-frontend/src/lib/clipSpec.ts
- criclab-web-backend/app/pipeline/clip_spec.py

Ball flight does not use this spec.
"""

from __future__ import annotations

from pathlib import Path

# --- numeric spec ---
MAX_BYTES = 100 * 1024 * 1024
MIN_BYTES = 1024
MAX_DURATION_S = 10.0
MIN_SHORT_SIDE = 1080
ALLOWED_SUFFIXES = frozenset({".mp4", ".mov"})
FPS_TARGETS = (120.0, 240.0)
FPS_TOLERANCE = 3.0

# --- user-facing sentences (same wording as the Action uploader) ---
MSG_EMPTY = "This file is empty or too small to be a video."
MSG_EXTENSION = "Use a .mp4 or .mov clip."
MSG_SIZE = (
    "Clip must be under 100 MB. Shoot 1080p 120 or 240 fps rather than 4K "
    "if the file is too large."
)
MSG_NO_VIDEO = "This file has no video track."
MSG_FPS_UNREADABLE = (
    "Could not read frame rate — use the phone's slow-mo 120 or 240 fps file, "
    "not a 30 fps export."
)
MSG_VFR = (
    "Variable frame rate is not supported. Export a 120 or 240 fps slow-mo clip."
)
MSG_FPS = (
    "Need slow-motion 120 or 240 fps. Standard 30/60 fps videos are not accepted. "
    "Use the phone's slow-mo mode and don't re-export as 30 fps."
)
MSG_DURATION = "Trim to one delivery, 10 seconds or less."
MSG_LANDSCAPE = "Film in landscape."
MSG_RESOLUTION = "Need 1080p or higher (1920×1080 or 4K)."


def last_suffix(name: str | None) -> str:
    return Path(name or "").suffix.lower()


def suffix_ok(name: str | None) -> bool:
    return last_suffix(name) in ALLOWED_SUFFIXES


def tagged_fps_ok(fps: float) -> bool:
    if not fps or fps <= 0:
        return False
    return any(abs(float(fps) - target) <= FPS_TOLERANCE for target in FPS_TARGETS)


def display_size(
    width: int, height: int, rotation_deg: float | None = None
) -> tuple[int, int]:
    """Swap axes when the track is stored rotated 90/270 degrees."""
    w, h = int(width or 0), int(height or 0)
    if w <= 0 or h <= 0:
        return w, h
    rot = abs(float(rotation_deg or 0.0)) % 360.0
    if 45.0 <= rot <= 135.0 or 225.0 <= rot <= 315.0:
        return h, w
    return w, h


def _finite_positive(value: float | None) -> bool:
    try:
        n = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return False
    return n > 0.0 and n != float("inf")


def evaluate_clip(
    *,
    filename: str | None = None,
    size_bytes: int | None = None,
    duration_s: float | None = None,
    width: int | None = None,
    height: int | None = None,
    fps: float | None = None,
    fps_unreadable: bool = False,
    variable_frame_rate: bool = False,
    has_video_track: bool = True,
    rotation_deg: float | None = None,
) -> list[str]:
    """Return every failing rule. Empty list means the clip may be analysed."""
    errors: list[str] = []
    size = int(size_bytes or 0)
    if size < MIN_BYTES:
        errors.append(MSG_EMPTY)
    if not suffix_ok(filename):
        errors.append(MSG_EXTENSION)
    if size > MAX_BYTES:
        errors.append(MSG_SIZE)
    if not has_video_track:
        errors.append(MSG_NO_VIDEO)
        return errors
    if fps_unreadable or fps is None:
        errors.append(MSG_FPS_UNREADABLE)
    elif variable_frame_rate:
        errors.append(MSG_VFR)
    elif not tagged_fps_ok(float(fps)):
        errors.append(MSG_FPS)
    if duration_s is None or not _finite_positive(duration_s) or float(duration_s) > MAX_DURATION_S:
        errors.append(MSG_DURATION)
    dw, dh = display_size(int(width or 0), int(height or 0), rotation_deg)
    if dw <= 0 or dh <= 0:
        errors.append(MSG_RESOLUTION)
    else:
        if dw <= dh:
            errors.append(MSG_LANDSCAPE)
        if min(dw, dh) < MIN_SHORT_SIDE:
            errors.append(MSG_RESOLUTION)
    return errors


def join_errors(errors: list[str]) -> str:
    return " ".join(errors)
