"""Frame extraction via OpenCV (FFmpeg optional)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import cv2
import numpy as np


def extract_video_meta(video_path: Path) -> dict[str, Any]:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise ValueError(f"Could not open video: {video_path}")
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 30.0)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    cap.release()
    duration_s = frame_count / fps if fps > 0 else 0.0
    return {
        "fps": fps,
        "frame_count": frame_count,
        "width": width,
        "height": height,
        "duration_s": duration_s,
    }


def iter_frames(video_path: Path, max_frames: int | None = 450, stride: int = 1):
    """Yield (frame_index, bgr_frame) sampling up to max_frames."""
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise ValueError(f"Could not open video: {video_path}")

    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    if max_frames and total > max_frames * stride:
        stride = max(1, total // max_frames)

    idx = 0
    yielded = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if idx % stride == 0:
            yield idx, frame
            yielded += 1
            if max_frames and yielded >= max_frames:
                break
        idx += 1
    cap.release()


def save_frame(path: Path, frame: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), frame)


def iter_frame_range(video_path: Path, start: int, end: int):
    """Yield (frame_index, bgr_frame) for a contiguous [start, end] window, stride 1."""
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise ValueError(f"Could not open video: {video_path}")
    start = max(0, int(start))
    end = max(start, int(end))
    idx = 0
    try:
        while idx <= end:
            ok, frame = cap.read()
            if not ok:
                break
            if idx >= start:
                yield idx, frame
            idx += 1
    finally:
        cap.release()
