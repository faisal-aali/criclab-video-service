"""Motion-based ball candidates for a down-the-pitch camera."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import cv2
import numpy as np

from app.pipeline.cv_vision import enhance_bgr


def collect_candidates(
    video_path,
    max_seconds: float = 180.0,
    target_width: int = 640,
    on_progress: Callable[[int, int], None] | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise ValueError("Could not open session video")
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 30.0)
    src_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    src_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    scale = min(1.0, target_width / max(src_w, 1))
    w = max(1, int(src_w * scale))
    h = max(1, int(src_h * scale))
    max_frames = int(max_seconds * fps) if fps > 1 else n
    limit = min(max_frames, n) if n > 0 else max_frames
    step = 2 if fps >= 40 else 1

    bg = cv2.createBackgroundSubtractorMOG2(history=120, varThreshold=28, detectShadows=False)
    kernel = np.ones((3, 3), np.uint8)
    frames: list[dict[str, Any]] = []
    idx = 0
    while idx < limit:
        ok, frame = cap.read()
        if not ok:
            break
        if idx % step == 0:
            small = cv2.resize(frame, (w, h)) if scale < 0.999 else frame
            small = enhance_bgr(small)
            mask = bg.apply(small)
            _, mask = cv2.threshold(mask, 200, 255, cv2.THRESH_BINARY)
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
            cands = _blobs(mask, 1.0 / scale)
            frames.append({"frame": idx, "candidates": cands})
        idx += 1
        if on_progress:
            on_progress(*((idx, limit) if limit > 0 else (idx, idx + 40)))
    cap.release()
    meta = {
        "fps": fps,
        "width": src_w,
        "height": src_h,
        "frame_count": n,
        "duration_s": (n / fps) if fps else 0,
        "scale": scale,
        "step": step,
    }
    return frames, meta


def _blobs(mask: np.ndarray, to_src: float) -> list[dict[str, Any]]:
    h, w = mask.shape[:2]
    min_r = max(2.0, min(w, h) * 0.003)
    max_r = min(w, h) * 0.06
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    out: list[dict[str, Any]] = []
    for c in contours:
        area = cv2.contourArea(c)
        if area < 8:
            continue
        (cx, cy), radius = cv2.minEnclosingCircle(c)
        if radius < min_r or radius > max_r:
            continue
        circularity = area / (np.pi * radius * radius + 1e-6)
        bx, by, bw, bh = cv2.boundingRect(c)
        thin = float(min(bw, bh))
        long = float(max(bw, bh))
        aspect = long / max(thin, 1.0)
        streak = aspect >= 1.7 and circularity >= 0.04
        if circularity < 0.10 and not streak:
            continue
        out.append(
            {
                "x": float(cx * to_src),
                "y": float(cy * to_src),
                "r": float(radius * to_src),
                "score": float(circularity),
                "streak": streak,
            }
        )
    out.sort(key=lambda d: d["score"], reverse=True)
    return out[:6]
