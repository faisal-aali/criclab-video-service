"""Core OpenCV vision helpers — enhance, extra ball hypotheses, motion proof.

Metrics still come from physics (pixels × scale × time). These functions
improve *detection* and *reject* tracks that do not move independently on
the actual video (camera shake, trees, locked wrists).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import cv2
import numpy as np


def enhance_bgr(bgr: np.ndarray) -> np.ndarray:
    """CLAHE on L in LAB — recovers a dark/red ball in glare or shade."""
    if bgr is None or bgr.size == 0:
        return bgr
    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    merged = cv2.merge([clahe.apply(l), a, b])
    return cv2.cvtColor(merged, cv2.COLOR_LAB2BGR)


def gray_for_flow(bgr: np.ndarray) -> np.ndarray:
    g = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    return cv2.GaussianBlur(g, (5, 5), 0)


def _flow_working(gray: np.ndarray, max_side: int = 1920) -> tuple[np.ndarray, float, float]:
    """Downscale 4K grays so Lucas–Kanade's 21×21 window still covers the blob."""
    h, w = gray.shape[:2]
    side = max(h, w)
    if side <= max_side:
        return gray, 1.0, 1.0
    s = max_side / float(side)
    nw = max(2, int(round(w * s)) & ~1)
    nh = max(2, int(round(h * s)) & ~1)
    work = cv2.resize(gray, (nw, nh), interpolation=cv2.INTER_AREA)
    return work, nw / float(w), nh / float(h)


def hough_ball_candidates(
    bgr: np.ndarray, max_candidates: int = 8, *, enhance: bool = True
) -> list[dict[str, Any]]:
    """Extra circular hypotheses via Hough gradient (cricket ball ≈ circle).

    Motion-blurred balls often fail this test — contours/MOG2 still lead.
    """
    h, w = bgr.shape[:2]
    src = enhance_bgr(bgr) if enhance else bgr
    gray = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)
    gray = cv2.medianBlur(gray, 5)
    min_r = max(3, int(min(w, h) * 0.004))
    max_r = max(min_r + 2, int(min(w, h) * 0.045))
    circles = cv2.HoughCircles(
        gray,
        cv2.HOUGH_GRADIENT,
        dp=1.2,
        minDist=max(12, min_r * 3),
        param1=90,
        param2=16,
        minRadius=min_r,
        maxRadius=max_r,
    )
    if circles is None:
        return []
    out: list[dict[str, Any]] = []
    for x, y, r in np.round(circles[0]).astype(int):
        if x < 4 or y < 4 or x >= w - 4 or y >= h - 4:
            continue
        out.append({"x": float(x), "y": float(y), "r": float(r), "score": 0.55, "source": "hough"})
        if len(out) >= max_candidates:
            break
    return out


def _lk(
    prev_gray: np.ndarray,
    gray: np.ndarray,
    pts: np.ndarray,
) -> tuple[np.ndarray | None, np.ndarray | None]:
    if pts is None or len(pts) == 0:
        return None, None
    nxt, status, _err = cv2.calcOpticalFlowPyrLK(
        prev_gray,
        gray,
        pts,
        None,
        winSize=(21, 21),
        maxLevel=3,
        criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 20, 0.03),
    )
    return nxt, status


def _background_flow(
    prev_gray: np.ndarray,
    gray: np.ndarray,
    avoid_xy: tuple[float, float] | None,
    avoid_r: float,
) -> tuple[float, float]:
    """Median Lucas–Kanade flow of Shi–Tomasi corners — camera/shake motion."""
    mask = np.ones(prev_gray.shape, dtype=np.uint8) * 255
    if avoid_xy is not None:
        cv2.circle(mask, (int(avoid_xy[0]), int(avoid_xy[1])), int(max(24, avoid_r * 4)), 0, -1)
    corners = cv2.goodFeaturesToTrack(
        prev_gray,
        maxCorners=80,
        qualityLevel=0.02,
        minDistance=16,
        mask=mask,
        blockSize=7,
    )
    if corners is None or len(corners) < 8:
        return 0.0, 0.0
    nxt, status = _lk(prev_gray, gray, corners)
    if nxt is None or status is None:
        return 0.0, 0.0
    ok = status.reshape(-1) == 1
    if int(np.count_nonzero(ok)) < 6:
        return 0.0, 0.0
    flow = (nxt.reshape(-1, 2) - corners.reshape(-1, 2))[ok]
    return float(np.median(flow[:, 0])), float(np.median(flow[:, 1]))


def validate_ball_path_on_video(
    video_path: Path | str,
    path: list[dict[str, Any]],
    frame_w: int,
    frame_h: int,
) -> tuple[bool, str, dict[str, Any]]:
    """Prove the track is a moving object on the pixels, not a tree or handshake.

    For each consecutive pair of track points we:
      1. Lucas–Kanade the blob centre to the next frame
      2. Subtract background (camera) flow from Shi–Tomasi corners
      3. Require independent object flow to agree with the track step

    If too few samples can be measured, we *pass* (do not invent a reject).
    """
    stats: dict[str, Any] = {"samples": 0, "agree": 0, "independent": 0, "method": "lk+gftt"}
    if not path or len(path) < 6:
        return False, "path too short for optical-flow check", stats

    by_fr = {int(p["frame"]): p for p in path}
    frames = sorted(by_fr)
    start, end = frames[0], frames[-1]
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return True, "could not re-open video for flow — skipped", stats

    prev_gray = None
    prev_idx: int | None = None
    idx = 0
    try:
        while idx <= end:
            ok, bgr = cap.read()
            if not ok:
                break
            if idx < start:
                idx += 1
                continue
            gray, sx, sy = _flow_working(gray_for_flow(bgr))
            if (
                prev_gray is not None
                and prev_idx is not None
                and prev_idx in by_fr
                and idx in by_fr
            ):
                a = by_fr[prev_idx]
                b = by_fr[idx]
                expected = np.array(
                    [(b["x"] - a["x"]) * sx, (b["y"] - a["y"]) * sy], dtype=np.float32
                )
                exp_mag = float(np.linalg.norm(expected))
                p0 = np.array([[[a["x"] * sx, a["y"] * sy]]], dtype=np.float32)
                nxt, status = _lk(prev_gray, gray, p0)
                cam = _background_flow(
                    prev_gray,
                    gray,
                    (a["x"] * sx, a["y"] * sy),
                    float(a.get("r") or 8) * (sx + sy) * 0.5,
                )
                cam_v = np.array(cam, dtype=np.float32)
                if nxt is not None and status is not None and int(status.reshape(-1)[0]) == 1:
                    raw = nxt.reshape(2) - p0.reshape(2)
                    obj = raw - cam_v
                    obj_mag = float(np.linalg.norm(obj))
                    stats["samples"] += 1
                    if exp_mag >= 2.0:
                        cos = float(np.dot(obj, expected) / (obj_mag * exp_mag + 1e-6))
                        if cos > 0.25 and obj_mag >= 0.35 * exp_mag:
                            stats["agree"] += 1
                        cam_mag = float(np.linalg.norm(cam_v))
                        if obj_mag > max(1.5, 1.4 * cam_mag):
                            stats["independent"] += 1
                    else:
                        stats["agree"] += 1
            if idx in by_fr:
                prev_gray = gray
                prev_idx = idx
            idx += 1
    finally:
        cap.release()

    n = int(stats["samples"])
    if n < 3:
        return True, "optical-flow samples sparse — not used as a reject", stats
    agree = stats["agree"] / n
    indep = stats["independent"] / n
    stats["agree_frac"] = round(agree, 3)
    stats["independent_frac"] = round(indep, 3)
    if agree < 0.40:
        return False, "Optical flow on the video does not follow this blob — not reporting ball speed", stats
    if indep < 0.30:
        return False, "Tracked point moves with the camera, not as a flying ball", stats
    return True, "optical-flow agrees with track", stats
