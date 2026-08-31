"""Pitch-plane homography from two stump boxes (normalized 0–1 of the frame)."""

from __future__ import annotations

from typing import Any

import numpy as np

PITCH_LENGTH_M = 20.12
PITCH_WIDTH_M = 3.05
# Three stumps + gaps ≈ 9 inches. Calibration boxes fit the *wickets*, not the whole pitch.
STUMP_WIDTH_M = 0.2286


def _box_corners_px(box: dict[str, Any], w: int, h: int) -> np.ndarray:
    x = float(box["x"]) * w
    y = float(box["y"]) * h
    bw = float(box["w"]) * w
    bh = float(box["h"]) * h
    # Bottom edge of the stump box sits on the pitch.
    bl = (x, y + bh)
    br = (x + bw, y + bh)
    return np.array([bl, br], dtype=np.float32)


def homography_from_boxes(
    bowler: dict[str, Any],
    batter: dict[str, Any],
    frame_w: int,
    frame_h: int,
    pitch_length_m: float = PITCH_LENGTH_M,
    pitch_width_m: float = PITCH_WIDTH_M,
) -> dict[str, Any]:
    if frame_w < 16 or frame_h < 16:
        raise ValueError("Video frame is too small")
    for box, name in ((bowler, "bowler stumps"), (batter, "batter stumps")):
        if not box or min(float(box.get("w", 0)), float(box.get("h", 0))) < 0.02:
            raise ValueError(f"Calibration box for {name} is too small")

    bow_bl, bow_br = _box_corners_px(bowler, frame_w, frame_h)
    bat_bl, bat_br = _box_corners_px(batter, frame_w, frame_h)

    src = np.array([bow_bl, bow_br, bat_br, bat_bl], dtype=np.float32)
    dst = np.array(
        [
            [0.0, 0.0],
            [0.0, STUMP_WIDTH_M],
            [pitch_length_m, STUMP_WIDTH_M],
            [pitch_length_m, 0.0],
        ],
        dtype=np.float32,
    )
    import cv2

    H, _ = cv2.findHomography(src, dst, method=0)
    if H is None:
        raise ValueError("Could not compute pitch homography — realign both stump sets")
    H_inv = np.linalg.inv(H)
    return {
        "H": H.tolist(),
        "H_inv": H_inv.tolist(),
        "pitch_length_m": pitch_length_m,
        "pitch_width_m": pitch_width_m,
        "stump_width_m": STUMP_WIDTH_M,
        "frame_w": frame_w,
        "frame_h": frame_h,
    }


def image_to_pitch(H: np.ndarray, x: float, y: float) -> tuple[float, float]:
    p = H @ np.array([x, y, 1.0], dtype=np.float64)
    if abs(p[2]) < 1e-9:
        return (float("nan"), float("nan"))
    return float(p[0] / p[2]), float(p[1] / p[2])


def pitch_to_image(H_inv: np.ndarray, length_m: float, width_m: float) -> tuple[float, float]:
    p = H_inv @ np.array([length_m, width_m, 1.0], dtype=np.float64)
    if abs(p[2]) < 1e-9:
        return (float("nan"), float("nan"))
    return float(p[0] / p[2]), float(p[1] / p[2])
