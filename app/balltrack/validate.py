"""Reject noise tracks that are not a cricket ball in flight."""

from __future__ import annotations

from typing import Any

import numpy as np

from app.balltrack.calibrate import STUMP_WIDTH_M

# Recreational to elite pace. 174 km/h from a random clip is always noise.
MIN_KMH = 45.0
MAX_KMH = 155.0
MIN_FLIGHT_S = 0.22
MAX_FLIGHT_S = 1.80
MIN_PITCH_TRAVEL_M = 7.0
MIN_POINTS = 8


def reject_reason(
    metrics: dict[str, Any],
    fps: float,
    frame_h: int,
    pitch_length_m: float,
    pitch_width_m: float,
) -> str | None:
    track = metrics.get("track") or []
    if len(track) < MIN_POINTS:
        return "track too short"
    dt = (int(metrics["end_frame"]) - int(metrics["start_frame"])) / max(fps, 1.0)
    if dt < MIN_FLIGHT_S:
        return "too brief to be a delivery"
    if dt > MAX_FLIGHT_S:
        return "too long to be a single ball"

    kmh = (metrics.get("speed_kmh") or {}).get("value")
    if kmh is None:
        return "no speed"
    if not (MIN_KMH <= float(kmh) <= MAX_KMH):
        return f"speed {kmh:.0f} km/h is not a plausible delivery"

    finite = [p for p in track if np.isfinite(p.get("length_m", float("nan")))]
    if len(finite) < MIN_POINTS:
        return "could not map the path onto the pitch"

    lengths = np.array([p["length_m"] for p in finite], dtype=np.float64)
    travel = float(lengths[-1] - lengths[0])
    if travel < MIN_PITCH_TRAVEL_M:
        return "path does not travel down the pitch toward the batter"

    # Mostly increasing length (bowler → batter), allow small jitter.
    diffs = np.diff(lengths)
    if diffs.size and float(np.mean(diffs > -0.4)) < 0.65:
        return "path is not a down-the-wicket flight"

    bounce = metrics.get("bounce")
    if not bounce or bounce.get("length_m") is None:
        return "no bounce on the pitch"
    bl = float(bounce["length_m"])
    bw = float(bounce.get("width_m") if bounce.get("width_m") is not None else STUMP_WIDTH_M / 2)
    if not (4.0 <= bl <= pitch_length_m - 1.2):
        return "bounce is not on the pitch"
    if abs(bw - STUMP_WIDTH_M / 2) > pitch_width_m * 0.7:
        return "bounce is off the pitch"

    ys = [p["y"] for p in track]
    # Behind-the-arm: ball should move up the frame (toward striker / smaller y).
    if ys[0] - ys[-1] < frame_h * 0.06:
        return "ball does not move toward the striker stumps"

    return None
