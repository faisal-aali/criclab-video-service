"""Speed, line, and length from a pitch-plane homography."""

from __future__ import annotations

from typing import Any

import numpy as np

from app.balltrack.calibrate import STUMP_WIDTH_M, image_to_pitch


def analyze_delivery(
    points: list[dict[str, Any]],
    H: np.ndarray,
    fps: float,
    pitch_length_m: float,
    pitch_width_m: float,
) -> dict[str, Any]:
    pts = sorted(points, key=lambda p: int(p["frame"]))
    mapped: list[dict[str, Any]] = []
    for p in pts:
        length_m, width_m = image_to_pitch(H, p["x"], p["y"])
        mapped.append({**p, "length_m": length_m, "width_m": width_m})

    bounce = _bounce_point(mapped, pitch_length_m)
    pre = [p for p in mapped if bounce and p["frame"] <= bounce["frame"]]
    if len(pre) < 4:
        pre = mapped

    dt = (pre[-1]["frame"] - pre[0]["frame"]) / max(fps, 1.0)
    dist = 0.0
    for a, b in zip(pre, pre[1:]):
        if np.isfinite(a["length_m"]) and np.isfinite(b["length_m"]):
            dist += float(np.hypot(b["length_m"] - a["length_m"], b["width_m"] - a["width_m"]))
    # Homography is pitch-plane; lofted flight is a bit longer.
    # Tiny dt + a noisy hop is how empty clips invented 174 km/h.
    mps = (dist / dt) * 1.18 if dt >= 0.18 and dist >= 6.0 else None
    kmh = mps * 3.6 if mps else None

    line_m = None
    length_m = None
    length_note = "Metres from bowler stumps toward the batter"
    if bounce and np.isfinite(bounce.get("length_m", float("nan"))):
        raw_len = float(bounce["length_m"])
        # Reject — never clamp — a bounce that mapped off the pitch.
        if 0.0 <= raw_len <= float(pitch_length_m):
            length_m = raw_len
            if np.isfinite(bounce.get("width_m", float("nan"))):
                line_m = float(bounce["width_m"] - STUMP_WIDTH_M / 2.0)
        else:
            length_note = (
                f"Bounce mapped to {raw_len:.1f} m, outside a {pitch_length_m:.1f} m pitch — not reported"
            )

    conf = 0.0
    if kmh and bounce:
        conf = min(0.82, 0.2 + 0.025 * len(pts) + 0.15)
    status = "ok" if kmh and bounce else "unavailable"
    return {
        "speed_kmh": {
            "value": round(kmh, 1) if kmh else None,
            "unit": "km/h",
            "confidence": conf,
            "status": status if kmh else "unavailable",
            "estimated": False,
        },
        "line_m": {
            "value": round(line_m, 2) if line_m is not None else None,
            "unit": "m",
            "confidence": conf if line_m is not None else 0,
            "status": "ok" if line_m is not None else "unavailable",
            "note": "Negative is off stump to the left of a right-hand batter",
        },
        "length_m": {
            "value": round(length_m, 2) if length_m is not None else None,
            "unit": "m",
            "confidence": conf if length_m is not None else 0,
            "status": "ok" if length_m is not None else "unavailable",
            "note": length_note,
        },
        "bounce": bounce,
        "n_points": len(pts),
        "start_frame": pts[0]["frame"],
        "end_frame": pts[-1]["frame"],
        "track": mapped,
    }


def _bounce_point(mapped: list[dict[str, Any]], pitch_length_m: float) -> dict[str, Any] | None:
    interior = [
        p
        for p in mapped
        if np.isfinite(p.get("length_m", float("nan"))) and 3.0 <= p["length_m"] <= pitch_length_m - 1.5
    ]
    if len(interior) < 4:
        return None
    # Kink in image velocity ≈ bounce. Never invent a midpoint bounce.
    best = None
    best_score = 0.16
    for i in range(1, len(interior) - 1):
        a, b, c = interior[i - 1], interior[i], interior[i + 1]
        v1 = np.array([b["x"] - a["x"], b["y"] - a["y"]], dtype=np.float64)
        v2 = np.array([c["x"] - b["x"], c["y"] - b["y"]], dtype=np.float64)
        n1 = np.linalg.norm(v1) + 1e-6
        n2 = np.linalg.norm(v2) + 1e-6
        cos = float(np.clip(np.dot(v1, v2) / (n1 * n2), -1, 1))
        score = (1 - cos) + 0.15 * (n1 - n2) / n1
        if score > best_score:
            best_score = score
            best = b
    return best
