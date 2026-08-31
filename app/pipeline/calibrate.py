"""Pixel-to-world scale — truth-first.

Physical units (km/h, metres) are ONLY returned when the user provides a real
scale (meters_per_pixel or bowler height). We never silently invent 1.7 m as fact.

Body pixel height should be computed from upright frames (see runner) so crouch
during the delivery does not inflate meters-per-pixel.
"""

from __future__ import annotations

from typing import Any

# Nose-to-ankle vertical distance as a fraction of full standing stature.
NOSE_ANKLE_FRACTION = 0.87


def _full_height_px(body_px_height: float | None) -> float | None:
    if not body_px_height or body_px_height <= 20:
        return None
    return body_px_height / NOSE_ANKLE_FRACTION


def resolve_scale(
    meters_per_pixel: float | None,
    reference_height_m: float | None,
    body_px_height: float | None,
) -> dict[str, Any]:
    """Resolve meters-per-pixel only from user-provided truth.

    Without user scale, ``meters_per_pixel`` is None and physical metrics must
    be reported as unavailable — not fabricated from an assumed height.
    """
    if meters_per_pixel and meters_per_pixel > 0:
        return {
            "meters_per_pixel": float(meters_per_pixel),
            "method": "provided_meters_per_pixel",
            "calibrated": True,
            "confidence": 0.85,
            "body_px_height": body_px_height,
        }

    full_px = _full_height_px(body_px_height)

    if reference_height_m and reference_height_m > 0.5 and full_px:
        mpp = float(reference_height_m) / full_px
        return {
            "meters_per_pixel": mpp,
            "method": "player_height_reference",
            "calibrated": True,
            "confidence": 0.65,
            "body_px_height": body_px_height,
            "reference_height_m": float(reference_height_m),
            "note": "Scale from provided bowler height via upright pose height",
        }

    # No user scale → physical units unavailable (truth-first).
    reason = "Enter bowler height (m) or meters/pixel to compute physical speed and height"
    if not full_px:
        reason = "No upright pose height measured — enter meters/pixel, or use a clearer full-body video plus bowler height"
    return {
        "meters_per_pixel": None,
        "method": "needs_calibration",
        "calibrated": False,
        "confidence": 0.0,
        "body_px_height": body_px_height,
        "note": reason,
    }


def upright_body_px_height(heights: list[float]) -> float | None:
    """Prefer tall (upright) frames — 90th percentile — over crouch-biased median."""
    if not heights:
        return None
    arr = [h for h in heights if h and h > 20]
    if not arr:
        return None
    return float(np_percentile(arr, 90))


def np_percentile(values: list[float], pct: float) -> float:
    import numpy as np

    return float(np.percentile(values, pct))
