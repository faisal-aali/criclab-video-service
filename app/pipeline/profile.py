"""Bowler profile used to scale and interpret a delivery."""

from __future__ import annotations

from datetime import date, datetime
from typing import Any

from fastapi import HTTPException


def height_m_from_ft_in(height_ft: float | None, height_in: float | None) -> float | None:
    if height_ft is None and height_in is None:
        return None
    ft = float(height_ft or 0.0)
    inches = float(height_in or 0.0)
    meters = ft * 0.3048 + inches * 0.0254
    return meters if meters > 0 else None


def age_from_dob(dob: str | None) -> int | None:
    if not dob:
        return None
    try:
        born = datetime.strptime(dob[:10], "%Y-%m-%d").date()
    except ValueError:
        return None
    today = date.today()
    years = today.year - born.year - ((today.month, today.day) < (born.month, born.day))
    if years < 5 or years > 90:
        return None
    return years


def parse_player_profile(
    *,
    player_name: str,
    first_name: str | None,
    last_name: str | None,
    date_of_birth: str | None,
    height_ft: float | None,
    height_in: float | None,
    reference_height_m: float | None,
    weight_lbs: float | None,
    bowling_arm: str | None,
    bowling_style: str | None,
    meters_per_pixel: float | None,
) -> dict[str, Any]:
    first = (first_name or "").strip()
    last = (last_name or "").strip()
    name = " ".join(p for p in (first, last) if p) or (player_name or "").strip() or "Bowler"

    arm = (bowling_arm or "").strip().lower()
    if arm not in {"left", "right"}:
        raise HTTPException(400, "Bowling arm is required (left or right).")

    style = (bowling_style or "").strip().lower()
    if style not in {"pace", "spin", "medium"}:
        raise HTTPException(400, "Bowling style is required (pace, spin, or medium).")

    height_m = reference_height_m if reference_height_m and reference_height_m > 0 else None
    if height_m is None:
        height_m = height_m_from_ft_in(height_ft, height_in)
    if height_m is None or height_m < 1.2 or height_m > 2.3:
        raise HTTPException(400, "Enter a realistic bowler height (about 4 ft to 7 ft 6 in).")

    if weight_lbs is None or weight_lbs < 50 or weight_lbs > 400:
        raise HTTPException(400, "Enter a realistic weight in pounds.")

    dob = (date_of_birth or "").strip() or None
    age = age_from_dob(dob)
    if not dob or age is None:
        raise HTTPException(400, "Enter a valid date of birth.")

    return {
        "player_name": name,
        "first_name": first,
        "last_name": last,
        "date_of_birth": dob,
        "age_years": age,
        "height_m": round(float(height_m), 3),
        "height_ft": height_ft,
        "height_in": height_in,
        "weight_lbs": round(float(weight_lbs), 1),
        "weight_kg": round(float(weight_lbs) * 0.453592, 1),
        "bowling_arm": arm,
        "bowling_style": style,
        "meters_per_pixel": meters_per_pixel if meters_per_pixel and meters_per_pixel > 0 else None,
    }
