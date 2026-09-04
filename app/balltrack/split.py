"""Split RANSAC tracks into deliveries using temporal gaps."""

from __future__ import annotations

from typing import Any


def split_deliveries(
    tracks: list[list[dict[str, Any]]],
    fps: float,
    gap_s: float = 1.4,
) -> list[list[dict[str, Any]]]:
    if not tracks:
        return []
    gap = max(8, int((fps or 30) * gap_s))
    merged: list[list[dict[str, Any]]] = []
    current = list(tracks[0])
    for nxt in tracks[1:]:
        if nxt[0]["frame"] - current[-1]["frame"] <= gap:
            current.extend(nxt)
            current.sort(key=lambda p: p["frame"])
        else:
            merged.append(current)
            current = list(nxt)
    merged.append(current)
    return [t for t in merged if len(t) >= 5]
