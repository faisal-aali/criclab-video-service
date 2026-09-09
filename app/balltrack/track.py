"""RANSAC ballistic tracks for a behind-the-arm camera."""

from __future__ import annotations

import random
from typing import Any

import numpy as np


def build_tracks(
    frames: list[dict[str, Any]],
    frame_w: int,
    frame_h: int,
    fps: float = 30.0,
    min_len: int = 8,
    iterations: int = 400,
) -> list[list[dict[str, Any]]]:
    rng = random.Random(7)
    diag = float(np.hypot(frame_w, frame_h) or 1.0)
    rate = max(float(fps), 1.0)
    min_span = max(5, int(rate * 0.18))
    max_span = int(rate * 1.8)
    tol = diag * 0.022
    with_c = [f for f in frames if f["candidates"]]
    if len(with_c) < min_len:
        return []

    remaining = set(range(len(with_c)))
    tracks: list[list[dict[str, Any]]] = []
    used_frames: set[int] = set()

    for _attempt in range(8):
        pool = [with_c[i] for i in remaining if with_c[i]["frame"] not in used_frames]
        if len(pool) < min_len:
            break
        best: list[dict[str, Any]] = []
        frame_index = [f["frame"] for f in pool]
        cand_by_frame = {f["frame"]: f["candidates"] for f in pool}
        n = len(pool)
        for _ in range(iterations):
            i0 = rng.randint(0, max(0, n - 4))
            window = pool[i0 : min(n, i0 + 24)]
            if len(window) < 3:
                continue
            picks = rng.sample(window, 3)
            picks.sort(key=lambda f: f["frame"])
            ts = [float(f["frame"]) for f in picks]
            if ts[0] == ts[2]:
                continue
            pxs = [rng.choice(f["candidates"]) for f in picks]
            try:
                cx = np.polyfit(ts, [p["x"] for p in pxs], 1)
                cy = np.polyfit(ts, [p["y"] for p in pxs], 2)
            except Exception:
                continue
            inliers: list[dict[str, Any]] = []
            for fr in frame_index:
                px = float(np.polyval(cx, fr))
                py = float(np.polyval(cy, fr))
                best_c = None
                best_d = tol
                for c in cand_by_frame[fr]:
                    d = float(np.hypot(c["x"] - px, c["y"] - py))
                    if d < best_d:
                        best_d = d
                        best_c = c
                if best_c is not None:
                    inliers.append(
                        {
                            "frame": int(fr),
                            "x": float(best_c["x"]),
                            "y": float(best_c["y"]),
                            "r": float(best_c.get("r") or 4),
                        }
                    )
            if len(inliers) > len(best):
                xs = [p["x"] for p in inliers]
                ys = [p["y"] for p in inliers]
                travel = float(np.hypot(xs[-1] - xs[0], ys[-1] - ys[0])) if inliers else 0
                if travel > diag * 0.04:
                    span = inliers[-1]["frame"] - inliers[0]["frame"] if inliers else 0
                    # A real ball is a short burst, not someone walking the whole clip.
                    if min_span <= span <= max_span:
                        dy = inliers[0]["y"] - inliers[-1]["y"]
                        if dy > frame_h * 0.05:
                            best = inliers
        if len(best) < min_len:
            break
        best.sort(key=lambda p: p["frame"])
        tracks.append(best)
        for p in best:
            used_frames.add(int(p["frame"]))

    tracks.sort(key=lambda t: t[0]["frame"])
    return tracks
