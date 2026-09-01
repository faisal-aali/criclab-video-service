"""Map measured weaknesses to catalog tags. Never invent URLs.

Read-only snapshot of drills.json for the video worker. Train / admin catalog
HTTP lives in criclab-web-backend; admin edits there do not auto-sync here.
"""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import Any

_CATALOG_PATH = Path(__file__).with_name("drills.json")

LOW_SCORE = 65.0

# Ball-flight: length is metres from bowling stumps toward the batter.
# Stock good length sits ~4–10 m short of the batter (10–16 m from the bowler).
STOCK_LENGTH_MIN_M = 10.0
STOCK_LENGTH_MAX_M = 16.0
# line_m = 0 is middle stump; beyond 30 cm is clearly off the corridor.
LINE_CORRIDOR_M = 0.30


@lru_cache(maxsize=1)
def load_catalog() -> list[dict[str, Any]]:
    with _CATALOG_PATH.open() as f:
        items = json.load(f)
    out = []
    for item in items:
        did = str(item.get("id") or "").strip()
        yt = str(item.get("youtube_id") or "").strip()
        if not did or not yt:
            continue
        out.append(
            {
                "id": did,
                "youtube_id": yt,
                "title": str(item.get("title") or did),
                "tags": [str(t) for t in (item.get("tags") or [])],
            }
        )
    return out


def catalog_by_id() -> dict[str, dict[str, Any]]:
    return {d["id"]: d for d in load_catalog()}


def allowed_drill_summaries(tags: list[str] | None = None) -> list[dict[str, Any]]:
    """id / title / tags only — what Gemma is allowed to see."""
    wanted = set(tags or [])
    rows = []
    for d in load_catalog():
        if wanted and not (wanted & set(d["tags"])):
            continue
        rows.append({"id": d["id"], "title": d["title"], "tags": d["tags"]})
    if not rows:
        rows = [{"id": d["id"], "title": d["title"], "tags": d["tags"]} for d in load_catalog()]
    return rows


def _score(scores: dict[str, Any], key: str) -> float | None:
    v = scores.get(key)
    if v is None:
        return None
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def _metric_ok(metrics: dict[str, Any], key: str) -> dict[str, Any] | None:
    m = metrics.get(key)
    if not isinstance(m, dict):
        return None
    if m.get("status") != "ok" or m.get("value") is None:
        return None
    return m


def _metric_value(row: dict[str, Any], key: str) -> float | None:
    m = row.get(key)
    if isinstance(m, dict):
        if m.get("status") != "ok" or m.get("value") is None:
            return None
        try:
            return float(m["value"])
        except (TypeError, ValueError):
            return None
    if m is None:
        return None
    try:
        return float(m)
    except (TypeError, ValueError):
        return None


def weakness_tags(metrics: dict[str, Any] | None = None, *, extra: list[str] | None = None) -> list[str]:
    """Deterministic tags from Action metrics / scores. Gemma does not choose tags."""
    metrics = metrics or {}
    scores = metrics.get("scores") or {}
    quality = metrics.get("quality") or {}
    tags: list[str] = []

    brace = _score(scores, "front_leg_brace")
    if brace is not None and brace < LOW_SCORE:
        tags.append("front_leg_brace")

    stride = _metric_ok(metrics, "stride_length_pct_height")
    if stride:
        pct = float(stride["value"])
        if pct < 55 or pct > 95:
            tags.append("stride")

    seq = _score(scores, "sequencing")
    if seq is not None and seq < LOW_SCORE:
        tags.append("sequencing")
    if metrics.get("sequencing_ok") is False and "sequencing" not in tags:
        tags.append("sequencing")

    sep = _score(scores, "hip_shoulder_separation")
    if sep is not None and sep < LOW_SCORE:
        tags.append("hip_shoulder")

    rel_h = _metric_ok(metrics, "release_height_m")
    if rel_h:
        h = float(rel_h["value"])
        height_m = (metrics.get("player_profile") or {}).get("height_m")
        if height_m and h < 0.70 * float(height_m):
            tags.append("release_height")
        elif h < 1.4:
            tags.append("release_height")

    arm = _score(scores, "arm_speed")
    if arm is not None and arm < LOW_SCORE:
        tags.append("arm_speed")

    view = quality.get("camera_view")
    if view in ("front_on", "unknown") or quality.get("speed_view_ok") is False:
        tags.append("side_on_setup")

    for t in extra or []:
        if t and t not in tags:
            tags.append(t)

    return tags


def balltrack_tags(analyzed: list[dict[str, Any]] | None) -> list[str]:
    """Deterministic tags from stump-homography line / length. Gemma does not choose tags."""
    tags: list[str] = []
    for row in analyzed or []:
        line = _metric_value(row, "line_m")
        if line is not None and abs(line) > LINE_CORRIDOR_M and "line" not in tags:
            tags.append("line")
        length = _metric_value(row, "length_m")
        if (
            length is not None
            and not (STOCK_LENGTH_MIN_M <= length <= STOCK_LENGTH_MAX_M)
            and "length" not in tags
        ):
            tags.append("length")
    return tags


def fallback_picks(tags: list[str], *, limit: int = 3) -> list[dict[str, Any]]:
    catalog = load_catalog()
    by_id = catalog_by_id()
    picked: list[str] = []
    for tag in tags:
        for d in catalog:
            if tag in d["tags"] and d["id"] not in picked:
                picked.append(d["id"])
                break
        if len(picked) >= limit:
            break
    if not picked:
        picked = [d["id"] for d in catalog[:limit]]
    reasons = {
        "front_leg_brace": "Front-leg brace scored low on this clip — work a block/plant drill.",
        "stride": "Stride length was outside a typical band — groove a repeatable approach.",
        "hip_shoulder": "Hip–shoulder separation was limited on this delivery.",
        "release_height": "Release looked low for this bowler — keep the wrist up through the crease.",
        "sequencing": "The measured sequence did not flow hip → trunk → arm.",
        "line": "Bounce was off the intended line — target off stump.",
        "length": "Length was too full or too short for a stock ball.",
        "arm_speed": "Arm speed at leave-hand was below the heuristic band.",
        "side_on_setup": "This camera angle cannot support truthful speed — re-film side-on.",
    }
    out = []
    for i, did in enumerate(picked[:limit]):
        d = by_id.get(did)
        if not d:
            continue
        tag = next((t for t in d["tags"] if t in tags), (d["tags"][0] if d["tags"] else ""))
        out.append(
            {
                "drill_id": d["id"],
                "youtube_id": d["youtube_id"],
                "title": d["title"],
                "tags": d["tags"],
                "reason": reasons.get(tag, "Selected from the Cric-Lab drill catalog."),
                "priority": i + 1,
                "source": "fallback",
            }
        )
    return out


def hydrate_recommendations(
    picks: list[dict[str, Any]] | None,
    *,
    tags: list[str],
) -> list[dict[str, Any]]:
    """Keep only catalog IDs. Drop Gemma-invented drill_id values."""
    by_id = catalog_by_id()
    hydrated: list[dict[str, Any]] = []
    seen: set[str] = set()
    for raw in picks or []:
        did = str(raw.get("drill_id") or raw.get("id") or "").strip()
        if did not in by_id or did in seen:
            continue
        d = by_id[did]
        try:
            prio = int(raw.get("priority") or (len(hydrated) + 1))
        except (TypeError, ValueError):
            prio = len(hydrated) + 1
        reason = str(raw.get("reason") or "").strip() or "Catalog drill matched to a measured weakness."
        hydrated.append(
            {
                "drill_id": d["id"],
                "youtube_id": d["youtube_id"],
                "title": d["title"],
                "tags": d["tags"],
                "reason": reason,
                "priority": prio,
                "source": "gemma",
            }
        )
        seen.add(did)
    if not hydrated:
        return fallback_picks(tags)
    hydrated.sort(key=lambda r: r.get("priority") or 99)
    return hydrated[:5]
