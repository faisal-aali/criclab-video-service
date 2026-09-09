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
                "styles": [str(s) for s in (item.get("styles") or ["any"])] or ["any"],
            }
        )
    return out


def catalog_by_id() -> dict[str, dict[str, Any]]:
    return {d["id"]: d for d in load_catalog()}


def normalize_bowling_style(bowling_style: str | None) -> str | None:
    """Collapse free-text bowling_style into 'pace' or 'spin', or None if unclear.

    None means "don't filter" — we only exclude a drill when we're confident it
    targets the other style, never when the player's style is unknown.
    """
    if not bowling_style:
        return None
    s = str(bowling_style).strip().lower()
    if any(k in s for k in ("spin", "leg_break", "leg break", "off_break", "off break", "googly")):
        return "spin"
    if any(k in s for k in ("pace", "fast", "medium", "seam", "swing")):
        return "pace"
    return None


def _style_ok(drill: dict[str, Any], style: str | None) -> bool:
    if not style:
        return True
    styles = drill.get("styles") or ["any"]
    return "any" in styles or style in styles


def allowed_drill_summaries(
    tags: list[str] | None = None,
    *,
    bowling_style: str | None = None,
) -> list[dict[str, Any]]:
    """id / title / tags only — what Gemma is allowed to see.

    Never hands Gemma a drill for the wrong bowling style when the style is known
    (e.g. a spin-technique drill for a pace bowler) — this is a deterministic
    filter, not something left to the model's judgement.
    """
    wanted = set(tags or [])
    style = normalize_bowling_style(bowling_style)
    catalog = load_catalog()

    def _summary(d: dict[str, Any]) -> dict[str, Any]:
        return {"id": d["id"], "title": d["title"], "tags": d["tags"]}

    rows = [_summary(d) for d in catalog if (not wanted or (wanted & set(d["tags"]))) and _style_ok(d, style)]
    if not rows:
        # Relax the tag filter before the style filter — style correctness matters more
        # than tag overlap, so we never fall all the way back to an unfiltered catalog
        # when the player's bowling style is known.
        rows = [_summary(d) for d in catalog if _style_ok(d, style)]
    if not rows:
        rows = [_summary(d) for d in catalog]
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


def fallback_picks(
    tags: list[str],
    *,
    limit: int = 3,
    bowling_style: str | None = None,
) -> list[dict[str, Any]]:
    catalog = load_catalog()
    style = normalize_bowling_style(bowling_style)
    candidates = [d for d in catalog if _style_ok(d, style)] or catalog
    by_id = catalog_by_id()
    picked: list[str] = []
    for tag in tags:
        for d in candidates:
            if tag in d["tags"] and d["id"] not in picked:
                picked.append(d["id"])
                break
        if len(picked) >= limit:
            break
    if not picked:
        picked = [d["id"] for d in candidates[:limit]]
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
    bowling_style: str | None = None,
) -> list[dict[str, Any]]:
    """Keep only catalog IDs. Drop Gemma-invented drill_id values and any pick that
    targets the wrong bowling style — a safety net in case Gemma ignores the style
    instruction in the prompt.
    """
    by_id = catalog_by_id()
    style = normalize_bowling_style(bowling_style)
    hydrated: list[dict[str, Any]] = []
    seen: set[str] = set()
    for raw in picks or []:
        did = str(raw.get("drill_id") or raw.get("id") or "").strip()
        if did not in by_id or did in seen:
            continue
        d = by_id[did]
        if not _style_ok(d, style):
            continue
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
        return fallback_picks(tags, bowling_style=bowling_style)
    hydrated.sort(key=lambda r: r.get("priority") or 99)
    return hydrated[:5]
