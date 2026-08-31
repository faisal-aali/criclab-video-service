"""Ollama agent — reasons over structured metrics only."""

from __future__ import annotations

import json
import re
from typing import Any

import httpx

from app.config import get_settings


_CONTROL_RE = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f]")


def _clean(text: str) -> str:
    return _CONTROL_RE.sub("", text or "").strip()


def _model_installed(configured: str, installed: list[str]) -> bool:
    """Match Ollama tag names with or without the :latest suffix."""
    if configured in installed:
        return True
    base = configured.split(":", 1)[0]
    return any(m == base or m.startswith(f"{base}:") for m in installed)


async def ollama_available() -> dict[str, Any]:
    settings = get_settings()
    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            r = await client.get(f"{settings.ollama_base_url}/api/tags")
            r.raise_for_status()
            models = [m.get("name", "") for m in r.json().get("models", [])]
            has_model = _model_installed(settings.ollama_model, models)
            has_nomic = _model_installed(settings.ollama_embed_model, models)
            return {
                "ok": has_model,
                "configured_model": settings.ollama_model,
                "configured_embed_model": settings.ollama_embed_model,
                "models": models,
                "has_model": has_model,
                "has_nomic": has_nomic,
            }
    except Exception as e:
        return {
            "ok": False,
            "error": str(e),
            "configured_model": settings.ollama_model,
            "models": [],
            "has_model": False,
            "has_nomic": False,
        }


async def generate_text(
    prompt: str,
    system: str | None = None,
    *,
    num_predict: int = 500,
    timeout_s: float = 300.0,
) -> str:
    """Generate once. `timeout_s` is short for interactive callers.

    A delivery report can take minutes and nobody is waiting on the page;
    somebody typing into a chat box is, so that caller passes a low ceiling
    and falls back rather than leaving the request hanging."""
    settings = get_settings()
    if settings.llm_provider == "bedrock":
        from app.agent.bedrock_client import converse

        return await converse(prompt, system=system, max_tokens=num_predict, timeout_s=timeout_s)
    payload: dict[str, Any] = {
        "model": settings.ollama_model,
        "prompt": prompt,
        "stream": False,
        "keep_alive": "10m",
        "options": {"temperature": 0.3, "num_predict": num_predict},
    }
    if system:
        payload["system"] = system
    timeout = httpx.Timeout(timeout_s, connect=10.0)
    async with httpx.AsyncClient(timeout=timeout) as client:
        r = await client.post(f"{settings.ollama_base_url}/api/generate", json=payload)
        r.raise_for_status()
        return (r.json().get("response") or "").strip()


async def generate_text_stream(
    prompt: str,
    system: str | None = None,
    *,
    num_predict: int = 500,
    timeout_s: float = 60.0,
):
    """Yield response text as it is generated, instead of waiting for all of it.

    Ollama's `/api/generate` with `stream: true` returns one JSON object per
    line, each carrying the next slice of `response`. A person watching a chat
    bubble fill in word by word perceives this as far faster than the same
    total generation time delivered as one blob at the end — which is the
    entire point of wiring this through, not a cosmetic touch.

    Yields plain text deltas. Raises on a connection failure or non-2xx status,
    same as `generate_text` — the caller decides the fallback.
    """
    settings = get_settings()
    if settings.llm_provider == "bedrock":
        from app.agent.bedrock_client import converse_stream

        async for piece in converse_stream(
            prompt, system=system, max_tokens=num_predict, timeout_s=timeout_s
        ):
            yield piece
        return
    payload: dict[str, Any] = {
        "model": settings.ollama_model,
        "prompt": prompt,
        "stream": True,
        "keep_alive": "10m",
        "options": {"temperature": 0.3, "num_predict": num_predict},
    }
    if system:
        payload["system"] = system
    timeout = httpx.Timeout(timeout_s, connect=10.0)
    async with httpx.AsyncClient(timeout=timeout) as client:
        async with client.stream("POST", f"{settings.ollama_base_url}/api/generate", json=payload) as r:
            r.raise_for_status()
            async for line in r.aiter_lines():
                if not line.strip():
                    continue
                try:
                    chunk = json.loads(line)
                except ValueError:
                    continue
                piece = chunk.get("response")
                if piece:
                    yield piece
                if chunk.get("done"):
                    break


def get_delivery_metrics(metrics: dict[str, Any]) -> dict[str, Any]:
    """Compact, LLM-friendly view of metrics (no bulky trajectory arrays)."""

    def mv(key: str) -> dict[str, Any] | None:
        m = metrics.get(key)
        if not isinstance(m, dict):
            return None
        status = m.get("status")
        value = m.get("value") if status == "ok" else None
        return {
            "value": value,
            "unit": m.get("unit"),
            "confidence": m.get("confidence"),
            "estimated": m.get("estimated"),
            "status": status,
            "note": m.get("note"),
        }

    scale = metrics.get("scale") or {}
    quality = metrics.get("quality") or {}
    tb = metrics.get("timebase") or {}
    return {
        "throwing_side": metrics.get("throwing_side"),
        "release_frame": metrics.get("release_frame"),
        "ball_speed_kmh": mv("ball_speed_kmh"),
        "ball_speed_mps": mv("ball_speed_mps"),
        "arm_speed_kmh": mv("arm_speed_kmh"),
        "arm_speed_mps": mv("arm_speed_mps"),
        "hand_speed_px_per_frame": mv("hand_speed_px_per_frame"),
        "release_time_ms": mv("release_time_ms"),
        "arm_swing_speed_deg_s": mv("arm_swing_speed_deg_s"),
        "release_height_m": mv("release_height_m"),
        "release_angle_deg": mv("release_angle_deg"),
        "stride_length_pct_height": mv("stride_length_pct_height"),
        "elbow_extension_deg": mv("elbow_extension_deg"),
        "elbow_extension_range_deg": mv("elbow_extension_range_deg"),
        "front_knee_flexion_deg": mv("front_knee_flexion_deg"),
        "hip_shoulder_separation_deg": mv("hip_shoulder_separation_deg"),
        "hip_to_trunk_peak_gap_ms": mv("hip_to_trunk_peak_gap_ms"),
        "line_m": mv("line_m"),
        "length_m": mv("length_m"),
        "delivery_type": metrics.get("delivery_type"),
        "action_legality": metrics.get("action_legality"),
        "speed_consistency": metrics.get("speed_consistency"),
        "timebase": {
            "fps": tb.get("fps"),
            "container_fps": tb.get("container_fps"),
            "slow_motion": tb.get("slow_motion"),
            "source": tb.get("source"),
            "note": tb.get("note"),
        },
        "scores": metrics.get("scores"),
        "sequencing_ok": metrics.get("sequencing_ok"),
        "scale": {"method": scale.get("method"), "calibrated": scale.get("calibrated"), "note": scale.get("note")},
        "quality": {
            "pose_frames": quality.get("pose_frames"),
            "tracking_ok": quality.get("tracking_ok"),
            "calibrated": quality.get("calibrated"),
            "has_front_foot_contact": quality.get("has_front_foot_contact"),
            "camera_view": quality.get("camera_view"),
            "camera_view_note": quality.get("camera_view_note"),
            "speed_view_ok": quality.get("speed_view_ok"),
            "speed_view_from_ball": quality.get("speed_view_from_ball"),
            "capture_fps": quality.get("capture_fps"),
            "slow_motion": quality.get("slow_motion"),
        },
    }


def _ok_speed(m: dict[str, Any], key: str) -> float | None:
    cell = m.get(key) or {}
    if cell.get("status") != "ok":
        return None
    v = cell.get("value")
    if v is None:
        return None
    v = float(v)
    if v < 20.0 or v > 175.0:
        return None
    return v


def compare_deliveries(current: dict[str, Any], previous: list[dict[str, Any]]) -> dict[str, Any]:
    """Ball-speed delta only. Arm speed is a different quantity — never a stand-in."""
    cur_ball = _ok_speed(current, "ball_speed_kmh")
    prev_balls = [s for s in (_ok_speed(p, "ball_speed_kmh") for p in previous) if s is not None]
    ball_delta = (cur_ball - (sum(prev_balls) / len(prev_balls))) if (cur_ball is not None and prev_balls) else None

    cur_arm = _ok_speed(current, "arm_speed_kmh")
    prev_arms = [s for s in (_ok_speed(p, "arm_speed_kmh") for p in previous) if s is not None]
    arm_delta = (cur_arm - (sum(prev_arms) / len(prev_arms))) if (cur_arm is not None and prev_arms) else None

    return {
        "basis": "ball_speed" if cur_ball is not None else None,
        "current_speed_kmh": cur_ball,
        "previous_avg_speed_kmh": (sum(prev_balls) / len(prev_balls)) if prev_balls else None,
        "delta_kmh": ball_delta,
        "previous_count": len(prev_balls),
        "current_arm_speed_kmh": cur_arm,
        "previous_avg_arm_speed_kmh": (sum(prev_arms) / len(prev_arms)) if prev_arms else None,
        "arm_delta_kmh": arm_delta,
        "previous_arm_count": len(prev_arms),
    }


async def generate_report(
    *,
    metrics: dict[str, Any],
    comparison: dict[str, Any] | None = None,
    player_name: str = "Bowler",
    player_profile: dict[str, Any] | None = None,
    allowed_drills: list[dict[str, Any]] | None = None,
    candidate_tags: list[str] | None = None,
) -> dict[str, Any]:
    """Tool: generateReport — coaching narrative + catalog-only drill IDs."""
    from app.coaching.recommend import (
        allowed_drill_summaries,
        hydrate_recommendations,
        weakness_tags,
    )

    profile = player_profile or metrics.get("player_profile") or {}
    tags = list(candidate_tags) if candidate_tags is not None else weakness_tags(metrics)
    drills = allowed_drills if allowed_drills is not None else allowed_drill_summaries(tags)
    tool_payload = {
        "player": {
            "name": profile.get("player_name") or player_name,
            "age_years": profile.get("age_years"),
            "height_m": profile.get("height_m"),
            "weight_lbs": profile.get("weight_lbs"),
            "bowling_arm": profile.get("bowling_arm"),
            "bowling_style": profile.get("bowling_style"),
        },
        "metrics": get_delivery_metrics(metrics),
        "comparison": comparison or {},
        "candidate_tags": tags,
        "allowed_drills": drills,
    }
    system = (
        "You are Cric-Lab, an AI cricket bowling coach. "
        "Measurements come from pose estimation or stump-calibrated ball flight. "
        "Use ONLY the provided JSON; never invent or fill in null values. "
        "You do not measure speed or angles — never invent km/h, metres, or degrees. "
        "Coach to this bowler's age, height, bowling arm, and style (pace/spin/medium). "
        "Never claim radar-grade ball speed unless ball_speed_kmh status is ok — that value is still an estimate. "
        "If ball speed is unavailable or speed_view_ok is false, say so and do not substitute arm/hand speed as ball speed. "
        "If speed_consistency.ok is false, treat the ball figure as a lower bound (the delivery recedes from the camera). "
        "If delivery_type.status is not ok, do not name a pace band. A stated bowling style is not a measurement. "
        "If action_legality.verdict is null, do not call the action legal or illegal — the 15° test was not run. "
        "Never claim true 3D hip/trunk rotation from one camera. "
        "Give cricket-specific coaching from the available angles and timing. "
        "Keep each prose section to 1-3 short sentences. "
        "After CONFIDENCE_NOTE, add RECOMMENDATIONS: a JSON array of objects "
        '{"drill_id","reason","priority"} using ONLY drill_id values from allowed_drills. '
        "Pick 2-3 drills that match candidate_tags. Do not invent ids or URLs. "
        "Respond in plain text with these labeled sections exactly:\n"
        "SUMMARY:\nOBSERVATIONS:\nSTRENGTHS:\nIMPROVEMENTS:\nCONFIDENCE_NOTE:\nRECOMMENDATIONS:"
    )
    prompt = (
        f"Player: {player_name}\n"
        f"Measured data JSON:\n{json.dumps(tool_payload)}\n\n"
        "Write the coaching report sections now."
    )

    try:
        text = _clean(await generate_text(prompt, system=system, num_predict=750))
    except Exception as e:
        text = _fallback_report(metrics, comparison, error=str(e), tags=tags)

    sections = _parse_sections(text)
    recs = hydrate_recommendations(_parse_recommendation_list(sections.get("RECOMMENDATIONS", "")), tags=tags)
    return {
        "raw": text,
        "summary": _clean(sections.get("SUMMARY", text[:500])),
        "observations": _clean(sections.get("OBSERVATIONS", "")),
        "strengths": _clean(sections.get("STRENGTHS", "")),
        "improvements": _clean(sections.get("IMPROVEMENTS", "")),
        "confidence_note": _clean(
            sections.get("CONFIDENCE_NOTE", "Metrics are video-derived estimates.")
        ),
        "recommendations": recs,
        "comparison": comparison or {},
    }


def _parse_sections(text: str) -> dict[str, str]:
    keys = ["SUMMARY", "OBSERVATIONS", "STRENGTHS", "IMPROVEMENTS", "CONFIDENCE_NOTE", "RECOMMENDATIONS"]
    out: dict[str, str] = {}
    current = None
    buf: list[str] = []
    for line in text.splitlines():
        stripped = line.strip()
        matched = None
        for k in keys:
            if stripped.upper().startswith(k + ":") or stripped.upper() == k:
                matched = k
                break
        if matched:
            if current:
                out[current] = "\n".join(buf).strip()
            current = matched
            rest = stripped.split(":", 1)[1].strip() if ":" in stripped else ""
            buf = [rest] if rest else []
        else:
            buf.append(line)
    if current:
        out[current] = "\n".join(buf).strip()
    return out


def _parse_recommendation_list(blob: str) -> list[dict[str, Any]]:
    text = (blob or "").strip()
    if not text:
        return []
    fence = re.search(r"```(?:json)?\s*(.*?)```", text, re.DOTALL | re.IGNORECASE)
    if fence:
        text = fence.group(1).strip()
    start, end = text.find("["), text.rfind("]")
    if start < 0 or end <= start:
        return []
    try:
        data = json.loads(text[start : end + 1])
    except json.JSONDecodeError:
        return []
    if not isinstance(data, list):
        return []
    return [row for row in data if isinstance(row, dict)]


def _fallback_report(
    metrics: dict[str, Any],
    comparison: dict[str, Any] | None,
    error: str,
    tags: list[str] | None = None,
) -> str:
    from app.coaching.recommend import fallback_picks

    ball = metrics.get("ball_speed_kmh") or {}
    ball_ok = ball.get("status") == "ok" and ball.get("value") is not None
    speed = ball.get("value") if ball_ok else None
    angle_m = metrics.get("release_angle_deg") or {}
    angle = angle_m.get("value") if angle_m.get("status") == "ok" else None
    cal = (metrics.get("scale") or {}).get("calibrated")
    if speed is not None:
        speed_txt = f"{speed:.1f} km/h (image-plane estimate)"
    else:
        speed_txt = "unavailable"
    summary = (
        f"Delivery analyzed. Ball speed {speed_txt}; "
        f"release angle {f'{angle:.1f}°' if angle is not None else 'n/a'}."
    )
    conf = (
        "Uncalibrated estimate — validate with radar when possible."
        if not cal
        else "Calibrated scale used; still an estimate, not a speed gun."
    )
    hist = ""
    if comparison and comparison.get("delta_kmh") is not None:
        hist = f"Ball-speed delta vs this bowler's recent average: {comparison['delta_kmh']:+.1f} km/h."
    recs = fallback_picks(tags or [])
    rec_json = json.dumps(
        [{"drill_id": r["drill_id"], "reason": r["reason"], "priority": r["priority"]} for r in recs]
    )
    return (
        f"SUMMARY:\n{summary} {hist}\n\n"
        f"OBSERVATIONS:\nRelease frame={'detected' if metrics.get('release_frame') is not None else 'uncertain'}. "
        f"(LLM unavailable: {error})\n\n"
        "STRENGTHS:\nConsistent tracking window available for review.\n\n"
        "IMPROVEMENTS:\nUse a side-on camera for Action mechanics, or Ball flight (behind-bowler + both wickets) for ICC-style speed.\n\n"
        f"CONFIDENCE_NOTE:\n{conf}\n\n"
        f"RECOMMENDATIONS:\n{rec_json}"
    )
