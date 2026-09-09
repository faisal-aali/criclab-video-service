"""Cric-Lab bowling analysis PDF — SpinLab layout, cricket labels.

Pages:
  1. Bowling mechanics overview — 2×2 labeled event stills + headline tiles
  2. Sequencing & key metrics   — kinematic sequence + scores + extra tiles
  3. Joint angles               — bowling-arm & trunk/leg charts
  4. Joint velocities           — hand-speed + angular-velocity charts
  5. Tabular data               — angles by phase + angular-velocity min/max
  6. Delivery summary           — stride, timing, speeds, scores
  7. AI coach notes             — Gemma narrative + catalog drill links
"""

from __future__ import annotations

import textwrap
from pathlib import Path
from typing import Any

from reportlab.graphics.shapes import Drawing, Line, Rect, String, Circle
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.platypus import (
    Image, PageBreak, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle,
)

from app.pdf import charts

PITCH = colors.HexColor("#0B3D2E")
SEAM = colors.HexColor("#C45C26")
BALL = colors.HexColor("#B91C1C")
MIST = colors.HexColor("#F0FFF4")
GREY = colors.HexColor("#4A5568")
LINE = colors.HexColor("#CBD5E0")
INK = colors.HexColor("#1A1A1A")

BANDS = {
    "ball_speed_kmh": (70, 145, "km/h"),
    "arm_speed_kmh": (40, 120, "km/h"),
    "release_time_ms": (350, 700, "ms"),
    "hip_rotation_speed_deg_s": (200, 900, "deg/s"),
    "trunk_rotation_speed_deg_s": (200, 800, "deg/s"),
    "release_height_m": (1.6, 2.6, "m"),
    "stride_length_pct_height": (40, 75, "%ht"),
    "arm_swing_speed_deg_s": (300, 1600, "deg/s"),
    "elbow_extension_deg": (120, 180, "deg"),
    "front_knee_flexion_deg": (120, 178, "deg"),
    "hip_shoulder_separation_deg": (5, 45, "deg"),
    "release_angle_deg": (20, 160, "deg"),
}

# Page-1 stills: SpinLab's 4 event frames, cricket names.
PAGE1_STILLS = [
    ("front_foot_contact", "FRONT-FOOT CONTACT", ("front_foot_contact", "back_foot_contact")),
    ("hip_rotation", "HIP ROTATION", ("hip_rotation",)),
    ("max_external_rotation", "MAX ARM COCKING", ("max_external_rotation", "arm_horizontal")),
    ("release", "RELEASE", ("release",)),
]


def _ok_val(metrics: dict[str, Any], key: str):
    m = metrics.get(key) or {}
    if m.get("status") != "ok" or m.get("value") is None:
        return None
    return m.get("value")


def _tile_value(metrics: dict[str, Any], key: str):
    m = metrics.get(key) or {}
    if m.get("status") != "ok" or m.get("value") is None:
        # _tile word-wraps up to 3 lines (~24 chars each) — don't pre-truncate
        # the honest reason below what the tile can actually show.
        note = m.get("note") or "Unavailable"
        short = note if len(note) <= 84 else note[:81] + "…"
        return None, short
    return m.get("value"), None


BAND_RED = colors.HexColor("#DC2626")
BAND_YELLOW = colors.HexColor("#EAB308")
BAND_GREEN = colors.HexColor("#16A34A")

# Metrics where the good zone sits in the middle of the band (SpinLab stride tile),
# instead of "higher is better".
CENTERED_BAND_KEYS = {"stride_length_pct_height", "release_angle_deg"}

# Metrics whose band is about magnitude — a signed value pins by |value| so a
# good negative separation doesn't land in the red zone the score contradicts.
ABS_BAND_KEYS = {"hip_shoulder_separation_deg"}


def _fmt_tile_value(value, unit: str) -> str:
    if unit == "m":
        return f"{value:.2f}"
    return f"{value:.0f}"


def _tile(label: str, value, unit: str, lo: float, hi: float,
          reason: str | None = None, centered: bool = False, pin_abs: bool = False) -> Drawing:
    w, h = 116, 86
    d = Drawing(w, h)
    d.add(Rect(0, 0, w, h, rx=8, ry=8, fillColor=MIST, strokeColor=LINE, strokeWidth=0.7))
    d.add(String(10, h - 16, label.upper(), fontName="Helvetica-Bold", fontSize=6.5, fillColor=GREY))
    if value is None:
        d.add(String(10, h - 40, "—", fontName="Helvetica-Bold", fontSize=22, fillColor=GREY))
        if reason:
            lines = textwrap.wrap(reason, width=24, max_lines=3, placeholder="…")
            for i, ln in enumerate(lines):
                d.add(String(10, h - 56 - 9 * i, ln, fontName="Helvetica", fontSize=5.5, fillColor=GREY))
    else:
        vtxt = _fmt_tile_value(value, unit)
        d.add(String(10, h - 44, vtxt, fontName="Helvetica-Bold", fontSize=24, fillColor=PITCH))
        d.add(String(12 + 15 * len(vtxt), h - 44, unit, fontName="Helvetica", fontSize=8, fillColor=SEAM))
        # SpinLab-style zoned reference band with a pin at the measured value.
        bx, by, bw, bh = 10, 14, w - 20, 5
        if centered:
            zones = [(0.0, 0.25, BAND_RED), (0.25, 0.75, BAND_GREEN), (0.75, 1.0, BAND_RED)]
        else:
            zones = [(0.0, 1 / 3, BAND_RED), (1 / 3, 2 / 3, BAND_YELLOW), (2 / 3, 1.0, BAND_GREEN)]
        for z0, z1, zc in zones:
            d.add(Rect(bx + z0 * bw, by, (z1 - z0) * bw, bh, fillColor=zc,
                       strokeColor=None, strokeWidth=0))
        if hi > lo:
            pv = abs(float(value)) if pin_abs else float(value)
            t = max(0.0, min(1.0, (pv - lo) / (hi - lo)))
            mx = bx + t * bw
            d.add(Rect(mx - 1.6, by - 2, 3.2, bh + 4, rx=1.4, ry=1.4,
                       fillColor=colors.white, strokeColor=PITCH, strokeWidth=0.9))
        d.add(String(bx, by - 9, f"{lo:g}", fontName="Helvetica", fontSize=6, fillColor=GREY))
        d.add(String(bx + bw - 12, by - 9, f"{hi:g}", fontName="Helvetica", fontSize=6, fillColor=GREY))
    return d


def _score_ring(label: str, score) -> Drawing:
    w, h = 98, 96
    d = Drawing(w, h)
    cx, cy, r = w / 2, h / 2 + 4, 30
    d.add(Circle(cx, cy, r, fillColor=colors.white, strokeColor=LINE, strokeWidth=6))
    if score is not None:
        frac = max(0.0, min(1.0, float(score) / 100.0))
        col = PITCH if frac >= 0.66 else (SEAM if frac >= 0.4 else BALL)
        d.add(String(cx, cy - 7, f"{score:.0f}", fontName="Helvetica-Bold", fontSize=22, fillColor=col, textAnchor="middle"))
    else:
        d.add(String(cx, cy - 7, "—", fontName="Helvetica-Bold", fontSize=20, fillColor=GREY, textAnchor="middle"))
    d.add(String(cx, 6, label.upper(), fontName="Helvetica-Bold", fontSize=6.5, fillColor=GREY, textAnchor="middle"))
    return d


def _styles():
    ss = getSampleStyleSheet()
    return {
        "kicker": ParagraphStyle(
            "kicker", parent=ss["BodyText"], textColor=SEAM, fontSize=8,
            fontName="Helvetica-Bold", spaceAfter=1,
        ),
        "title": ParagraphStyle(
            "t", parent=ss["Title"], textColor=PITCH, fontSize=18, leading=22,
            spaceAfter=2, alignment=0, fontName="Helvetica-Bold",
        ),
        "sub": ParagraphStyle("s", parent=ss["BodyText"], fontSize=9, textColor=GREY, spaceAfter=8),
        "h2": ParagraphStyle(
            "h2", parent=ss["Heading2"], textColor=PITCH, fontSize=12,
            spaceBefore=6, spaceAfter=5, fontName="Helvetica-Bold",
        ),
        "cap": ParagraphStyle(
            "cap", parent=ss["BodyText"], fontSize=7.5, textColor=GREY,
            fontName="Helvetica-Bold", spaceAfter=2, leading=10,
        ),
        "seq": ParagraphStyle("seq", parent=ss["BodyText"], fontSize=10, textColor=INK, leading=14),
        "body": ParagraphStyle("b", parent=ss["BodyText"], fontSize=9.5, leading=13.5, textColor=INK),
        "small": ParagraphStyle("sm", parent=ss["BodyText"], fontSize=7.5, textColor=GREY, leading=10),
        "missing": ParagraphStyle("miss", parent=ss["BodyText"], fontSize=8, textColor=GREY, leading=11),
    }


def _fmt(v, dp=0, dash="—"):
    if v is None:
        return dash
    return f"{v:.{dp}f}" if isinstance(v, (int, float)) else str(v)


def _dt_ms(phases: dict[str, Any], a: str, b: str, fps: float):
    fa, fb = phases.get(a), phases.get(b)
    if fa is None or fb is None:
        return None
    return (int(fb) - int(fa)) / max(float(fps), 1e-6) * 1000.0


def _resolve_still(stills_dir: Path | None, keys: tuple[str, ...]) -> Path | None:
    if stills_dir is None:
        return None
    for key in keys:
        p = stills_dir / f"{key}.jpg"
        if p.exists():
            return p
    return None


def _frame_cell(caption: str, path: Path | None, st) -> Table:
    heading = Paragraph(caption, st["cap"])
    if path is not None and path.exists():
        body: Any = Image(str(path), width=85 * mm, height=48 * mm)
    else:
        body = _missing_still()
    inner = Table([[heading], [body]], colWidths=[85 * mm])
    inner.setStyle(TableStyle([
        ("LEFTPADDING", (0, 0), (-1, -1), 0),
        ("RIGHTPADDING", (0, 0), (-1, -1), 0),
        ("TOPPADDING", (0, 0), (-1, -1), 1),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 2),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
    ]))
    return inner


def _missing_still() -> Drawing:
    w, h = 240, 136
    d = Drawing(w, h)
    d.add(Rect(0, 0, w, h, rx=6, ry=6, fillColor=MIST, strokeColor=LINE, strokeWidth=0.7))
    d.add(String(18, h / 2 + 6, "NOT SEEN ON THIS CLIP", fontName="Helvetica-Bold", fontSize=8, fillColor=GREY))
    d.add(String(18, h / 2 - 10, "This phase was not detected.", fontName="Helvetica", fontSize=7, fillColor=GREY))
    return d


def _tile_row(metrics: dict[str, Any], items: list[tuple[str, str]]) -> Table:
    cells = []
    for label, key in items:
        lo, hi, unit = BANDS.get(key, (0, 100, ""))
        value, reason = _tile_value(metrics, key)
        cells.append(_tile(label, value, unit, lo, hi, reason=reason,
                           centered=key in CENTERED_BAND_KEYS,
                           pin_abs=key in ABS_BAND_KEYS))
    while len(cells) < 4:
        cells.append("")
    t = Table([cells], colWidths=[122] * 4, hAlign="LEFT")
    t.setStyle(TableStyle([("TOPPADDING", (0, 0), (-1, -1), 2), ("BOTTOMPADDING", (0, 0), (-1, -1), 2)]))
    return t


def _section_header(st, kicker: str, title: str, player: str, date_str: str):
    return [
        Paragraph(kicker, st["kicker"]),
        Paragraph(title, st["title"]),
        Paragraph(f"Player: <b>{player}</b> &nbsp;·&nbsp; {date_str}", st["sub"]),
    ]


def build_pdf(
    *,
    out_path: Path,
    player_name: str,
    delivery_id: str,
    metrics: dict[str, Any],
    analysis: dict[str, Any],
    release_still: Path | None = None,
    date_str: str = "",
    chart_dir: Path | None = None,
    stills_dir: Path | None = None,
) -> Path:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    chart_dir = chart_dir or out_path.parent
    date_str = date_str or delivery_id
    st = _styles()
    doc = SimpleDocTemplate(
        str(out_path), pagesize=A4,
        leftMargin=16 * mm, rightMargin=16 * mm, topMargin=16 * mm, bottomMargin=14 * mm,
    )
    story: list[Any] = []
    phases = metrics.get("phases") or {}
    labels = metrics.get("phase_labels") or {}
    fps = float(metrics.get("fps") or 30.0)
    scores = metrics.get("scores") or {}
    profile = metrics.get("player_profile") or {}

    # ---------------- Page 1: stills + headline tiles ----------------
    story += _section_header(st, "CRICLAB BOWLING REPORT", "BOWLING MECHANICS OVERVIEW", player_name, date_str)

    still_cells = []
    for _key, caption, keys in PAGE1_STILLS:
        still_cells.append(_frame_cell(caption, _resolve_still(stills_dir, keys), st))
    still_tbl = Table(
        [[still_cells[0], still_cells[1]], [still_cells[2], still_cells[3]]],
        colWidths=[90 * mm, 90 * mm],
    )
    still_tbl.setStyle(TableStyle([
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("LEFTPADDING", (0, 0), (-1, -1), 2),
        ("RIGHTPADDING", (0, 0), (-1, -1), 2),
        ("TOPPADDING", (0, 0), (-1, -1), 2),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
    ]))
    story.append(still_tbl)
    story.append(Spacer(1, 4))
    story.append(_tile_row(metrics, [
        ("Ball speed", "ball_speed_kmh"),
        ("Arm speed", "arm_speed_kmh"),
        ("Release time", "release_time_ms"),
        ("Release height", "release_height_m"),
    ]))
    story.append(Spacer(1, 6))
    cal = (metrics.get("scale") or {}).get("calibrated")
    view = (metrics.get("quality") or {}).get("camera_view")
    view_bit = f" Camera view: {view}." if view else ""
    story.append(Paragraph(
        "Every number is measured from pose, or shown as — with a reason. "
        "Physical speed and height use the bowler’s stated height"
        + (" (provided)." if cal else " (not provided).")
        + " Angles are image-plane values. Hip/trunk figures are 2D proxies, not 3D lab rotation."
        + view_bit,
        st["small"],
    ))

    # ---------------- Page 2: sequencing + scores + extra tiles ----------------
    story.append(PageBreak())
    story += _section_header(st, "CRICLAB BOWLING REPORT", "SEQUENCING AND KEY METRICS", player_name, date_str)

    events_ms = metrics.get("event_t_ms") or {}
    seq_png = charts.sequencing_chart(chart_dir, metrics.get("rotation_series") or [], events_ms)
    if seq_png and Path(seq_png).exists():
        story.append(Image(str(seq_png), width=178 * mm, height=66 * mm))
        story.append(Spacer(1, 4))

    story.append(Paragraph("KINEMATICS SEQUENCE", st["h2"]))

    seq = metrics.get("kinematic_sequence") or []
    if not seq:
        order = metrics.get("phase_order") or []
        seq = [{"n": i, "label": labels.get(ph, ph), "frame": phases.get(ph), "estimated": ph == "hip_rotation"}
               for i, ph in enumerate(order, 1)]
    seq_lines = []
    for item in seq:
        n = item.get("n") or ""
        lab = item.get("label") or item.get("key") or ""
        seen = item.get("frame") is not None
        mark = "measured" if seen and not item.get("estimated") else ("estimated" if seen else "not seen")
        seq_lines.append(f"<b>{n}. {lab}</b> — {mark}")
    story.append(Paragraph("<br/>".join(seq_lines) or "No sequence events detected.", st["seq"]))
    story.append(Spacer(1, 8))

    story.append(Paragraph("ACTION SCORES", st["h2"]))
    ring_row = [
        _score_ring("Overall", scores.get("overall")),
        _score_ring("Ball speed", scores.get("ball_speed")),
        _score_ring("Sequencing", scores.get("sequencing")),
        _score_ring("Brace", scores.get("front_leg_brace")),
        _score_ring("Separation", scores.get("hip_shoulder_separation")),
    ]
    rt = Table([ring_row], colWidths=[100] * 5, hAlign="LEFT")
    story.append(rt)
    story.append(Paragraph(
        "Heuristic 0–100 indicators from measured angles, speed and timing — for trends, not clinical grading.",
        st["small"],
    ))
    story.append(Spacer(1, 8))
    story.append(_tile_row(metrics, [
        ("Elbow extension", "elbow_extension_deg"),
        ("Stride length", "stride_length_pct_height"),
        ("Front-knee flex", "front_knee_flexion_deg"),
        ("Hip/shoulder sep.", "hip_shoulder_separation_deg"),
    ]))
    story.append(Spacer(1, 4))
    story.append(_tile_row(metrics, [
        ("Arm-swing speed", "arm_swing_speed_deg_s"),
        ("Hip-line proxy", "hip_rotation_speed_deg_s"),
        ("Trunk-line proxy", "trunk_rotation_speed_deg_s"),
        ("Release angle", "release_angle_deg"),
    ]))

    # ---------------- Page 3: joint angles ----------------
    story.append(PageBreak())
    story += _section_header(st, "CRICLAB BOWLING REPORT", "JOINT ANGLES", player_name, date_str)
    angle_series = metrics.get("angle_series") or []
    arm_png = charts.arm_angles_chart(chart_dir, angle_series, events_ms)
    leg_png = charts.leg_trunk_chart(chart_dir, angle_series, events_ms)
    story.append(Paragraph("BOWLING ARM", st["h2"]))
    if arm_png and Path(arm_png).exists():
        story.append(Image(str(arm_png), width=178 * mm, height=74 * mm))
    else:
        story.append(Paragraph("Bowling-arm angle series unavailable (pose too noisy in the delivery window).", st["body"]))
    story.append(Paragraph("TRUNK AND LEGS", st["h2"]))
    if leg_png and Path(leg_png).exists():
        story.append(Image(str(leg_png), width=178 * mm, height=74 * mm))
    else:
        story.append(Paragraph("Trunk/leg angle series unavailable.", st["body"]))
    story.append(Paragraph("Image-plane degrees. 180° elbow or knee = straight.", st["small"]))

    # ---------------- Page 4: joint velocities ----------------
    story.append(PageBreak())
    story += _section_header(st, "CRICLAB BOWLING REPORT", "JOINT VELOCITIES (ANGULAR)", player_name, date_str)
    story.append(Paragraph("BOWLING ARM", st["h2"]))
    base_frame = phases.get("front_foot_contact")
    ws_png = charts.wrist_speed_chart(
        chart_dir, metrics.get("wrist_speed_series") or [], metrics.get("release_frame"),
        base_frame=base_frame, fps=fps, events_ms=events_ms,
    )
    arm_v = charts.arm_velocity_chart(chart_dir, angle_series, events_ms)
    if ws_png and Path(ws_png).exists():
        story.append(Image(str(ws_png), width=178 * mm, height=62 * mm))
        story.append(Spacer(1, 4))
    if arm_v and Path(arm_v).exists():
        story.append(Image(str(arm_v), width=178 * mm, height=62 * mm))
    if not ((ws_png and Path(ws_png).exists()) or (arm_v and Path(arm_v).exists())):
        story.append(Paragraph("Arm velocity series unavailable.", st["body"]))
    story.append(Paragraph("TRUNK AND LEGS", st["h2"]))
    leg_v = charts.leg_trunk_velocity_chart(chart_dir, angle_series, events_ms)
    if leg_v and Path(leg_v).exists():
        story.append(Image(str(leg_v), width=178 * mm, height=62 * mm))
    else:
        story.append(Paragraph("Trunk/leg velocity series unavailable.", st["body"]))
    story.append(Paragraph("Angular velocity is image-plane deg/s — not a 3D motion-lab reading.", st["small"]))

    # ---------------- Page 5: tabular angles + ω ----------------
    story.append(PageBreak())
    story += _section_header(st, "CRICLAB BOWLING REPORT", "TABULAR DATA", player_name, date_str)
    story.append(Paragraph("ANGLES", st["h2"]))
    jt = metrics.get("joint_angle_table") or {}
    phase_cols = [p for p in (metrics.get("phase_order") or []) if p in jt] or list(jt.keys())
    angle_names = [
        ("elbow_extension", "Elbow extension"),
        ("shoulder_abduction", "Shoulder abduction"),
        ("trunk_flexion", "Trunk flexion"),
        ("front_knee_flexion", "Front-knee flexion"),
        ("hip_shoulder_separation", "Hip/shoulder separation"),
        ("bowling_arm_angle", "Bowling-arm angle"),
    ]
    header = ["Angle"] + [labels.get(ph, ph.replace("_", " ").title()) for ph in phase_cols]
    tab_rows = [header]
    for key, name in angle_names:
        r = [name]
        for ph in phase_cols:
            v = (jt.get(ph, {}) or {}).get(key)
            r.append("—" if v is None else f"{v:.0f}°")
        tab_rows.append(r)
    ncols = max(2, len(header))
    colw = [46 * mm] + [(132 * mm) / (ncols - 1)] * (ncols - 1)
    at = Table(tab_rows, colWidths=colw)
    at.setStyle(_table_style(small=True))
    story.append(at)
    story.append(Spacer(1, 10))

    story.append(Paragraph("ANGULAR VELOCITIES", st["h2"]))
    vel_rows = [["Segment (image plane)", "Minimum", "Maximum"]]
    avr = metrics.get("angular_velocity_range") or {}
    named = [
        ("bowling_arm", "Bowling arm"),
        ("hip_line", "Pelvis line (2D proxy)"),
        ("shoulder_line", "Shoulder line (2D proxy)"),
    ]
    for key, name in named:
        r = avr.get(key) or {}
        vel_rows.append([name, _fmt_deg_s(r.get("min")), _fmt_deg_s(r.get("max"))])
    for key, name in [
        ("elbow_extension", "Elbow extension"),
        ("shoulder_abduction", "Shoulder abduction"),
        ("trunk_flexion", "Trunk flexion"),
        ("front_knee_flexion", "Front-knee flexion"),
    ]:
        r = charts.angle_velocity_range(angle_series, key)
        vel_rows.append([name, _fmt_deg_s(r.get("min")), _fmt_deg_s(r.get("max"))])
    vt = Table(vel_rows, colWidths=[78 * mm, 50 * mm, 50 * mm])
    vt.setStyle(_table_style())
    story.append(vt)
    story.append(Paragraph("Min/max are 5th–95th percentile. Values outside a realistic band are omitted, not clamped.", st["small"]))

    # ---------------- Page 6: stride / time / speed / scores ----------------
    story.append(PageBreak())
    story += _section_header(st, "CRICLAB BOWLING REPORT", "DELIVERY SUMMARY", player_name, date_str)

    height_m = profile.get("height_m")
    stride_pct = _ok_val(metrics, "stride_length_pct_height")
    stride_in = None
    if stride_pct is not None and height_m:
        stride_in = (float(height_m) * float(stride_pct) / 100.0) / 0.0254
    stride_ms = _dt_ms(phases, "back_foot_contact", "front_foot_contact", fps)
    mer_to_rel = _dt_ms(phases, "max_external_rotation", "release", fps)
    if mer_to_rel is None:
        mer_to_rel = _dt_ms(phases, "arm_horizontal", "release", fps)
    hip_to_rel = _dt_ms(phases, "hip_rotation", "release", fps)
    ffc_to_mer = _dt_ms(phases, "front_foot_contact", "max_external_rotation", fps)
    if ffc_to_mer is None:
        # Same arm_horizontal fallback mer_to_rel uses — the two rows must agree
        # on whether MER was seen.
        ffc_to_mer = _dt_ms(phases, "front_foot_contact", "arm_horizontal", fps)

    # Delivery classification + how the clip was timed. The capture rate belongs
    # on the page because every speed and duration above it is divided by that
    # number — if it was recovered from the ball's fall rather than read from the
    # file, the reader should be able to see that.
    legality = metrics.get("action_legality") or {}
    pace = metrics.get("delivery_type") or {}
    tbase = metrics.get("timebase") or {}
    ext = legality.get("extension_deg")
    verdict_label = {
        "within_limit": "Within the 15° limit",
        "borderline": "Borderline — re-film square-on",
        "above_limit_screening": "Above 15° on this view (screening only)",
        "flexing": "Elbow flexes into release — no extension",
    }.get(legality.get("verdict"), "Not assessable from this camera angle")
    story.append(Paragraph("DELIVERY & ARM ACTION", st["h2"]))
    action_tbl = Table([
        ["Metric", "Value"],
        ["Pace band", pace.get("value") or "—"],
        ["Pace band measured from", (pace.get("basis") or "—").replace("_", " ")],
        ["Elbow angle at release (deg)", _fmt(_ok_val(metrics, "elbow_extension_deg"), 1)],
        ["Elbow extension, arm-horizontal → release (deg)", _fmt(ext, 1)],
        ["Throwing-action screening (ICC 15°)", verdict_label],
        ["Capture frame rate (fps)", _fmt(tbase.get("fps"), 0)],
        ["Frame rate source", "measured from ball's fall" if tbase.get("slow_motion") else "video file"],
    ], colWidths=[110 * mm, 68 * mm])
    action_tbl.setStyle(_table_style())
    story.append(action_tbl)
    if tbase.get("slow_motion"):
        story.append(Spacer(1, 3))
        story.append(Paragraph(str(tbase.get("note") or ""), st["small"]))
    if legality.get("note"):
        story.append(Spacer(1, 3))
        story.append(Paragraph(str(legality["note"]), st["small"]))
    story.append(Spacer(1, 8))

    story.append(Paragraph("STRIDE STEP", st["h2"]))
    stride_tbl = Table([
        ["Metric", "Value"],
        ["Length of stride (% height)", _fmt(stride_pct, 1)],
        ["Length of stride (inches)", _fmt(stride_in, 1)],
        ["Length of stride (m)", _fmt((float(height_m) * float(stride_pct) / 100.0) if stride_pct is not None and height_m else None, 2)],
        ["Time of stride step, BFC → FFC (ms)", _fmt(stride_ms, 0)],
        ["Stated height (m)", _fmt(height_m, 2)],
        ["Bowling arm", (profile.get("bowling_arm") or metrics.get("throwing_side") or "—")],
        ["Bowling style", (profile.get("bowling_style") or "—")],
    ], colWidths=[110 * mm, 68 * mm])
    stride_tbl.setStyle(_table_style())
    story.append(stride_tbl)
    story.append(Spacer(1, 8))

    story.append(Paragraph("TIME / SPEED", st["h2"]))
    time_tbl = Table([
        ["Metric", "Value"],
        ["Ball speed (km/h)", _fmt(_ok_val(metrics, "ball_speed_kmh"), 1)],
        ["Arm speed at leave-hand (km/h)", _fmt(_ok_val(metrics, "arm_speed_kmh"), 1)],
        ["Release time, FFC → REL (ms)", _fmt(_ok_val(metrics, "release_time_ms"), 0)],
        ["Release height (m)", _fmt(_ok_val(metrics, "release_height_m"), 2)],
        ["Release angle (deg, image plane)", _fmt(_ok_val(metrics, "release_angle_deg"), 1)],
        ["Arm-swing speed (deg/s)", _fmt(_ok_val(metrics, "arm_swing_speed_deg_s"), 0)],
        ["Peak hip-line proxy (deg/s)", _fmt(_ok_val(metrics, "hip_rotation_speed_deg_s"), 0)],
        ["Peak trunk-line proxy (deg/s)", _fmt(_ok_val(metrics, "trunk_rotation_speed_deg_s"), 0)],
        ["Time between peak hip / trunk (ms)", _fmt(_ok_val(metrics, "hip_to_trunk_peak_gap_ms"), 0)],
        ["Time FFC → MER (ms)", _fmt(ffc_to_mer, 0)],
        ["Time MER → REL (ms)", _fmt(mer_to_rel, 0)],
        ["Time hip rotation → REL (ms)", _fmt(hip_to_rel, 0)],
    ], colWidths=[110 * mm, 68 * mm])
    time_tbl.setStyle(_table_style())
    story.append(time_tbl)
    cons = metrics.get("speed_consistency") or {}
    if cons.get("note"):
        story.append(Spacer(1, 3))
        story.append(Paragraph(str(cons["note"]), st["small"]))
    story.append(Spacer(1, 8))

    story.append(Paragraph("SCORES", st["h2"]))
    score_tbl = Table([
        ["Score", "Value"],
        ["Sequencing", _fmt(scores.get("sequencing"), 0)],
        ["Ball speed", _fmt(scores.get("ball_speed"), 0)],
        ["Arm speed", _fmt(scores.get("arm_speed"), 0)],
        ["Front-leg brace", _fmt(scores.get("front_leg_brace"), 0)],
        ["Hip/shoulder separation", _fmt(scores.get("hip_shoulder_separation"), 0)],
        ["Overall", _fmt(scores.get("overall"), 0)],
    ], colWidths=[110 * mm, 68 * mm])
    score_tbl.setStyle(_table_style())
    story.append(score_tbl)

    # ---------------- Page 7: AI coach ----------------
    story.append(PageBreak())
    story += _section_header(st, "CRICLAB BOWLING REPORT", "AI COACH NOTES", player_name, date_str)
    for label, key in [
        ("Summary", "summary"), ("Observations", "observations"),
        ("Strengths", "strengths"), ("Areas to improve", "improvements"),
        ("Confidence note", "confidence_note"),
    ]:
        story.append(Paragraph(label, st["h2"]))
        story.append(Paragraph((analysis.get(key) or "—").replace("\n", "<br/>"), st["body"]))
    evidence = None
    if stills_dir:
        evidence = _resolve_still(stills_dir, ("release",))
    if evidence is None:
        evidence = release_still
    recs = analysis.get("recommendations") or []
    if recs:
        story.append(Paragraph("Recommended drills", st["h2"]))
        story.append(Paragraph(
            "Catalog videos only — the model cannot invent YouTube links.",
            st["sub"],
        ))
        for rec in recs:
            title = rec.get("title") or rec.get("drill_id") or "Drill"
            yt = rec.get("youtube_id")
            url = f"https://www.youtube.com/watch?v={yt}" if yt else ""
            reason = rec.get("reason") or ""
            if url:
                story.append(Paragraph(
                    f'<b>{title}</b> — <link href="{url}" color="blue">{url}</link>',
                    st["body"],
                ))
            else:
                story.append(Paragraph(f"<b>{title}</b>", st["body"]))
            if reason:
                story.append(Paragraph(reason.replace("\n", "<br/>"), st["body"]))
            story.append(Spacer(1, 4))

    if evidence and Path(evidence).exists():
        story.append(Spacer(1, 6))
        story.append(Paragraph("RELEASE-FRAME EVIDENCE", st["h2"]))
        story.append(Image(str(evidence), width=150 * mm, height=84 * mm))

    def _on_page(canvas, doc_):
        canvas.saveState()
        w, h = A4
        canvas.setFillColor(GREY)
        canvas.setFont("Helvetica", 7)
        canvas.drawString(16 * mm, h - 10 * mm, "CricLab  ·  Private bowling analysis")
        canvas.drawRightString(w - 16 * mm, h - 10 * mm, date_str)
        canvas.setStrokeColor(LINE)
        canvas.setLineWidth(0.4)
        canvas.line(16 * mm, h - 11.5 * mm, w - 16 * mm, h - 11.5 * mm)
        canvas.drawString(16 * mm, 8 * mm, "CricLab bowling action report  ·  cricket metrics, not a radar gun")
        canvas.drawRightString(w - 16 * mm, 8 * mm, f"{doc_.page}")
        canvas.restoreState()

    doc.build(story, onFirstPage=_on_page, onLaterPages=_on_page)
    return out_path


def _fmt_deg_s(v) -> str:
    if v is None:
        return "—"
    return f"{v:.0f}°/s"


def _table_style(small: bool = False) -> TableStyle:
    fs = 7 if small else 9
    return TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), PITCH),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTSIZE", (0, 0), (-1, -1), fs),
        ("GRID", (0, 0), (-1, -1), 0.4, LINE),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, MIST]),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("LEFTPADDING", (0, 0), (-1, -1), 5),
        ("RIGHTPADDING", (0, 0), (-1, -1), 5),
        ("TOPPADDING", (0, 0), (-1, -1), 4),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
    ])
