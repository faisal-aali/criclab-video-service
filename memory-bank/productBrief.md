# Product Brief — Cric-Lab (Video service)

## Purpose

Cric-Lab is an **AI Cricket Bowling Laboratory**: upload a bowling video, get measurable bowling insights on the page, and download a professional PDF analysis report. Inspired by SpinLab AI’s video-analysis experience, focused on cricket bowling for v1.

**This repo** is the worker that *measures* the clip and writes the delivery. The website API queues the job; React only displays what Mongo already holds.

**Core promise:** Two film modes. **Action** (side-on) → pose mechanics + honest 2D estimates. **Ball flight** (behind-bowler + both stumps) → pitch-plane speed, line, length. Gemma coaches and picks catalog drills — it never invents km/h.

Like **SpinLab AI** for quarterbacks (slowed video with overlaid analysis + a
biomechanics PDF), but for **cricket bowling**.

## Users

- Individual bowlers / athletes analyzing their own action
- Coaches reviewing bowling sessions
- Single-user or small coaching workflows first

## What this repo owns (v1 — bowling only)

1. **Pose pipeline** — MediaPipe BlazePose landmarks per frame (measurement engine)
2. **Release & phases** — leave-hand release; back-foot → front-foot →
   arm-horizontal → release → follow-through (omit if not seen)
3. **Metrics JSON** — ball speed only with an in-air lock; leave-hand arm/hand
   speed; release height/angle/time; joint angles; stride; rotation proxies;
   action scores from `status === ok` metrics only
4. **Slow-motion overlay clip** — original colour footage with SpinLab-style HUD
5. **AI analysis** — Gemma coaches from structured metrics only; picks YouTube drills from this repo's **closed catalog snapshot** (`app/coaching/drills.json`)
6. **PDF report** — SpinLab-style cricket report + drill URLs as text
7. **Ball flight** — stump homography → pitch-plane speed, line, length
8. **Delivery writes** — persist analysis in the shared MongoDB

Upload, auth, Train catalog **HTTP**, and job **reads** live in `criclab-web-backend`.

## Success Metrics

| Metric | Target |
|--------|--------|
| End-to-end: queued job → overlay + PDF + delivery | Works on controlled bowling videos |
| Release frame identified | Detectable on side-on / clear deliveries |
| Metrics shown with confidence | Always (estimates labeled as estimates) |
| Analysis persisted | MongoDB document per session/delivery |
| PDF written | After analysis completes |
| AI summary grounded in metrics | No hallucinated radar-grade claims |

## Out of Scope (v1)

- Batting, fielding, wicket-keeping analysis
- Multi-camera synchronized capture
- Claiming radar-gun accuracy without calibration + ground-truth validation
- JWT, CORS, SMTP, or browser-facing drill CRUD in this process
- Frame-by-frame measurement by the LLM

## Product Principles

- **CV + physics first, LLM second** — Gemma coaches and selects catalog drills; it does not measure
- **Estimates until calibrated** — Action km/h is 2D + height; Ball flight is stump pitch-plane, still not a radar gun
- **Two modes, two truths** — never merge stump speed into a pose job
- **Bowling-only MVP** — reliable workflow over every cricket scenario
- **Modular pipeline** — new metrics later without rewriting the worker loop
- **One job per worker process** — run more PM2 instances for parallel clips
