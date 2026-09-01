# System Patterns — Cric-Lab (Video service)

> **Most important Memory Bank file for this repo.** Read before writing worker / CV code.
> Website HTTP, auth, and Train catalog CRUD: `../criclab-web-backend/memory-bank/systemPatterns.md`.
> UI truth contract: `../criclab-web-frontend/memory-bank/systemPatterns.md`.

## Architecture rule #1

**Computer vision + physics produce measurements. The LLM coaches; it never measures.**

Never ask Gemma to estimate speed/angles from raw video frames (that *increases* false km/h). There is no custom cricket LLM. Gemma runs *after* CV and only narrates `status === ok` JSON, then picks **catalog drill IDs** from this repo's `app/coaching/drills.json`. Never invent YouTube URLs.

Action pipeline order:

```text
Claim queued job → Extract meta → POSE → Action/release → Calibrate
  → Best-effort ball track → Metrics JSON → Slow-mo overlay
  → Cloudinary overlay → Gemma narrative + drill matching
  → PDF → Cloudinary PDF → Persist delivery → job completed
```

## Three processes (do not treat `app` as one package)

| Process | Repo | Owns |
|---------|------|------|
| Website API | `criclab-web-backend` | Auth, upload, insert `queued` jobs, job/delivery **reads**, stump *still* calibration, signed Cloudinary upload, chat assistant, Train catalog HTTP, bookings |
| Video worker | **this repo** | Everything after claim: MediaPipe/OpenCV, metrics, overlay, PDF, Gemma *video* notes, drill matching, overlay/PDF upload, delivery **writes** |
| UI | `criclab-web-frontend` | Display API JSON only |

`app.coaching` **here** is matching (`weakness_tags`, `balltrack_tags`, hydrate) plus a read-only `drills.json` snapshot. `app.coaching` on the website API is catalog I/O (`load_catalog` / `save_catalog`). They are not the same package. Admin edits on the website do not auto-sync here.

Same relative filenames (`agent/ollama_agent.py`, `balltrack/stumps.py`) are allowed when the role differs. Do not import website-backend modules from this worker, or vice versa.

## Worker loop

- One process = one clip at a time (`python -m app.worker`).
- Claims Action jobs first, then Ball-flight if the Action queue is empty.
- Parallel clips = more processes (production PM2 `criclab-video-worker` `instances: 2`).
- Stale `claimed` jobs older than 45 minutes are re-queued.
- `pipeline/runner.py` re-raises after writing `status=failed` so the worker logs `job failed`, not `finished`.

## Two film modes (do not merge their numbers)

| Mode | Camera | Truth |
|------|--------|--------|
| **Action** | Side-on full-body | Mechanics from pose. Ball km/h is image-plane + height, or `—`. |
| **Ball flight** | Behind non-striker + both stumps | Pitch-plane speed / line / length. Separate job. |

Never paste stump speed onto a pose job.

## Architecture rule #2 — pose is the measurement engine; ball speed needs a real lock

Pose (MediaPipe) drives release, FFC, joint angles, stride, arm-swing. **Ball
speed is the headline metric only when the ball is tracked in flight** — a path
that leaves the bowling hand and keeps moving downrange. A lock on the wrist,
torso, fence, or a stationary tree is not a ball track: return null + reason.

- **Release = leave-hand**, not the highest wrist (cocking / MER).
- **Front-foot contact** = lead-ankle plant **60–600 ms** before release (omitted if not found).
- **Arm speed** = bowling-wrist px/frame **at leave-hand** × scale × fps.
- **Scale** from upright (90th-pct) head→ankle × **user-provided** height or mpp. Never invent 1.7 m.
- **Ball speed (Action)** = one physics-constrained robust fit on the in-air path (`track.robust_release_velocity_px_per_frame`): gravity pinned to `g/(mpp·fps²)`, Theil–Sen slopes, interpolated points excluded. Refuse below `MIN_BALL_FIT_QUALITY`. Sanity band ~25–160 km/h. Reject — never clamp.
- **Ball speed (Ball flight)** = stump-homography pitch-plane distance/time. Separate job.
- **Truth contract**: measure or null+reason. UI/PDF/overlay share the same metrics JSON (`status === ok`).
- **The timebase is a measurement.** Slow-mo containers lie about fps. `pipeline/timebase.py` recovers capture rate from the ball's fall.
- **Finding the ball and quoting its speed are separate.** Keep a real in-air path for overlay/release/timebase even when geometry cannot support km/h.
- **No threshold in absolute pixels or px/frame.** Scale by body height, ball radius, or real m/s through `mpp × fps`.
- **Throwing (ICC 15°) screening is refused unless the view earns it.** A false "illegal action" is worse than an honest `—`.
- **Cross-validate independently measured quantities** (`metrics._cross_validate`). Do not silently "correct" contradictions.

## Pipeline modules (this repo)

| Stage | Module |
|-------|--------|
| Extract | `pipeline/extract.py` |
| Pose | `pipeline/pose.py` |
| Action | `pipeline/action.py` |
| Timebase | `pipeline/timebase.py` |
| Calibrate | `pipeline/calibrate.py` |
| Ball (Action, opt.) | `pipeline/detect.py` + `track.py` |
| Metrics | `pipeline/metrics.py` |
| Render | `pipeline/render.py` |
| Upload | `services/cloudinary_service.py` |
| Agent | `agent/ollama_agent.py` |
| Matching | `coaching/recommend.py` + `drills.json` |
| PDF | `pdf/report.py` + `pdf/charts.py` |
| Ball flight | `balltrack/` |

Add a metric by extending the metrics stage + JSON. UI cards live in `criclab-web-frontend`. Do not rewrite the worker loop.

## Overlay + PDF

- Overlay burns onto **original colour** frames. Tiles match metrics JSON (`—` if status ≠ ok). One ball speed per delivery — never a per-frame label that contradicts the headline.
- PDF is SpinLab-style cricket pages + catalog drill URLs as **text** (no iframes).
- Cloudinary overlay/PDF upload failures must not fail the job; fall back to local `/artifacts/...`.

## Coaching (matching, not catalog HTTP)

- Tags are deterministic (`weakness_tags` for Action, `balltrack_tags` for Ball flight). Gemma does not choose tags.
- `hydrate_recommendations` keeps only IDs that exist in this repo's `drills.json`.
- Do not add `GET /coaching/drills` or admin CRUD here.

## Agentic layer

Python helpers the runner calls (not Ollama tool-calling):

- `getDeliveryMetrics` — compact `status === ok` view
- `compareDeliveries` — ball-speed delta vs this bowler's prior deliveries
- `generateReport` — narrative + catalog-only drill IDs

`APP_ENV=local` → Ollama. `APP_ENV=production` → Bedrock.

## Anti-patterns (do not introduce)

- Using the LLM as the motion engine or to invent km/h / YouTube IDs
- Reporting ball speed from a hand/body/tree lock
- Merging stump (Ball flight) speed into an Action job
- Adding JWT, CORS, SMTP, or Train admin routes to this process
- Importing `criclab-web-backend` packages as if they were this `app`
- Running MediaPipe on Python 3.13/3.14
- Processing two clips in one worker process (run a second process instead)

## MVP workflow checklist

1. Website API inserts `queued` job
2. This worker claims it
3. Pose → action → scale → track → metrics
4. Overlay + Cloudinary
5. Gemma notes + catalog drills
6. PDF + Cloudinary
7. Persist delivery; website API serves results
