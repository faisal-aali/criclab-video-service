# Roadmap — Cric-Lab (Video service)

High-level features **this worker** runs. Website auth / Train HTTP / UI live in the sibling repos. Detail lives in `tasks/`.

| ID | Feature | Status | Summary |
|----|---------|--------|---------|
| FEAT-003 | Video pipeline | Done | OpenCV extract + metadata |
| FEAT-004 | **Pose estimation** | Done | MediaPipe BlazePose 33-landmark track |
| FEAT-005 | **Action/release detection** | Done | Throwing side + leave-hand release + phases |
| FEAT-006 | Calibration | Done (basic) | Scale from pose body height or provided reference |
| FEAT-007 | **Biomechanics metrics** | Done | Leave-hand arm speed, joints, timing, scores from ok metrics only |
| FEAT-008 | **Slow-mo overlay video** | Done | SpinLab HUD on original colour frames |
| FEAT-009 | **Cloudinary hosting** | Done | Overlay + PDF upload from this process |
| FEAT-011 | AI agent (Gemma) | Done | Coaching from metrics JSON; catalog-only drill IDs |
| FEAT-012 | **SpinLab-style PDF** | Done | Event stills, charts, AI notes, drill URLs as text |
| FEAT-014 | Ball tracking (Action) | Done | In-air flight lock; headline km/h when the path leaves the hand |
| FEAT-014b | **Ball flight (stumps)** | Done | Pitch-plane speed / line / length |
| FEAT-018 | **Capture-rate recovery** | Done | Slow-mo timed from the ball's fall (`pipeline/timebase.py`) |
| FEAT-019 | **Delivery type & throwing screen** | Done | Pace band; ICC 15° only where the view supports it |
| FEAT-026 | **Dedicated video workers** | Done | Claim Mongo jobs; matching in `app/coaching/`; catalog HTTP stays on the website API |
| FEAT-015 | Coaching memory | Planned | Embeddings over historical notes — not wired |
| FEAT-016 | Validation | Planned | Radar / ground-truth; multi-view rotation + depth speed |
| FEAT-027 | **CI/deploy for this repo** | Planned | GitHub Actions + PM2 on the instance (API/frontend already have this) |

## Change log

- Pipeline originally lived in `criclab-web-backend`. It now runs here so the website process stays free and closing a tab cannot stop a job.
- **1 Sep 2026 (FEAT-026):** Drill matching (`weakness_tags`, `balltrack_tags`, hydrate) plus a read-only `drills.json` snapshot live in `app/coaching/`. Train / admin catalog HTTP stays on the website API. Failed Action jobs re-raise so the worker logs `job failed`, not `finished`.
