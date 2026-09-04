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
| FEAT-009 | **S3 hosting** | Done | GetObject original; overlay/PDF/compressed PutObject (no CloudFront) |
| FEAT-011 | AI agent (Gemma) | Done | Coaching from metrics JSON; catalog-only drill IDs |
| FEAT-012 | **SpinLab-style PDF** | Done | Event stills, charts, AI notes, drill URLs as text |
| FEAT-014 | Ball tracking (Action) | Done | In-air flight lock; headline km/h when the path leaves the hand |
| FEAT-014b | **Ball flight (stumps)** | Done | Pitch-plane speed / line / length |
| FEAT-018 | **Capture-rate recovery** | Done | Slow-mo timed from the ball's fall (`pipeline/timebase.py`) |
| FEAT-019 | **Delivery type & throwing screen** | Done | Pace band; ICC 15° only where the view supports it |
| FEAT-026 | **Dedicated video workers** | Done | Claim Mongo jobs; matching in `app/coaching/`; catalog HTTP stays on the website API |
| FEAT-015 | Coaching memory | Planned | Embeddings over historical notes — not wired |
| FEAT-016 | Validation | Planned | Radar / ground-truth; multi-view rotation + depth speed |
| FEAT-027 | **CI/deploy for this repo** | Done | Push to `main` → GitHub Actions CI + self-hosted Deploy (`pull.sh` + `restart.sh` + PM2) |
| FEAT-031 | **Daily video quota** | Done | Atomic `quota_days` lease; global FIFO claim; slot release on fail/stale |
| FEAT-032 | **Glacier originals** | Done | Archive `original/` on completed/failed/honored-cancel; skip while a live job still needs GetObject |
| FEAT-033 | **Honor in-flight cancel** | Done | Next-stage abort; do not overwrite `cancelled`; no delivery; quota slot stays used |
| FEAT-034 | **Action clip gates** | Done | 100 MiB Action download; tagged 120/240, landscape 1080p, ≤10 s before pose |

## Change log

- **4 Sep 2026 (FEAT-034):** Action download cap 100 MiB; `clip_probe` + `clip_spec` before pose (tagged 120/240, landscape 1080p, ≤10 s). Ball flight stays 180 MB.
- Pipeline originally lived in `criclab-web-backend`. It now runs here so the website process stays free and closing a tab cannot stop a job.
- **1 Sep 2026 (FEAT-026):** Drill matching (`weakness_tags`, `balltrack_tags`, hydrate) plus a read-only `drills.json` snapshot live in `app/coaching/`. Train / admin catalog HTTP stays on the website API. Failed Action jobs re-raise so the worker logs `job failed`, not `finished`.
- **1 Sep 2026 (FEAT-027):** Same CI/deploy shape as the website API. `compileall` on GitHub; Deploy on the instance runner pulls `/var/www/criclab-video-service` and restarts one `criclab-video-worker`. No PM2 health API.
- **2 Sep 2026 (TASK-002 / FEAT-031):** Daily start cap. Claim the oldest eligible job across both collections after winning a `quota_days` lease. Fail and stale re-queue release today's slot. Idle-stop when nothing is claimable now; the website API starts this instance at 00:00 UTC for leftover FIFO jobs.
- **3 Sep 2026 (TASK-003 / FEAT-032):** After `completed`/`failed`/honored-`cancelled`, CopyObject the original to Glacier Flexible Retrieval. Stale mid-run fails archive; stale `claimed` re-queue does not. No RestoreObject.
- **3 Sep 2026 (TASK-004 / FEAT-033):** Cancel button is honored at the next stage. `update_job` cannot flip `cancelled` back to `processing`. `JobCancelled` does not become `failed` and does not persist a delivery. In-flight cancel keeps the daily slot.
