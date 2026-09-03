# Tech Context — Cric-Lab (Video service)

Sibling website API: `../criclab-web-backend`. Sibling UI: `../criclab-web-frontend`.
This repo claims Mongo jobs and runs computer vision, overlay, PDF, and Gemma video notes.

## Stack

| Layer | Technology | Purpose |
|-------|------------|---------|
| Worker | `python -m app.worker` | Claim `queued` Action / Ball-flight jobs |
| Pose | MediaPipe BlazePose | 33 landmarks/frame — Action measurement engine |
| Video / CV | OpenCV | Extract, track, overlay MP4 (avc1/H.264) |
| Ball flight | `app/balltrack/` | Stump homography speed / line / length |
| Motion | `app/pipeline/metrics.py` | Arm speed, joints, timing, scores |
| Charts | matplotlib (Agg) | PDF charts |
| PDF | ReportLab | SpinLab-style bowling report |
| Video LLM | Ollama `gemma3:4b` (local) or Bedrock (production) | Coaching narrative + catalog drill IDs |
| Storage | S3 + local `STORAGE_DIR` | Source download, overlay + PDF upload. CloudFront signing is the website API |
| DB | MongoDB (shared with website API) | Job progress, delivery writes |

## Python environment

MediaPipe requires **Python 3.10–3.12** and pins `numpy<2`. Use `.venv312`.

```bash
cd criclab-video-service
source .venv312/bin/activate      # Windows: .\.venv312\Scripts\Activate.ps1
pip install -r requirements.txt
python -m app.worker
```

## High-level data flow

```text
Vite React (upload)
  → website FastAPI inserts queued job + video/session row
    → this worker claims the job
      → pose / ball-flight → metrics → overlay → S3
        → Gemma notes + drill matching (local drills.json)
          → PDF → Mongo delivery
            → React polls website API for results
```

## Repo layout

```text
criclab-video-service/
├── app/
│   ├── worker.py         # claim loop
│   ├── pipeline/         # Action: extract · pose · action · track · metrics · render
│   ├── balltrack/        # Ball flight: stumps · homography · overlay
│   ├── coaching/         # matching + read-only drills.json snapshot
│   ├── agent/            # Gemma video notes (not the website chat assistant)
│   ├── pdf/              # report + charts
│   ├── services/         # S3 download / overlay+PDF upload / Glacier originals
│   ├── db/               # Mongo (shared)
│   └── main.py           # optional local health only — not run in production PM2
├── memory-bank/
├── requirements.txt
└── .env.example
```

## Production

Dedicated worker EC2 (not the website API/frontend box). Clone to `/var/www/criclab-video-service`, matching Mongo / S3 / `STORAGE_DIR`, then:

```bash
pm2 start deploy/ecosystem.config.cjs
```

That starts **one** `criclab-video-worker` process. Push to `main` deploys via the self-hosted runner (FEAT-027): `deploy/pull.sh` then `deploy/restart.sh`. The website API stays on the always-on box (`:8000`); this worker has no public HTTP. After `EC2_IDLE_STOP_SECONDS` with nothing claimable, this instance stops itself; the API starts it again when a clip can run (including 00:00 UTC).

## MongoDB (this process writes)

| Collection | Role |
|------------|------|
| `jobs` / `balltrack_jobs` | Claim `queued` → `claimed` / progress / complete / fail |
| `quota_days` | UTC-day start counter (`started`); this process leases/releases |
| `quota_state` | `dirty_at` so the website API recomputes expected start times |
| `deliveries` | Action analysis documents |
| `balltrack_sessions` / `balltrack_deliveries` | Ball-flight session + per-ball docs |
| `videos` | `source_key` (`original/…`); worker writes `compressed_key` |

## Constraints

- Do not put frame measurement in Gemma
- Do not add JWT, CORS, SMTP, or Train/admin drill CRUD here
- Catalog HTTP stays on the website API; this repo keeps a **copy** of `drills.json` for matching (admin edits do not auto-sync)
- Mongo, S3, and `STORAGE_DIR` must match the website API
- Never merge Ball-flight stump speed into an Action pose job
- Do not add CloudFront signing or `CLOUDFRONT_*` env here — persist object keys only
- Playback encode is ffmpeg 1280×720 / 30 fps / 1.5 Mbps (`imageio-ffmpeg`); CV always uses the original file
- After a job is `completed` or `failed`, this process CopyObjects `original/` to Glacier Flexible Retrieval (`GLACIER`). IAM needs GetObject + PutObject + HeadObject on `original/*`. 90-day minimum storage charge. Do not RestoreObject.
