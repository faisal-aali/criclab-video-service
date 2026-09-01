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
| Storage | Cloudinary + local `STORAGE_DIR` | Source download, overlay + PDF upload |
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
      → pose / ball-flight → metrics → overlay → Cloudinary
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
│   ├── services/         # Cloudinary download / overlay+PDF upload
│   ├── db/               # Mongo (shared)
│   └── main.py           # health API only (:8001)
├── memory-bank/
├── requirements.txt
└── .env.example
```

## Production

Same machine as the website API (EC2 / Lightsail). Clone to `/var/www/criclab-video-service`, matching Mongo / Cloudinary / `STORAGE_DIR`, then:

```bash
pm2 start deploy/ecosystem.config.cjs
```

That starts health on `127.0.0.1:8001` and **two** worker processes. Do not expose `:8001` publicly. GitHub Actions deploy for this repo is not wired yet (FEAT-027).

## MongoDB (this process writes)

| Collection | Role |
|------------|------|
| `jobs` / `balltrack_jobs` | Claim `queued` → `claimed` / progress / complete / fail |
| `deliveries` | Action analysis documents |
| `balltrack_sessions` / `balltrack_deliveries` | Ball-flight session + per-ball docs |
| `videos` | Read path / `source_url` (created by the website API) |

## Constraints

- Do not put frame measurement in Gemma
- Do not add JWT, CORS, SMTP, or Train/admin drill CRUD here
- Catalog HTTP stays on the website API; this repo keeps a **copy** of `drills.json` for matching (admin edits do not auto-sync)
- Mongo, Cloudinary, and `STORAGE_DIR` must match the website API
- Never merge Ball-flight stump speed into an Action pose job
