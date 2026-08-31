# CricLab Video Service

Dedicated workers for Action analysis and Ball flight. The website API only
creates `queued` jobs in Mongo. These processes claim them and run OpenCV /
MediaPipe so the web process stays free — closing a tab does not stop a job.

## Run locally

Same Mongo and `STORAGE_DIR` as `criclab-web-backend`.

```bash
python3.11 -m venv .venv312
source .venv312/bin/activate
pip install -r requirements.txt
cp .env.example .env   # point MONGODB_URI at the same database

# terminal 1 — optional health API on :8001
uvicorn app.main:app --port 8001

# terminal 2+ — one worker per process (run two for parallel clips)
python -m app.worker
```

Website backend no longer starts analysis in-process. If no worker is running,
jobs stay at `queued` until one is.

## Production (Lightsail)

Clone to `/var/www/criclab-video-service`, same `.env` Mongo/Cloudinary as the
API, same `STORAGE_DIR`, then:

```bash
pm2 start deploy/ecosystem.config.cjs
```

Two worker processes claim jobs independently.
