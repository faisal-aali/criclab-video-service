# CricLab Video Service

Dedicated workers for Action analysis and Ball flight. The website API only
creates `queued` jobs in Mongo. These processes claim them and run OpenCV /
MediaPipe, overlay, PDF, and drill matching so the web process stays free —
closing a tab does not stop a job.

Train / admin catalog HTTP stays on the website API. This repo keeps a
read-only copy of `app/coaching/drills.json` for matching after CV.

## Run locally

Same Mongo, Cloudinary, and `STORAGE_DIR` as `criclab-web-backend`.
Auth, SMTP, and CORS stay on the website API — they are not used here.

```bash
python3.11 -m venv .venv312
source .venv312/bin/activate
pip install -r requirements.txt
cp .env.example .env   # already filled locally; keep it gitignored

python -m app.worker
```

Website backend no longer starts analysis in-process. If no worker is running,
jobs stay at `queued` until one is.

## Production (Lightsail)

Clone to `/var/www/criclab-video-service`, copy `.env` (Mongo / Cloudinary /
`STORAGE_DIR` matching the API), then:

```bash
pm2 start deploy/ecosystem.config.cjs
```

One worker process claims jobs (one clip at a time). Push to `main` deploys via the self-hosted runner.
