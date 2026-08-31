#!/usr/bin/env bash
set -euo pipefail
cd /var/www/criclab-video-service
exec .venv312/bin/python -m uvicorn app.main:app --host 127.0.0.1 --port 8001 --workers 1
