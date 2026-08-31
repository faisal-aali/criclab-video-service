#!/usr/bin/env bash
set -euo pipefail
cd /var/www/criclab-video-service
exec .venv312/bin/python -m app.worker
