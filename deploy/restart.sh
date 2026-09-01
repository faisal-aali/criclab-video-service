#!/usr/bin/env bash
# Install, compile, restart PM2. Does not delete the venv or .env.
set -euo pipefail

ROOT="/var/www/criclab-video-service"
cd "$ROOT"

pick_python() {
  local cmd ver
  if [[ -n "${PYTHON:-}" ]] && command -v "$PYTHON" >/dev/null 2>&1; then
    cmd="$PYTHON"
  else
    cmd=""
    for cand in python3.12 python3.11 python3.10 python3; do
      if command -v "$cand" >/dev/null 2>&1; then
        cmd="$cand"
        break
      fi
    done
  fi
  if [[ -z "$cmd" ]]; then
    echo "No Python 3 interpreter on PATH." >&2
    exit 1
  fi
  ver="$("$cmd" -c 'import sys; print("%d.%d" % sys.version_info[:2])')"
  case "$ver" in
    3.10|3.11|3.12) echo "$cmd" ;;
    *)
      echo "Need Python 3.10–3.12 (found $cmd = $ver)." >&2
      exit 1
      ;;
  esac
}

PY="$(pick_python)"
echo "Using $($PY --version 2>&1)"

if [[ ! -f .env ]]; then
  echo "Missing ${ROOT}/.env — create it on the instance." >&2
  exit 1
fi

if [[ ! -x .venv312/bin/python ]]; then
  "$PY" -m venv .venv312
fi
# shellcheck source=/dev/null
source .venv312/bin/activate
python -m pip install -q --upgrade pip
pip install -q -r requirements.txt

echo "Lint / compile"
python -c "import uvicorn, fastapi"
python -m compileall -q app

if ! command -v pm2 >/dev/null 2>&1; then
  echo "pm2 is not on PATH for $(whoami)." >&2
  exit 1
fi

echo "Deploy"
if pm2 describe criclab-video-api >/dev/null 2>&1 \
  && pm2 describe criclab-video-worker >/dev/null 2>&1; then
  pm2 restart criclab-video-api criclab-video-worker --update-env
else
  pm2 delete criclab-video-api >/dev/null 2>&1 || true
  pm2 delete criclab-video-worker >/dev/null 2>&1 || true
  # leftover from the current single criclab-video* process (id 2)
  while read -r name; do
    [[ -z "$name" ]] && continue
    pm2 delete "$name" >/dev/null 2>&1 || true
  done < <(pm2 jlist | python -c '
import json,sys
try:
    apps=json.load(sys.stdin)
except Exception:
    raise SystemExit(0)
for a in apps:
    n=a.get("name") or ""
    if n.startswith("criclab-video") and n not in ("criclab-video-api","criclab-video-worker"):
        print(n)
')
  pm2 start "${ROOT}/deploy/ecosystem.config.cjs"
fi
pm2 save

ok=""
for _ in $(seq 1 30); do
  if curl -fsS --max-time 2 http://127.0.0.1:8001/health >/dev/null 2>&1; then
    ok=1
    break
  fi
  sleep 2
done

if [[ -z "$ok" ]]; then
  echo "API did not listen on 127.0.0.1:8001:" >&2
  pm2 describe criclab-video-api || true
  pm2 logs criclab-video-api --nostream --lines 80 || true
  pm2 logs criclab-video-worker --nostream --lines 80 || true
  exit 1
fi

curl -fsS --max-time 10 http://127.0.0.1:8001/health
echo
echo "criclab-video-api and workers restarted from ${ROOT}"