#!/usr/bin/env sh
set -eu
cd "$(dirname "$0")/.."
if [ ! -x .venv/bin/python ]; then
  echo '먼저 python3 -m venv .venv 와 .venv/bin/pip install -r backend/requirements.txt 를 실행해 주세요.'
  exit 1
fi
exec .venv/bin/python -m uvicorn backend.app:app --host 127.0.0.1 --port "${PORT:-8787}"
