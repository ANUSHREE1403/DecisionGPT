#!/usr/bin/env bash
# scripts/run.sh
# Start DecisionGPT FastAPI server
# Usage: bash scripts/run.sh [--prod]

set -euo pipefail
source .venv/bin/activate
export PYTHONPATH=backend

RELOAD="--reload"
PORT=8000
HOST=0.0.0.0

for arg in "$@"; do
  [[ "$arg" == "--prod" ]] && RELOAD=""
done

echo ""
echo "╔══════════════════════════════════════════════╗"
echo "║   DecisionGPT API starting                   ║"
echo "║   http://localhost:$PORT                       ║"
echo "║   Docs: http://localhost:$PORT/docs            ║"
echo "╚══════════════════════════════════════════════╝"
echo ""

uvicorn app.main:app --host $HOST --port $PORT $RELOAD --log-level info
