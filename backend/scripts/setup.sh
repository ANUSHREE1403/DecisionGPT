#!/usr/bin/env bash
# scripts/setup.sh
# One-shot setup for DecisionGPT backend
# Usage: bash scripts/setup.sh

set -euo pipefail

echo ""
echo "╔══════════════════════════════════════╗"
echo "║       DecisionGPT — Setup            ║"
echo "╚══════════════════════════════════════╝"
echo ""

# ── 1. Python version check ───────────────────────────────────────────────────
python_version=$(python3 --version 2>&1 | awk '{print $2}')
required="3.11"
if [[ "$(printf '%s\n' "$required" "$python_version" | sort -V | head -n1)" != "$required" ]]; then
  echo "❌  Python >= 3.11 required. Found: $python_version"
  exit 1
fi
echo "✅  Python $python_version"

# ── 2. Virtual env ────────────────────────────────────────────────────────────
if [ ! -d ".venv" ]; then
  echo "→  Creating virtual environment..."
  python3 -m venv .venv
fi
source .venv/bin/activate
echo "✅  Virtual env active"

# ── 3. Install dependencies ───────────────────────────────────────────────────
echo "→  Installing dependencies..."
pip install --quiet --upgrade pip
pip install --quiet -e ".[dev]"
echo "✅  Dependencies installed"

# ── 4. .env file ─────────────────────────────────────────────────────────────
if [ ! -f ".env" ]; then
  cp .env.example .env
  echo "⚠️   .env created from .env.example — add your OPENAI_API_KEY before running!"
else
  echo "✅  .env exists"
fi

# ── 5. Data directories ───────────────────────────────────────────────────────
mkdir -p data/raw data/processed data/indexes logs
echo "✅  Data directories ready"

# ── 6. Docker services ────────────────────────────────────────────────────────
echo "→  Starting Docker services (Redis + Neo4j)..."
docker compose -f docker/docker-compose.yml up -d
echo "✅  Docker services started"
echo "   Redis:  localhost:6379"
echo "   Neo4j:  http://localhost:7474  (neo4j / decisiongpt)"

echo ""
echo "╔══════════════════════════════════════╗"
echo "║   Setup complete!                    ║"
echo "╠══════════════════════════════════════╣"
echo "║   Next steps:                        ║"
echo "║   1. Edit .env → add OPENAI_API_KEY  ║"
echo "║   2. Drop files in data/raw/         ║"
echo "║   3. bash scripts/ingest.sh          ║"
echo "║   4. bash scripts/run.sh             ║"
echo "╚══════════════════════════════════════╝"
echo ""
