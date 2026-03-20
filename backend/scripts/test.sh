#!/usr/bin/env bash
# scripts/test.sh
set -euo pipefail
source .venv/bin/activate
export PYTHONPATH=backend

echo "→  Running DecisionGPT test suite..."
pytest backend/tests/ -v --tb=short --cov=app --cov-report=term-missing "$@"
