#!/usr/bin/env bash
# scripts/ingest.sh
# Ingest documents, build FAISS index, and build Neo4j knowledge graph.
# Usage: bash scripts/ingest.sh [--skip-graph]

set -euo pipefail
source .venv/bin/activate
export PYTHONPATH=backend

SKIP_GRAPH=false
for arg in "$@"; do
  [[ "$arg" == "--skip-graph" ]] && SKIP_GRAPH=true
done

echo ""
echo "╔══════════════════════════════════════╗"
echo "║     DecisionGPT — Ingestion          ║"
echo "╚══════════════════════════════════════╝"
echo ""

# ── Step 1: Ingest documents ──────────────────────────────────────────────────
echo "→  Step 1/3: Ingesting documents from data/raw/ ..."
python -m app.ingestion.pipeline --input data/raw --output data/processed
echo "✅  Ingestion complete"

# ── Step 2: Build FAISS index ─────────────────────────────────────────────────
echo "→  Step 2/3: Building FAISS vector index ..."
python - <<'PYEOF'
from pathlib import Path
from app.core.logging import setup_logging
from app.embeddings.vector_store import build_index
from app.ingestion.pipeline import load_processed_chunks

setup_logging()
chunks = load_processed_chunks(Path("data/processed"))
print(f"   Loaded {len(chunks)} chunks")
build_index(chunks)
print("   FAISS index built ✓")
PYEOF
echo "✅  FAISS index ready"

# ── Step 3: Build Neo4j graph ─────────────────────────────────────────────────
if [ "$SKIP_GRAPH" = false ]; then
  echo "→  Step 3/3: Building Neo4j knowledge graph (this may take a few minutes) ..."
  python -m app.graph.builder
  echo "✅  Knowledge graph ready"
else
  echo "⏭   Skipping graph build (--skip-graph flag)"
fi

echo ""
echo "✅  All done! Run: bash scripts/run.sh"
echo ""
