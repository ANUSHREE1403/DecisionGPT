# DecisionGPT

Evidence-based decision engine using **RAG + CAG + KAG** with a Python/FastAPI backend and React frontend.

---

## Architecture

```
User Query
    │
    ▼
┌───────────────────────────────────────────────────┐
│  FastAPI  POST /query                             │
│                                                   │
│  1. CAG ──── Redis exact cache                    │
│              Redis semantic cache (embeddings)    │
│                                                   │
│  2. RAG ──── BM25 keyword index                   │
│              FAISS vector index (OpenAI embeds)   │
│              RRF fusion → top-k EvidenceItems     │
│                                                   │
│  3. KAG ──── Neo4j graph query                    │
│              Entity extraction → Cypher           │
│              → GraphEdge list + agreement score   │
│                                                   │
│  4. LLM ──── GPT-4o structured output            │
│              Evidence block + graph context       │
│              → tradeoffs, recommendation, conf    │
│                                                   │
│  5. CAG ──── Store result for future hits         │
└───────────────────────────────────────────────────┘
    │
    ▼
DecisionReport JSON → React Frontend
```

---

## Quick Start

### Prerequisites
- Python 3.11+
- Docker Desktop (for Redis + Neo4j)
- OpenAI API key

### 1. Setup
```bash
git clone <repo>
cd decisiongpt
bash scripts/setup.sh
```

### 2. Add your OpenAI key
```bash
# Edit .env
OPENAI_API_KEY=sk-...
```

### 3. Add documents
```bash
cp your_reports.pdf data/raw/
cp your_data.csv    data/raw/
```

### 4. Ingest + index
```bash
bash scripts/ingest.sh
# To skip graph build (faster for testing):
bash scripts/ingest.sh --skip-graph
```

### 5. Run the API
```bash
bash scripts/run.sh
# API: http://localhost:8000
# Docs: http://localhost:8000/docs
```

---

## API Endpoints

| Method | Path | Description |
|--------|------|-------------|
| GET | `/health` | Health check |
| POST | `/query` | Run full pipeline → DecisionReport |
| POST | `/ingest` | Upload + ingest a document |
| POST | `/index/build` | Rebuild FAISS index |
| POST | `/graph/build` | Rebuild Neo4j graph |
| GET | `/graph/edges` | Query graph edges |

### Example query
```bash
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{
    "question": "Should a city transit authority switch from diesel to EV buses?",
    "domain": "transport",
    "top_k": 8
  }'
```

### Response shape
```json
{
  "cache_hit": false,
  "latency_ms": 1420,
  "report": {
    "query": "...",
    "domain": "transport",
    "evidence_sources": [...],
    "graph_edges": [...],
    "tradeoffs": [
      {"sign": "+", "text": "68% lower lifecycle CO2", "citations": ["[E1]"]},
      {"sign": "-", "text": "2.1x higher upfront cost", "citations": ["[E2]"]}
    ],
    "final_recommendation": "...",
    "citations": {"[E1]": "DOE Report 2023", "[E2]": "Bloomberg NEF 2022"},
    "confidence": {
      "overall": 0.82,
      "retrieval_strength": 0.87,
      "graph_agreement": 0.80,
      "evidence_diversity": 0.74
    },
    "pipeline_trace": {
      "cag_ms": 4, "rag_ms": 310, "kag_ms": 220, "llm_ms": 890,
      "total_ms": 1420, "cache_hit": false
    }
  }
}
```

---

## Project Structure

```
decisiongpt/
├── backend/
│   ├── app/
│   │   ├── main.py               # FastAPI app + all routes
│   │   ├── core/
│   │   │   ├── config.py         # Pydantic Settings
│   │   │   ├── logging.py        # Loguru setup
│   │   │   └── models.py         # All shared data models
│   │   ├── ingestion/
│   │   │   ├── loader.py         # PDF / CSV / TXT loader
│   │   │   ├── chunker.py        # Token-aware semantic chunker
│   │   │   ├── metadata_extractor.py
│   │   │   └── pipeline.py       # CLI orchestrator
│   │   ├── embeddings/
│   │   │   ├── embedder.py       # OpenAI embeddings
│   │   │   └── vector_store.py   # FAISS index
│   │   ├── retrieval/
│   │   │   └── hybrid.py         # BM25 + vector RRF fusion
│   │   ├── cache/
│   │   │   └── cag.py            # Exact + semantic Redis cache
│   │   ├── graph/
│   │   │   ├── builder.py        # LLM triple extraction → Neo4j
│   │   │   └── retriever.py      # Cypher query + agreement score
│   │   ├── decision/
│   │   │   ├── engine.py         # LLM decision generation
│   │   │   └── confidence.py     # Scoring + explainability
│   │   ├── pipeline/
│   │   │   └── orchestrator.py   # Full CAG→RAG→KAG→LLM pipeline
│   │   └── tools/
│   │       ├── calculator.py     # Safe AST arithmetic
│   │       └── dataset_query.py  # DuckDB/pandas CSV queries
│   └── tests/
│       └── test_core.py
├── data/
│   ├── raw/                      # Drop source files here
│   ├── processed/                # Chunked JSON output
│   ├── indexes/                  # FAISS index files
│   └── graph_exports/
├── docker/
│   └── docker-compose.yml        # Redis + Neo4j
├── scripts/
│   ├── setup.sh                  # One-shot setup
│   ├── ingest.sh                 # Ingest → index → graph
│   ├── run.sh                    # Start API
│   └── test.sh                   # Run test suite
├── .env.example
└── pyproject.toml
```

---

## Running Tests

```bash
bash scripts/test.sh
# or with coverage report:
bash scripts/test.sh --cov-report=html
```

Tests cover: metadata extraction, chunking, calculator, confidence scoring, and model validation. No external services required for the test suite.

---

## Milestones Completed

- [x] M0 — Repo structure, config, logging
- [x] M1 — Document ingestion (PDF, CSV, TXT)
- [x] M2 — Embeddings + FAISS vector store
- [x] M3 — CAG exact + semantic Redis cache
- [x] M4 — Hybrid BM25 + vector retrieval (RRF)
- [x] M5 — KAG graph construction (Neo4j ETL)
- [x] M6 — KAG graph retrieval (Cypher + agreement)
- [x] M8 — LLM decision engine (structured output)
- [x] M9 — Full pipeline orchestration
- [x] M10 — FastAPI endpoints
- [x] M11 — Confidence scoring + explainability
- [x] M12 — Tools (calculator + dataset query)
- [ ] M13 — Evaluation harness (next)

---

## Environment Variables Reference

| Variable | Default | Description |
|----------|---------|-------------|
| `OPENAI_API_KEY` | required | Your OpenAI key |
| `OPENAI_EMBEDDING_MODEL` | `text-embedding-3-small` | Embedding model |
| `OPENAI_CHAT_MODEL` | `gpt-4o` | Decision LLM |
| `REDIS_URL` | `redis://localhost:6379/0` | Redis connection |
| `NEO4J_URI` | `bolt://localhost:7687` | Neo4j Bolt URI |
| `NEO4J_PASSWORD` | `decisiongpt` | Neo4j password |
| `CHUNK_SIZE_TOKENS` | `700` | Target chunk size |
| `RETRIEVAL_TOP_K` | `8` | Chunks to retrieve |
| `HYBRID_ALPHA` | `0.6` | Vector weight in RRF |
| `SEMANTIC_CACHE_THRESHOLD` | `0.92` | Semantic cache hit threshold |
