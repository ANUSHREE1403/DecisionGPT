"""
app/main.py
FastAPI application entry point.

Routes:
  GET  /health          → health check
  POST /query           → run full pipeline, return DecisionReport
  POST /ingest          → upload + ingest a document
  POST /index/build     → (re)build FAISS index from processed chunks
  POST /graph/build     → (re)build Neo4j graph from processed chunks
  GET  /graph/edges     → query graph edges for a term
"""
from __future__ import annotations

import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import List, Optional

from fastapi import FastAPI, File, Form, HTTPException, UploadFile, status
from fastapi.middleware.cors import CORSMiddleware
from loguru import logger

from app.core.config import get_settings
from app.core.logging import setup_logging
from app.core.models import GraphEdge, QueryRequest, QueryResponse


# ── lifespan ──────────────────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    setup_logging()
    settings = get_settings()
    logger.info("DecisionGPT starting | env={}", settings.app_env)
    yield
    logger.info("DecisionGPT shutting down")


# ── app ───────────────────────────────────────────────────────────────────────

settings = get_settings()

app = FastAPI(
    title        = "DecisionGPT API",
    description  = "Evidence-based decision engine (RAG + CAG + KAG)",
    version      = "0.1.0",
    lifespan     = lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins     = settings.cors_origins,
    allow_credentials = True,
    allow_methods     = ["*"],
    allow_headers     = ["*"],
)


# ── routes ────────────────────────────────────────────────────────────────────

@app.get("/health", tags=["system"])
async def health():
    return {"status": "ok", "timestamp": time.time(), "version": "0.1.0"}


@app.post("/query", response_model=QueryResponse, tags=["pipeline"])
async def query(request: QueryRequest):
    """
    Run the full DecisionGPT pipeline:
    CAG → RAG → KAG → LLM → structured DecisionReport
    """
    from app.pipeline.orchestrator import run_pipeline
    try:
        return await run_pipeline(request)
    except FileNotFoundError as exc:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"Index not ready: {exc}. Run /index/build first.",
        )
    except Exception as exc:
        logger.exception("Pipeline error: {}", exc)
        raise HTTPException(status_code=500, detail=str(exc))


@app.post("/ingest", tags=["ingestion"])
async def ingest_document(
    file:         UploadFile = File(...),
    domain:       Optional[str] = Form(None),
    year:         Optional[int] = Form(None),
    dataset_type: Optional[str] = Form(None),
):
    """Upload and ingest a PDF, TXT, or CSV document."""
    from app.ingestion.pipeline import ingest_file
    from app.core.models import DatasetType

    suffix = Path(file.filename).suffix.lower()
    if suffix not in {".pdf", ".txt", ".csv"}:
        raise HTTPException(400, "Unsupported file type. Use PDF, TXT, or CSV.")

    tmp_path = Path(settings.raw_data_path) / file.filename
    tmp_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path.write_bytes(await file.read())

    dtype = DatasetType(dataset_type) if dataset_type else None
    chunks = ingest_file(tmp_path, Path(settings.processed_data_path), domain=domain, year=year, dataset_type=dtype)

    return {
        "filename":    file.filename,
        "chunks":      len(chunks),
        "doc_id":      chunks[0].metadata.doc_id if chunks else None,
        "domain":      chunks[0].metadata.domain if chunks else None,
        "year":        chunks[0].metadata.year if chunks else None,
    }


@app.post("/index/build", tags=["ingestion"])
async def build_index():
    """(Re)build the FAISS vector index from all processed chunks."""
    from app.embeddings.vector_store import build_index
    from app.ingestion.pipeline import load_processed_chunks

    chunks = load_processed_chunks(Path(settings.processed_data_path))
    if not chunks:
        raise HTTPException(400, "No processed chunks found. Ingest documents first.")

    build_index(chunks)
    return {"status": "ok", "chunks_indexed": len(chunks)}


@app.post("/graph/build", tags=["graph"])
async def build_graph():
    """(Re)build the Neo4j knowledge graph from all processed chunks."""
    from app.graph.builder import build_graph as _build_graph
    from app.ingestion.pipeline import load_processed_chunks

    chunks  = load_processed_chunks(Path(settings.processed_data_path))
    triples = _build_graph(chunks)
    return {"status": "ok", "triples_stored": triples}


@app.get("/graph/edges", response_model=List[GraphEdge], tags=["graph"])
async def graph_edges(query: str, limit: int = 20):
    """Return graph edges matching a query term (for graph visualisation)."""
    from app.graph.retriever import get_graph_retriever
    retriever         = get_graph_retriever()
    edges, _agreement = retriever.retrieve(query, limit=limit)
    return edges
