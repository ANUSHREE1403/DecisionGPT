"""
app/pipeline/orchestrator.py
Full DecisionGPT pipeline:
  CAG lookup → RAG hybrid retrieval → KAG graph retrieval → LLM decision → CAG store

Returns QueryResponse with the decision report + trace metadata.
"""
from __future__ import annotations

import time
from typing import Optional

from loguru import logger

from app.cache.cag import cache_lookup, cache_store
from app.core.models import DecisionReport, QueryRequest, QueryResponse
from app.decision.engine import generate_decision
from app.graph.retriever import get_graph_retriever
from app.retrieval.hybrid import get_hybrid_retriever


async def run_pipeline(request: QueryRequest) -> QueryResponse:
    t_start = time.perf_counter()
    trace   = {}

    # ── 1. CAG lookup ─────────────────────────────────────────────────────────
    t0       = time.perf_counter()
    cached   = cache_lookup(request.question, request.domain)
    trace["cag_ms"] = round((time.perf_counter() - t0) * 1000, 1)

    if cached:
        latency_ms = round((time.perf_counter() - t_start) * 1000, 1)
        cached.pipeline_trace = {**trace, "cache_hit": True, "total_ms": latency_ms}
        return QueryResponse(report=cached, cache_hit=True, latency_ms=latency_ms)

    trace["cache_hit"] = False

    # ── 2. RAG hybrid retrieval ───────────────────────────────────────────────
    t0 = time.perf_counter()
    filters = {}
    if request.domain:
        filters["domain"] = request.domain
    if request.year_min or request.year_max:
        # year filtering handled post-retrieval for simplicity
        pass

    retriever  = get_hybrid_retriever()
    evidence   = retriever.retrieve(request.question, top_k=request.top_k, filters=filters or None)

    # optional year filter
    if request.year_min or request.year_max:
        evidence = [
            e for e in evidence
            if _year_in_range(e.year, request.year_min, request.year_max)
        ]

    trace["rag_ms"]      = round((time.perf_counter() - t0) * 1000, 1)
    trace["rag_chunks"]  = [e.chunk_id for e in evidence]
    logger.info("RAG → {} evidence items in {}ms", len(evidence), trace["rag_ms"])

    # ── 3. KAG graph retrieval ────────────────────────────────────────────────
    t0 = time.perf_counter()
    graph_edges, graph_agreement = [], 0.0
    try:
        graph_retriever            = get_graph_retriever()
        graph_edges, graph_agreement = graph_retriever.retrieve(request.question, domain=request.domain)
    except Exception as exc:
        logger.warning("KAG retrieval failed (non-fatal): {}", exc)
        graph_agreement = 0.0

    trace["kag_ms"]         = round((time.perf_counter() - t0) * 1000, 1)
    trace["kag_edge_count"] = len(graph_edges)
    logger.info("KAG → {} edges in {}ms | agreement={:.2f}", len(graph_edges), trace["kag_ms"], graph_agreement)

    # ── 4. LLM Decision Generation ────────────────────────────────────────────
    t0 = time.perf_counter()
    report: DecisionReport = generate_decision(
        query           = request.question,
        evidence        = evidence,
        graph_edges     = graph_edges,
        graph_agreement = graph_agreement,
        domain          = request.domain,
    )
    trace["llm_ms"] = round((time.perf_counter() - t0) * 1000, 1)
    logger.info("LLM decision in {}ms", trace["llm_ms"])

    # ── 5. CAG store ──────────────────────────────────────────────────────────
    try:
        cache_store(request.question, request.domain, report)
    except Exception as exc:
        logger.warning("CAG store failed (non-fatal): {}", exc)

    # ── 6. Attach trace ───────────────────────────────────────────────────────
    total_ms = round((time.perf_counter() - t_start) * 1000, 1)
    trace["total_ms"] = total_ms
    report.pipeline_trace = trace

    return QueryResponse(report=report, cache_hit=False, latency_ms=total_ms)


def _year_in_range(year: Optional[int], year_min: Optional[int], year_max: Optional[int]) -> bool:
    if year is None:
        return True   # don't exclude undated evidence
    if year_min and year < year_min:
        return False
    if year_max and year > year_max:
        return False
    return True
