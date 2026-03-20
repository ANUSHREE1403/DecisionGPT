"""
app/core/models.py
Shared Pydantic models used throughout the pipeline.
"""
from __future__ import annotations

from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


# ─────────────────────────────────────────────
# Ingestion
# ─────────────────────────────────────────────

class DatasetType(str, Enum):
    research = "research"
    report   = "report"
    dataset  = "dataset"
    policy   = "policy"
    news     = "news"
    other    = "other"


class ChunkMetadata(BaseModel):
    doc_id:       str
    chunk_id:     str
    source:       str                        # filename or URL
    domain:       Optional[str]  = None      # e.g. "transport", "energy"
    year:         Optional[int]  = None
    dataset_type: DatasetType    = DatasetType.other
    page_range:   Optional[str]  = None      # "12-14" for PDFs
    char_offset:  Optional[int]  = None


class Chunk(BaseModel):
    metadata: ChunkMetadata
    text:     str
    tokens:   int


# ─────────────────────────────────────────────
# Retrieval
# ─────────────────────────────────────────────

class EvidenceItem(BaseModel):
    citation_id:  str                        # "[E1]", "[E2]" …
    chunk_id:     str
    source:       str
    domain:       Optional[str] = None
    year:         Optional[int] = None
    text:         str
    score:        float                      # hybrid retrieval score 0–1
    retrieval_type: str = "hybrid"           # "vector" | "bm25" | "hybrid"


class GraphEdge(BaseModel):
    subject:      str
    predicate:    str
    obj:          str
    confidence:   float
    source_chunk_id: Optional[str] = None
    domain:       Optional[str] = None
    year:         Optional[int] = None


# ─────────────────────────────────────────────
# Decision output
# ─────────────────────────────────────────────

class Tradeoff(BaseModel):
    sign:   str        # "+" | "-" | "~"
    text:   str
    citations: List[str] = []


class ConfidenceBreakdown(BaseModel):
    overall:           float    # 0–1
    retrieval_strength: float
    graph_agreement:   float
    evidence_diversity: float


class DecisionReport(BaseModel):
    query:             str
    domain:            Optional[str]
    evidence_sources:  List[EvidenceItem]
    graph_edges:       List[GraphEdge]
    data_insights:     Optional[str]     = None
    tradeoffs:         List[Tradeoff]    = []
    final_recommendation: str
    citations:         Dict[str, str]    = {}   # {"[E1]": "Source name …"}
    confidence:        ConfidenceBreakdown
    pipeline_trace:    Dict[str, Any]    = {}   # stage timing + ids


# ─────────────────────────────────────────────
# API request / response
# ─────────────────────────────────────────────

class QueryRequest(BaseModel):
    question: str = Field(..., min_length=10, max_length=2000)
    domain:   Optional[str]  = None
    year_min: Optional[int]  = None
    year_max: Optional[int]  = None
    top_k:    int            = Field(default=8, ge=1, le=20)


class QueryResponse(BaseModel):
    report:       DecisionReport
    cache_hit:    bool
    latency_ms:   float
