"""
tests/test_core.py
Unit tests for ingestion, chunking, calculator, and confidence scoring.
These tests do NOT require OpenAI, Redis, or Neo4j — all external calls are mocked.
"""
from __future__ import annotations

import json
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from app.core.models import (
    Chunk,
    ChunkMetadata,
    ConfidenceBreakdown,
    DatasetType,
    EvidenceItem,
    GraphEdge,
)
from app.ingestion.metadata_extractor import detect_domain, detect_year, detect_dataset_type
from app.ingestion.chunker import token_count, _split_sentences, _build_overlap
from app.tools.calculator import calculate
from app.decision.confidence import score_confidence, explain_confidence


# ── metadata extractor ────────────────────────────────────────────────────────

class TestMetadataExtractor:
    def test_detect_domain_transport(self):
        text = "The electric vehicle bus fleet reduced diesel consumption by 40%."
        assert detect_domain(text) == "transport"

    def test_detect_domain_energy(self):
        text = "Solar and wind power now account for 30% of grid capacity."
        assert detect_domain(text) == "energy"

    def test_detect_domain_climate(self):
        text = "CO2 emissions must reach net zero by 2050 according to the IPCC."
        assert detect_domain(text) in ("climate", "energy")

    def test_detect_year_in_text(self):
        text = "This 2023 study found significant improvements in EV battery technology."
        assert detect_year(text) == 2023

    def test_detect_year_in_filename(self):
        assert detect_year("", "annual_report_2022.pdf") == 2022

    def test_detect_year_none(self):
        assert detect_year("No year mentioned here.") is None

    def test_detect_dataset_type_research(self):
        text = "This peer-reviewed study uses a novel methodology to assess outcomes."
        assert detect_dataset_type(text) == DatasetType.research

    def test_detect_dataset_type_report(self):
        text = "Annual whitepaper reviewing market performance."
        assert detect_dataset_type(text) == DatasetType.report

    def test_detect_dataset_type_csv(self):
        assert detect_dataset_type("", "fleet_data.csv") == DatasetType.dataset


# ── chunker ───────────────────────────────────────────────────────────────────

class TestChunker:
    def test_token_count_nonzero(self):
        assert token_count("Hello world, this is a test sentence.") > 0

    def test_split_sentences_basic(self):
        text   = "First sentence. Second sentence. Third sentence."
        result = _split_sentences(text)
        assert len(result) >= 2
        assert all(isinstance(s, str) for s in result)

    def test_split_paragraphs(self):
        text   = "Paragraph one.\n\nParagraph two.\n\nParagraph three."
        result = _split_sentences(text)
        assert len(result) >= 3

    def test_build_overlap_within_budget(self):
        units  = ["sentence one", "sentence two", "sentence three"]
        result = _build_overlap(units, overlap_tokens=20)
        assert isinstance(result, list)
        assert len(result) <= len(units)

    def test_chunk_document_produces_chunks(self):
        from app.ingestion.loader import RawDocument
        from app.ingestion.chunker import chunk_document
        doc = RawDocument(
            doc_id   = "test_doc",
            source   = "test.txt",
            text     = ("This is a test sentence about electric vehicles. " * 50),
            page_map = [],
        )
        chunks = chunk_document(doc, domain="transport", year=2023)
        assert len(chunks) >= 1
        assert all(c.tokens > 0 for c in chunks)
        assert all(c.metadata.doc_id == "test_doc" for c in chunks)
        assert all(c.metadata.domain == "transport" for c in chunks)

    def test_chunk_no_empty_chunks(self):
        from app.ingestion.loader import RawDocument
        from app.ingestion.chunker import chunk_document
        doc = RawDocument(
            doc_id="doc2", source="doc2.txt",
            text="Short text.\n\n" * 10,
            page_map=[],
        )
        chunks = chunk_document(doc)
        assert all(len(c.text.strip()) > 0 for c in chunks)

    def test_chunk_metadata_complete(self):
        from app.ingestion.loader import RawDocument
        from app.ingestion.chunker import chunk_document
        doc    = RawDocument(doc_id="d3", source="d3.pdf", text="Test " * 200, page_map=[])
        chunks = chunk_document(doc, domain="energy", year=2022)
        for c in chunks:
            assert c.metadata.chunk_id
            assert c.metadata.source == "d3.pdf"
            assert c.metadata.domain == "energy"
            assert c.metadata.year   == 2022


# ── calculator ────────────────────────────────────────────────────────────────

class TestCalculator:
    def test_basic_arithmetic(self):
        assert calculate("2 + 2")["result"] == 4
        assert calculate("10 * 5")["result"] == 50
        assert calculate("100 / 4")["result"] == 25.0

    def test_compound_expression(self):
        r = calculate("40 * 180000 * 1.15")
        assert r["error"] is None
        assert abs(r["result"] - 8280000.0) < 0.01

    def test_power(self):
        assert calculate("2 ** 10")["result"] == 1024

    def test_safe_function_round(self):
        r = calculate("round(3.14159, 2)")
        assert r["result"] == 3.14
        assert r["error"] is None

    def test_safe_function_sqrt(self):
        r = calculate("sqrt(144)")
        assert r["result"] == 12.0

    def test_division_by_zero(self):
        r = calculate("1 / 0")
        assert r["error"] is not None

    def test_unsafe_expression_rejected(self):
        r = calculate("__import__('os').system('ls')")
        assert r["error"] is not None

    def test_negative_numbers(self):
        assert calculate("-5 + 10")["result"] == 5

    def test_nested_expression(self):
        r = calculate("(100 - 20) * 1.3 + 50")
        assert r["error"] is None
        assert abs(r["result"] - 154.0) < 0.01


# ── confidence scoring ────────────────────────────────────────────────────────

def _make_evidence(n: int, score: float = 0.8, source_prefix: str = "src") -> list:
    return [
        EvidenceItem(
            citation_id    = f"[E{i}]",
            chunk_id       = f"chunk_{i}",
            source         = f"{source_prefix}_{i}",
            text           = "Evidence text.",
            score          = score,
            retrieval_type = "hybrid",
        )
        for i in range(n)
    ]


class TestConfidenceScoring:
    def test_high_confidence_with_strong_evidence(self):
        evidence = _make_evidence(3, score=0.9)
        c = score_confidence(evidence, [], graph_agreement=0.9, llm_self_score=0.9)
        assert c.overall >= 0.6

    def test_low_confidence_with_no_evidence(self):
        c = score_confidence([], [], graph_agreement=0.0, llm_self_score=0.4)
        assert c.overall < 0.3

    def test_confidence_bounded(self):
        evidence = _make_evidence(5, score=1.0)
        c = score_confidence(evidence, [], graph_agreement=1.0, llm_self_score=1.0)
        assert 0.0 <= c.overall <= 1.0

    def test_diversity_capped_at_three_sources(self):
        e1 = _make_evidence(5, score=0.8, source_prefix="same_source")
        # all from same source prefix → low diversity
        e2 = [
            EvidenceItem(citation_id=f"[E{i}]", chunk_id=f"c{i}",
                         source=f"unique_{i}", text=".", score=0.8, retrieval_type="hybrid")
            for i in range(5)
        ]
        c1 = score_confidence(e1, [], 0.5)
        c2 = score_confidence(e2, [], 0.5)
        assert c2.evidence_diversity >= c1.evidence_diversity

    def test_explain_confidence_returns_string(self):
        evidence = _make_evidence(2, score=0.75)
        c        = score_confidence(evidence, [], 0.6)
        text     = explain_confidence(c, evidence)
        assert isinstance(text, str)
        assert "confidence" in text.lower()

    def test_explain_confidence_mentions_sources(self):
        evidence = _make_evidence(3, score=0.8)
        c        = score_confidence(evidence, [], 0.7)
        text     = explain_confidence(c, evidence)
        assert "3" in text or "source" in text.lower()


# ── models validation ─────────────────────────────────────────────────────────

class TestModels:
    def test_chunk_roundtrip_json(self):
        meta  = ChunkMetadata(doc_id="d1", chunk_id="d1_c0001", source="doc.pdf")
        chunk = Chunk(metadata=meta, text="Some text.", tokens=4)
        data  = chunk.model_dump()
        restored = Chunk.model_validate(data)
        assert restored.text == "Some text."
        assert restored.metadata.chunk_id == "d1_c0001"

    def test_evidence_item_defaults(self):
        e = EvidenceItem(
            citation_id="[E1]", chunk_id="c1", source="src.pdf",
            text="Text.", score=0.85, retrieval_type="hybrid",
        )
        assert e.domain is None
        assert e.year   is None

    def test_graph_edge_model(self):
        edge = GraphEdge(subject="EV Bus", predicate="emits_low", obj="CO2", confidence=0.9)
        assert edge.subject == "EV Bus"
