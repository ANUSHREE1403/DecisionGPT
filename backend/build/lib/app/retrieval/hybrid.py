"""
app/retrieval/hybrid.py
Hybrid retriever: BM25 (keyword) + Vector (FAISS) fused via Reciprocal Rank Fusion.

  HybridRetriever.retrieve(query, top_k, filters) → List[EvidenceItem]
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional

from loguru import logger
from rank_bm25 import BM25Okapi

from app.core.config import get_settings
from app.core.models import EvidenceItem
from app.embeddings.vector_store import get_vector_retriever


def _tokenise(text: str) -> List[str]:
    return text.lower().split()


def _rrf_score(rank: int, k: int = 60) -> float:
    return 1.0 / (k + rank + 1)


class HybridRetriever:
    def __init__(self) -> None:
        self._vector   = get_vector_retriever()
        self._bm25: Optional[BM25Okapi] = None
        self._corpus:   List[str]   = []
        self._meta_list: List[dict] = []
        self._loaded    = False

    def _load_bm25(self) -> None:
        """Build BM25 index from the same metadata store used by FAISS."""
        if self._loaded:
            return
        settings  = get_settings()
        meta_path = Path(settings.faiss_metadata_path)
        if not meta_path.exists():
            raise FileNotFoundError(f"Metadata not found at {meta_path}. Run build_index first.")

        payload          = json.loads(meta_path.read_text())
        self._corpus     = payload["texts"]
        self._meta_list  = payload["metadata"]
        tokenised        = [_tokenise(t) for t in self._corpus]
        self._bm25       = BM25Okapi(tokenised)
        self._loaded     = True
        logger.info("BM25 index built over {} documents", len(self._corpus))

    def retrieve(
        self,
        query:   str,
        top_k:   int            = 8,
        filters: Optional[Dict] = None,
    ) -> List[EvidenceItem]:
        self._load_bm25()
        settings = get_settings()

        # ── BM25 top candidates ───────────────────────────────────────────────
        bm25_scores    = self._bm25.get_scores(_tokenise(query))
        bm25_ranked    = sorted(range(len(bm25_scores)), key=lambda i: bm25_scores[i], reverse=True)

        # ── Vector top candidates ─────────────────────────────────────────────
        vec_results    = self._vector.retrieve(query, top_k=top_k * 2, filters=filters)
        # vec_results: [(meta, text, score), ...]
        vec_idx_map    = {}    # original corpus idx → (meta, text, vec_score)
        for meta, text, score in vec_results:
            # find corpus position by matching text (fast enough for ≤10k chunks)
            for i, ct in enumerate(self._corpus):
                if ct == text:
                    vec_idx_map[i] = (meta, text, score)
                    break

        vec_ranked     = list(vec_idx_map.keys())

        # ── RRF fusion ────────────────────────────────────────────────────────
        rrf: Dict[int, float] = {}
        for rank, idx in enumerate(bm25_ranked[: top_k * 4]):
            rrf[idx] = rrf.get(idx, 0.0) + _rrf_score(rank)
        for rank, idx in enumerate(vec_ranked):
            rrf[idx] = rrf.get(idx, 0.0) + _rrf_score(rank)

        sorted_ids = sorted(rrf, key=rrf.__getitem__, reverse=True)

        # ── Build EvidenceItem list ───────────────────────────────────────────
        items: List[EvidenceItem] = []
        cite_counter = 1
        for idx in sorted_ids:
            if len(items) >= top_k:
                break
            meta = self._meta_list[idx]
            if filters and not _passes_filter(meta, filters):
                continue
            text = self._corpus[idx]
            items.append(
                EvidenceItem(
                    citation_id    = f"[E{cite_counter}]",
                    chunk_id       = meta.get("chunk_id", ""),
                    source         = meta.get("source", ""),
                    domain         = meta.get("domain"),
                    year           = meta.get("year"),
                    text           = text,
                    score          = round(rrf[idx], 4),
                    retrieval_type = "hybrid",
                )
            )
            cite_counter += 1

        logger.debug("Hybrid retrieve → {} items for '{:.60s}'", len(items), query)
        return items


def _passes_filter(meta: dict, filters: dict) -> bool:
    for key, val in filters.items():
        mval = meta.get(key)
        if isinstance(val, list):
            if mval not in val:
                return False
        elif mval != val:
            return False
    return True


_retriever: Optional[HybridRetriever] = None


def get_hybrid_retriever() -> HybridRetriever:
    global _retriever
    if _retriever is None:
        _retriever = HybridRetriever()
    return _retriever
