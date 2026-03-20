"""
app/embeddings/vector_store.py
FAISS-backed vector store.

  build_index(chunks)         → persists index + metadata JSON
  VectorRetriever.retrieve()  → returns top-k chunks with scores
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import faiss
import numpy as np
from loguru import logger

from app.core.config import get_settings
from app.core.models import Chunk
from app.embeddings.embedder import get_embedder


# ── index builder ─────────────────────────────────────────────────────────────

def build_index(chunks: List[Chunk]) -> None:
    settings  = get_settings()
    idx_path  = Path(settings.faiss_index_path)
    meta_path = Path(settings.faiss_metadata_path)
    idx_path.parent.mkdir(parents=True, exist_ok=True)

    texts = [c.text for c in chunks]
    logger.info("Embedding {} chunks for FAISS index …", len(texts))
    embedder = get_embedder()
    vecs     = embedder.embed_many(texts)   # (N, D)

    # L2-normalise → inner product == cosine similarity
    faiss.normalize_L2(vecs)
    dim   = vecs.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(vecs)

    faiss.write_index(index, str(idx_path))
    logger.info("FAISS index saved → {} ({} vectors)", idx_path, index.ntotal)

    # metadata list — same order as index rows
    metadata = [c.metadata.model_dump() for c in chunks]
    texts_map = {i: chunks[i].text for i in range(len(chunks))}

    payload = {"metadata": metadata, "texts": [c.text for c in chunks]}
    meta_path.write_text(json.dumps(payload, default=str))
    logger.info("Metadata saved → {}", meta_path)


# ── retriever ─────────────────────────────────────────────────────────────────

class VectorRetriever:
    def __init__(self) -> None:
        settings        = get_settings()
        self._idx_path  = Path(settings.faiss_index_path)
        self._meta_path = Path(settings.faiss_metadata_path)
        self._index: Optional[faiss.Index] = None
        self._texts:    List[str]          = []
        self._metadata: List[dict]         = []
        self._embedder  = get_embedder()

    def _load(self) -> None:
        if self._index is not None:
            return
        if not self._idx_path.exists():
            raise FileNotFoundError(f"FAISS index not found at {self._idx_path}. Run build_index first.")
        self._index = faiss.read_index(str(self._idx_path))
        payload     = json.loads(self._meta_path.read_text())
        self._metadata = payload["metadata"]
        self._texts    = payload["texts"]
        logger.info("FAISS index loaded ({} vectors)", self._index.ntotal)

    def retrieve(
        self,
        query:    str,
        top_k:    int = 8,
        filters:  Optional[Dict] = None,
    ) -> List[Tuple[dict, str, float]]:
        """
        Returns list of (metadata_dict, text, score) sorted by score desc.
        filters: dict of metadata field → value(s) to include.
        """
        self._load()

        qvec = self._embedder.embed_one(query).reshape(1, -1).astype(np.float32)
        faiss.normalize_L2(qvec)

        # over-fetch to allow post-filter
        fetch_k = min(top_k * 4, self._index.ntotal)
        scores, indices = self._index.search(qvec, fetch_k)

        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < 0:
                continue
            meta = self._metadata[idx]
            if filters and not _passes_filter(meta, filters):
                continue
            results.append((meta, self._texts[idx], float(score)))
            if len(results) >= top_k:
                break

        logger.debug("Vector retrieve → {} results for query '{:.60s}'", len(results), query)
        return results


def _passes_filter(meta: dict, filters: dict) -> bool:
    for key, val in filters.items():
        mval = meta.get(key)
        if isinstance(val, list):
            if mval not in val:
                return False
        elif mval != val:
            return False
    return True


# singleton
_retriever: Optional[VectorRetriever] = None


def get_vector_retriever() -> VectorRetriever:
    global _retriever
    if _retriever is None:
        _retriever = VectorRetriever()
    return _retriever
