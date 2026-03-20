"""
app/cache/cag.py
CAG — Cache-Augmented Generation layer.

Two-level cache:
  1. Exact:    key = sha256(normalised_query + domain) → full DecisionReport JSON
  2. Semantic: stores (query_embedding, cache_key) pairs;
               on new query, finds nearest cached embedding — if cosine ≥ threshold → hit

Both layers backed by Redis.
"""
from __future__ import annotations

import hashlib
import json
import re
import struct
from typing import Optional

import numpy as np
import redis
from loguru import logger

from app.core.config import get_settings
from app.core.models import DecisionReport
from app.embeddings.embedder import get_embedder

# Redis key prefixes
_EXACT_PFX   = "decisiongpt:exact:"
_SEM_KEYS    = "decisiongpt:sem:keys"      # Redis list of cache keys
_SEM_VEC_PFX = "decisiongpt:sem:vec:"      # binary vector per key


def _get_redis() -> redis.Redis:
    settings = get_settings()
    return redis.from_url(settings.redis_url, decode_responses=False)


def _normalise(query: str) -> str:
    return re.sub(r"\s+", " ", query.strip().lower())


def _cache_key(query: str, domain: Optional[str]) -> str:
    raw = _normalise(query) + "|" + (domain or "")
    return hashlib.sha256(raw.encode()).hexdigest()


def _vec_to_bytes(vec: np.ndarray) -> bytes:
    return struct.pack(f"{len(vec)}f", *vec.tolist())


def _bytes_to_vec(b: bytes) -> np.ndarray:
    n = len(b) // 4
    return np.array(struct.unpack(f"{n}f", b), dtype=np.float32)


# ── public interface ──────────────────────────────────────────────────────────

def cache_lookup(query: str, domain: Optional[str] = None) -> Optional[DecisionReport]:
    """
    Returns a cached DecisionReport if found, else None.
    Checks exact cache first, then semantic cache.
    """
    settings = get_settings()
    r        = _get_redis()
    key      = _cache_key(query, domain)

    # 1. Exact hit
    raw = r.get(_EXACT_PFX + key)
    if raw:
        logger.info("CAG exact hit | key={:.12s}…", key)
        return DecisionReport.model_validate_json(raw)

    # 2. Semantic hit
    try:
        cached_keys: list = r.lrange(_SEM_KEYS, 0, -1)
        if cached_keys:
            qvec = get_embedder().embed_one(query)
            best_score, best_key = 0.0, None
            for ck in cached_keys:
                ck = ck.decode() if isinstance(ck, bytes) else ck
                vec_bytes = r.get(_SEM_VEC_PFX + ck)
                if not vec_bytes:
                    continue
                cvec  = _bytes_to_vec(vec_bytes)
                score = float(np.dot(qvec, cvec) / (np.linalg.norm(qvec) * np.linalg.norm(cvec) + 1e-9))
                if score > best_score:
                    best_score, best_key = score, ck

            if best_score >= settings.semantic_cache_threshold and best_key:
                raw = r.get(_EXACT_PFX + best_key)
                if raw:
                    logger.info("CAG semantic hit | score={:.3f} | key={:.12s}…", best_score, best_key)
                    return DecisionReport.model_validate_json(raw)
    except Exception as exc:
        logger.warning("Semantic cache lookup failed (non-fatal): {}", exc)

    logger.debug("CAG miss | key={:.12s}…", key)
    return None


def cache_store(query: str, domain: Optional[str], report: DecisionReport) -> None:
    """Persist a DecisionReport to both cache layers."""
    settings = get_settings()
    r        = _get_redis()
    key      = _cache_key(query, domain)
    ttl      = settings.cache_ttl_seconds

    # exact cache
    r.setex(_EXACT_PFX + key, ttl, report.model_dump_json())

    # semantic cache — store embedding vector
    try:
        qvec = get_embedder().embed_one(query)
        r.setex(_SEM_VEC_PFX + key, ttl, _vec_to_bytes(qvec))
        r.rpush(_SEM_KEYS, key)
        r.expire(_SEM_KEYS, ttl)
    except Exception as exc:
        logger.warning("Semantic cache store failed (non-fatal): {}", exc)

    logger.info("CAG stored | key={:.12s}… | ttl={}s", key, ttl)
