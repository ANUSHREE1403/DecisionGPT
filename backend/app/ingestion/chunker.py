"""
app/ingestion/chunker.py
Token-aware semantic chunker.

Strategy:
  1. Split on paragraph/sentence boundaries first (semantic).
  2. Merge small pieces until we hit CHUNK_SIZE_TOKENS.
  3. Add CHUNK_OVERLAP_TOKENS from the previous chunk's tail.

Uses tiktoken for token counting (same tokeniser as OpenAI embeddings).
"""
from __future__ import annotations

import re
import uuid
from typing import List, Optional, Tuple

import tiktoken
from loguru import logger

from app.core.config import get_settings
from app.core.models import Chunk, ChunkMetadata, DatasetType
from app.ingestion.loader import RawDocument


# ── tokeniser (shared, lazy-loaded) ──────────────────────────────────────────

_enc: Optional[tiktoken.Encoding] = None


def _get_enc() -> tiktoken.Encoding:
    global _enc
    if _enc is None:
        _enc = tiktoken.get_encoding("cl100k_base")
    return _enc


def token_count(text: str) -> int:
    return len(_get_enc().encode(text))


# ── sentence/paragraph splitter ──────────────────────────────────────────────

_PARA_SEP = re.compile(r"\n{2,}")
_SENT_END  = re.compile(r"(?<=[.!?])\s+")


def _split_sentences(text: str) -> List[str]:
    """Split into paragraphs first, then sentences within each paragraph."""
    units: List[str] = []
    for para in _PARA_SEP.split(text):
        para = para.strip()
        if not para:
            continue
        sentences = _SENT_END.split(para)
        units.extend(s.strip() for s in sentences if s.strip())
    return units


# ── page range helper ─────────────────────────────────────────────────────────

def _page_for_offset(offset: int, page_map: List[tuple]) -> Optional[int]:
    """Return 1-based page number for a character offset."""
    for start, end, page_no in page_map:
        if start <= offset < end:
            return page_no
    return None


# ── core chunker ──────────────────────────────────────────────────────────────

def chunk_document(
    doc: RawDocument,
    domain:       Optional[str]  = None,
    year:         Optional[int]  = None,
    dataset_type: DatasetType    = DatasetType.other,
) -> List[Chunk]:
    settings    = get_settings()
    max_tokens  = settings.chunk_size_tokens
    overlap_tok = settings.chunk_overlap_tokens

    units = _split_sentences(doc.text)
    if not units:
        logger.warning("No text units found in document {}", doc.doc_id)
        return []

    chunks: List[Chunk] = []
    buffer: List[str]   = []
    buf_tokens: int     = 0
    char_offset: int    = 0

    def _flush(buf: List[str], offset: int) -> None:
        text = " ".join(buf).strip()
        if not text:
            return
        tok = token_count(text)

        # resolve page range for PDFs
        page_range: Optional[str] = None
        if doc.page_map:
            first_page = _page_for_offset(offset, doc.page_map)
            last_page  = _page_for_offset(offset + len(text), doc.page_map)
            if first_page and last_page:
                page_range = f"{first_page}" if first_page == last_page else f"{first_page}-{last_page}"

        chunk_id = f"{doc.doc_id}_c{len(chunks):04d}"
        chunks.append(
            Chunk(
                metadata=ChunkMetadata(
                    doc_id       = doc.doc_id,
                    chunk_id     = chunk_id,
                    source       = doc.source,
                    domain       = domain,
                    year         = year,
                    dataset_type = dataset_type,
                    page_range   = page_range,
                    char_offset  = offset,
                ),
                text   = text,
                tokens = tok,
            )
        )

    overlap_tail: List[str] = []   # units carried over from previous chunk

    for unit in units:
        unit_tok = token_count(unit)

        # single unit larger than max — hard split on words
        if unit_tok > max_tokens:
            if buffer:
                _flush(buffer, char_offset)
                buffer    = list(overlap_tail)
                buf_tokens = sum(token_count(u) for u in buffer)
            words = unit.split()
            sub   = []
            sub_t = 0
            for w in words:
                wt = token_count(w + " ")
                if sub_t + wt > max_tokens and sub:
                    _flush(sub, char_offset)
                    sub   = []
                    sub_t = 0
                sub.append(w)
                sub_t += wt
            if sub:
                buffer    = sub
                buf_tokens = sub_t
            char_offset += len(unit)
            continue

        if buf_tokens + unit_tok > max_tokens and buffer:
            _flush(buffer, char_offset)
            # carry overlap
            overlap_tail = _build_overlap(buffer, overlap_tok)
            buffer    = list(overlap_tail)
            buf_tokens = sum(token_count(u) for u in buffer)

        buffer.append(unit)
        buf_tokens += unit_tok
        char_offset += len(unit)

    if buffer:
        _flush(buffer, char_offset)

    logger.debug(
        "Chunked {} → {} chunks (avg {:.0f} tokens)",
        doc.source,
        len(chunks),
        sum(c.tokens for c in chunks) / max(len(chunks), 1),
    )
    return chunks


def _build_overlap(units: List[str], overlap_tokens: int) -> List[str]:
    """Take units from the END of the buffer to form an overlap tail."""
    tail: List[str] = []
    tail_tokens = 0
    for unit in reversed(units):
        ut = token_count(unit)
        if tail_tokens + ut > overlap_tokens:
            break
        tail.insert(0, unit)
        tail_tokens += ut
    return tail
