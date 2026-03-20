"""
app/ingestion/loader.py
Loads raw files into a list of (text, metadata) tuples.
Supported: PDF (pymupdf), CSV (pandas), TXT.
"""
from __future__ import annotations

import csv
import hashlib
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

import fitz  # pymupdf
import pandas as pd
from loguru import logger


@dataclass
class RawDocument:
    """One logical document before chunking."""
    doc_id:   str
    source:   str          # original filename
    text:     str
    page_map: List[tuple]  # [(start_char, end_char, page_no), …]  — PDFs only
    metadata: dict = field(default_factory=dict)


# ── helpers ──────────────────────────────────────────────────────────────────

def _make_doc_id(path: Path) -> str:
    return hashlib.md5(path.name.encode()).hexdigest()[:12]


def _clean(text: str) -> str:
    """Remove excessive whitespace and non-printable chars."""
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"[^\x20-\x7E\n]", "", text)
    return text.strip()


# ── loaders ──────────────────────────────────────────────────────────────────

def load_pdf(path: Path) -> RawDocument:
    doc_id   = _make_doc_id(path)
    full_text = ""
    page_map: List[tuple] = []

    with fitz.open(str(path)) as pdf:
        for page_no, page in enumerate(pdf, start=1):
            page_text = page.get_text("text")
            start     = len(full_text)
            full_text += page_text + "\n"
            end        = len(full_text)
            page_map.append((start, end, page_no))

    logger.debug("PDF loaded | {} | pages={} | chars={}", path.name, len(page_map), len(full_text))
    return RawDocument(
        doc_id   = doc_id,
        source   = path.name,
        text     = _clean(full_text),
        page_map = page_map,
    )


def load_txt(path: Path) -> RawDocument:
    text = path.read_text(encoding="utf-8", errors="ignore")
    logger.debug("TXT loaded | {} | chars={}", path.name, len(text))
    return RawDocument(
        doc_id   = _make_doc_id(path),
        source   = path.name,
        text     = _clean(text),
        page_map = [],
    )


def load_csv(path: Path, text_columns: Optional[List[str]] = None) -> RawDocument:
    """
    Concatenates all text columns into a single document.
    If text_columns is None, uses all non-numeric columns.
    """
    df = pd.read_csv(path, dtype=str).fillna("")

    if text_columns:
        cols = [c for c in text_columns if c in df.columns]
    else:
        cols = df.select_dtypes(include="object").columns.tolist()

    rows = []
    for _, row in df.iterrows():
        row_text = " | ".join(f"{c}: {row[c]}" for c in cols if row[c].strip())
        if row_text:
            rows.append(row_text)

    combined = "\n".join(rows)
    logger.debug("CSV loaded | {} | rows={} | chars={}", path.name, len(rows), len(combined))
    return RawDocument(
        doc_id   = _make_doc_id(path),
        source   = path.name,
        text     = _clean(combined),
        page_map = [],
        metadata = {"csv_columns": cols, "row_count": len(rows)},
    )


# ── public interface ──────────────────────────────────────────────────────────

LOADERS = {
    ".pdf": load_pdf,
    ".txt": load_txt,
    ".csv": load_csv,
}


def load_file(path: Path) -> Optional[RawDocument]:
    suffix = path.suffix.lower()
    loader = LOADERS.get(suffix)
    if not loader:
        logger.warning("Unsupported file type skipped: {}", path.name)
        return None
    try:
        return loader(path)
    except Exception as exc:
        logger.error("Failed to load {} — {}", path.name, exc)
        return None


def load_directory(directory: Path) -> List[RawDocument]:
    docs = []
    for path in sorted(directory.iterdir()):
        if path.is_file() and path.suffix.lower() in LOADERS:
            doc = load_file(path)
            if doc:
                docs.append(doc)
    logger.info("Loaded {} documents from {}", len(docs), directory)
    return docs
