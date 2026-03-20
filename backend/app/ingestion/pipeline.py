"""
app/ingestion/pipeline.py
Orchestrates: load → extract metadata → chunk → write processed JSON.

CLI usage:
    python -m app.ingestion.pipeline --input data/raw --output data/processed
    python -m app.ingestion.pipeline --file data/raw/report.pdf --domain energy --year 2023
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import List, Optional

from loguru import logger

from app.core.models import Chunk, DatasetType
from app.ingestion.chunker import chunk_document
from app.ingestion.loader import load_directory, load_file
from app.ingestion.metadata_extractor import extract_metadata


def ingest_file(
    path:         Path,
    output_dir:   Path,
    domain:       Optional[str]       = None,
    year:         Optional[int]       = None,
    dataset_type: Optional[DatasetType] = None,
) -> List[Chunk]:
    doc = load_file(path)
    if not doc:
        return []

    meta = extract_metadata(
        text         = doc.text[:3000],
        filename     = doc.source,
        domain       = domain,
        year         = year,
        dataset_type = dataset_type,
    )

    chunks = chunk_document(
        doc          = doc,
        domain       = meta["domain"],
        year         = meta["year"],
        dataset_type = meta["dataset_type"],
    )

    if not chunks:
        logger.warning("No chunks produced for {}", path.name)
        return []

    _write_chunks(chunks, output_dir, doc.doc_id)
    logger.info("Ingested {} → {} chunks | domain={} year={}", path.name, len(chunks), meta["domain"], meta["year"])
    return chunks


def ingest_directory(
    input_dir:  Path,
    output_dir: Path,
) -> List[Chunk]:
    output_dir.mkdir(parents=True, exist_ok=True)
    all_chunks: List[Chunk] = []

    for path in sorted(input_dir.iterdir()):
        if path.is_file():
            chunks = ingest_file(path, output_dir)
            all_chunks.extend(chunks)

    logger.info("Total chunks produced: {}", len(all_chunks))
    return all_chunks


def _write_chunks(chunks: List[Chunk], output_dir: Path, doc_id: str) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / f"{doc_id}.json"
    payload  = [c.model_dump() for c in chunks]
    out_path.write_text(json.dumps(payload, indent=2, default=str))
    logger.debug("Wrote {} chunks → {}", len(chunks), out_path)


def load_processed_chunks(processed_dir: Path) -> List[Chunk]:
    """Read all processed JSON files back into Chunk objects."""
    chunks: List[Chunk] = []
    for p in sorted(processed_dir.glob("*.json")):
        data = json.loads(p.read_text())
        for item in data:
            chunks.append(Chunk.model_validate(item))
    logger.info("Loaded {} chunks from {}", len(chunks), processed_dir)
    return chunks


# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="DecisionGPT ingestion pipeline")
    parser.add_argument("--input",   type=Path, default=Path("data/raw"))
    parser.add_argument("--output",  type=Path, default=Path("data/processed"))
    parser.add_argument("--file",    type=Path, default=None)
    parser.add_argument("--domain",  type=str,  default=None)
    parser.add_argument("--year",    type=int,  default=None)
    args = parser.parse_args()

    from app.core.logging import setup_logging
    setup_logging()

    if args.file:
        ingest_file(args.file, args.output, domain=args.domain, year=args.year)
    else:
        ingest_directory(args.input, args.output)
