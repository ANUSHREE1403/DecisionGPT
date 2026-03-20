"""
app/graph/builder.py
KAG ETL: extract (subject, predicate, object) triples from chunks via LLM,
then MERGE into Neo4j with provenance.

Run once after ingestion:
    python -m app.graph.builder
"""
from __future__ import annotations

import json
from typing import List, Optional

from loguru import logger
from neo4j import GraphDatabase
from openai import OpenAI
from tenacity import retry, stop_after_attempt, wait_exponential

from app.core.config import get_settings
from app.core.models import Chunk, GraphEdge

# ── Cypher ────────────────────────────────────────────────────────────────────

_MERGE_CYPHER = """
MERGE (s:Entity {name: $subject})
MERGE (o:Entity {name: $obj})
MERGE (s)-[r:RELATION {predicate: $predicate}]->(o)
ON CREATE SET
    r.confidence      = $confidence,
    r.source_chunk_id = $source_chunk_id,
    r.domain          = $domain,
    r.year            = $year,
    r.created_at      = timestamp()
ON MATCH SET
    r.confidence      = CASE WHEN $confidence > r.confidence THEN $confidence ELSE r.confidence END
"""

# ── LLM triple extraction ─────────────────────────────────────────────────────

_EXTRACT_PROMPT = """Extract factual knowledge triples from the following text.
Return ONLY a JSON array of objects with keys: subject, predicate, object, confidence (0-1).
Use short, canonical names (e.g. "EV Bus", "CO2", "upfront_cost").
Allowed predicates: emits_low, emits_high, has_high, has_low, leads_to, tradeoff_with,
affects_cost, requires, reduces, increases, compared_to, policy_mandates.

Text:
{text}

Return ONLY the JSON array, no markdown, no explanation."""


@retry(stop=stop_after_attempt(3), wait=wait_exponential(min=1, max=8))
def _extract_triples(text: str, client: OpenAI, model: str) -> List[dict]:
    response = client.chat.completions.create(
        model=model,
        temperature=0,
        messages=[
            {"role": "system", "content": "You are a knowledge graph extraction engine. Output only valid JSON."},
            {"role": "user",   "content": _EXTRACT_PROMPT.format(text=text[:2000])},
        ],
    )
    raw = response.choices[0].message.content.strip()
    # strip markdown fences if present
    raw = raw.removeprefix("```json").removeprefix("```").removesuffix("```").strip()
    return json.loads(raw)


# ── Neo4j driver ─────────────────────────────────────────────────────────────

def _get_driver():
    settings = get_settings()
    return GraphDatabase.driver(
        settings.neo4j_uri,
        auth=(settings.neo4j_user, settings.neo4j_password),
    )


def _store_triple(tx, triple: dict, meta: dict) -> None:
    tx.run(
        _MERGE_CYPHER,
        subject        = triple["subject"],
        predicate      = triple["predicate"],
        obj            = triple["object"],
        confidence     = float(triple.get("confidence", 0.8)),
        source_chunk_id= meta.get("chunk_id", ""),
        domain         = meta.get("domain", ""),
        year           = meta.get("year"),
    )


# ── public interface ──────────────────────────────────────────────────────────

def build_graph(chunks: List[Chunk]) -> int:
    """
    Process all chunks, extract triples, MERGE into Neo4j.
    Returns total triples stored.
    """
    settings = get_settings()
    client   = OpenAI(api_key=settings.openai_api_key)
    driver   = _get_driver()
    total    = 0

    with driver.session() as session:
        for chunk in chunks:
            try:
                triples = _extract_triples(chunk.text, client, settings.openai_chat_model)
            except Exception as exc:
                logger.warning("Triple extraction failed for {} — {}", chunk.metadata.chunk_id, exc)
                continue

            for triple in triples:
                try:
                    session.execute_write(
                        _store_triple,
                        triple,
                        chunk.metadata.model_dump(),
                    )
                    total += 1
                except Exception as exc:
                    logger.warning("Neo4j write failed: {} — {}", triple, exc)

    logger.info("Graph build complete — {} triples stored", total)
    driver.close()
    return total


if __name__ == "__main__":
    from app.core.logging import setup_logging
    from app.ingestion.pipeline import load_processed_chunks
    from pathlib import Path

    setup_logging()
    settings = get_settings()
    chunks   = load_processed_chunks(Path(settings.processed_data_path))
    build_graph(chunks)
