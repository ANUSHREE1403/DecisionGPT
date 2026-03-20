"""
app/graph/retriever.py
KAG retrieval: translate user query → entity names → Cypher → GraphEdge list.
Also computes an "agreement signal" (how many edges support the same conclusion).
"""
from __future__ import annotations

import json
import re
from typing import Dict, List, Optional, Tuple

from loguru import logger
from neo4j import GraphDatabase
from openai import OpenAI
from tenacity import retry, stop_after_attempt, wait_exponential

from app.core.config import get_settings
from app.core.models import GraphEdge

# ── entity extraction from query ──────────────────────────────────────────────

_ENTITY_PROMPT = """From the following question, extract the key entities and concepts
that should be searched for in a knowledge graph (technology names, metrics, outcomes, policies).
Return ONLY a JSON array of short strings. Max 6 entities.

Question: {query}"""


@retry(stop=stop_after_attempt(2), wait=wait_exponential(min=1, max=4))
def _extract_entities(query: str, client: OpenAI, model: str) -> List[str]:
    response = client.chat.completions.create(
        model=model,
        temperature=0,
        messages=[
            {"role": "system", "content": "Extract entities. Return only a JSON array of strings."},
            {"role": "user",   "content": _ENTITY_PROMPT.format(query=query)},
        ],
    )
    raw = response.choices[0].message.content.strip()
    raw = raw.removeprefix("```json").removeprefix("```").removesuffix("```").strip()
    return json.loads(raw)


# ── Cypher queries ────────────────────────────────────────────────────────────

_ENTITY_SUBGRAPH_CYPHER = """
MATCH (s:Entity)-[r:RELATION]->(o:Entity)
WHERE toLower(s.name) CONTAINS toLower($entity)
   OR toLower(o.name) CONTAINS toLower($entity)
RETURN s.name AS subject, r.predicate AS predicate, o.name AS obj,
       r.confidence AS confidence, r.source_chunk_id AS source_chunk_id,
       r.domain AS domain, r.year AS year
ORDER BY r.confidence DESC
LIMIT 30
"""

_AGREEMENT_CYPHER = """
MATCH (s:Entity)-[r:RELATION]->(o:Entity)
WHERE r.predicate IN $predicates
RETURN r.predicate AS predicate, count(r) AS support
ORDER BY support DESC
"""


# ── public interface ──────────────────────────────────────────────────────────

class GraphRetriever:
    def __init__(self) -> None:
        settings     = get_settings()
        self._driver = GraphDatabase.driver(
            settings.neo4j_uri,
            auth=(settings.neo4j_user, settings.neo4j_password),
        )
        self._client = OpenAI(api_key=settings.openai_api_key)
        self._model  = settings.openai_chat_model

    def retrieve(
        self,
        query:  str,
        domain: Optional[str] = None,
        limit:  int = 15,
    ) -> Tuple[List[GraphEdge], float]:
        """
        Returns (edges, agreement_score).
        agreement_score: 0–1 indicating how many edges converge on the same conclusion.
        """
        try:
            entities = _extract_entities(query, self._client, self._model)
        except Exception as exc:
            logger.warning("Entity extraction failed: {}. Falling back to regex.", exc)
            entities = _regex_entities(query)

        logger.debug("KAG entities: {}", entities)

        edges: List[GraphEdge] = []
        seen = set()

        with self._driver.session() as session:
            for entity in entities:
                rows = session.run(_ENTITY_SUBGRAPH_CYPHER, entity=entity).data()
                for row in rows:
                    key = (row["subject"], row["predicate"], row["obj"])
                    if key in seen:
                        continue
                    seen.add(key)
                    edges.append(
                        GraphEdge(
                            subject         = row["subject"],
                            predicate       = row["predicate"],
                            obj             = row["obj"],
                            confidence      = float(row.get("confidence") or 0.7),
                            source_chunk_id = row.get("source_chunk_id"),
                            domain          = row.get("domain"),
                            year            = row.get("year"),
                        )
                    )

        # dedupe + sort by confidence
        edges = sorted(edges, key=lambda e: e.confidence, reverse=True)[:limit]
        agreement = _compute_agreement(edges)

        logger.debug("KAG retrieve → {} edges | agreement={:.2f}", len(edges), agreement)
        return edges, agreement

    def close(self) -> None:
        self._driver.close()


def _compute_agreement(edges: List[GraphEdge]) -> float:
    """Agreement = fraction of edges sharing the most common predicate."""
    if not edges:
        return 0.0
    from collections import Counter
    counts = Counter(e.predicate for e in edges)
    top    = counts.most_common(1)[0][1]
    return round(top / len(edges), 3)


def _regex_entities(query: str) -> List[str]:
    """Fallback: extract capitalised words / known keywords."""
    words = re.findall(r"\b[A-Z][a-z]+(?:\s[A-Z][a-z]+)*\b", query)
    lower_kw = ["ev", "diesel", "bus", "cost", "emission", "co2", "carbon"]
    for kw in lower_kw:
        if kw in query.lower():
            words.append(kw)
    return list(set(words))[:6]


# singleton
_graph_retriever: Optional[GraphRetriever] = None


def get_graph_retriever() -> GraphRetriever:
    global _graph_retriever
    if _graph_retriever is None:
        _graph_retriever = GraphRetriever()
    return _graph_retriever
