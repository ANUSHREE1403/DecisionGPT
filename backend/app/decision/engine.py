"""
app/decision/engine.py
LLM Decision Engine.

Takes composed context (evidence + graph edges) and produces a DecisionReport
using a structured prompt that enforces citation, tradeoffs, and confidence.
"""
from __future__ import annotations

import json
import re
from typing import Dict, List, Optional

from loguru import logger
from openai import OpenAI
from tenacity import retry, stop_after_attempt, wait_exponential

from app.core.config import get_settings
from app.core.models import (
    ConfidenceBreakdown,
    DecisionReport,
    EvidenceItem,
    GraphEdge,
    Tradeoff,
)

# ── system prompt ─────────────────────────────────────────────────────────────

_SYSTEM_PROMPT = """You are DecisionGPT — an evidence-based decision engine.
You reason from provided evidence only. You NEVER invent facts.
Every factual claim in your answer MUST include a citation tag like [E1] or [E2].
If evidence is insufficient, say so explicitly and lower your confidence.

Output ONLY valid JSON matching this schema exactly:
{
  "data_insights": "string or null — computed numeric insights from evidence",
  "tradeoffs": [
    {"sign": "+", "text": "pro point", "citations": ["[E1]"]},
    {"sign": "-", "text": "con point", "citations": ["[E2]"]},
    {"sign": "~", "text": "uncertainty", "citations": []}
  ],
  "final_recommendation": "string — actionable recommendation with citation tags",
  "confidence_overall": 0.82
}"""

_USER_TEMPLATE = """QUESTION: {question}

DOMAIN: {domain}

EVIDENCE CHUNKS:
{evidence_block}

KNOWLEDGE GRAPH EDGES:
{graph_block}

Based on the above evidence only, produce the decision report JSON."""


def _format_evidence(items: List[EvidenceItem]) -> str:
    lines = []
    for item in items:
        lines.append(
            f"{item.citation_id} [{item.source}] (score={item.score:.2f})\n{item.text}"
        )
    return "\n\n".join(lines) if lines else "No evidence retrieved."


def _format_graph(edges: List[GraphEdge]) -> str:
    if not edges:
        return "No graph edges retrieved."
    return "\n".join(
        f"  {e.subject} --{e.predicate}--> {e.obj} (conf={e.confidence:.2f})"
        for e in edges
    )


# ── LLM call ─────────────────────────────────────────────────────────────────

@retry(stop=stop_after_attempt(3), wait=wait_exponential(min=1, max=10))
def _call_llm(messages: list, client: OpenAI, model: str, max_tokens: int) -> str:
    response = client.chat.completions.create(
        model       = model,
        temperature = 0.1,
        max_tokens  = max_tokens,
        messages    = messages,
        response_format={"type": "json_object"},
    )
    return response.choices[0].message.content.strip()


# ── confidence computation ────────────────────────────────────────────────────

def _compute_confidence(
    evidence:         List[EvidenceItem],
    graph_agreement:  float,
    llm_confidence:   float,
) -> ConfidenceBreakdown:
    if evidence:
        retrieval_strength = min(1.0, sum(e.score for e in evidence) / len(evidence))
    else:
        retrieval_strength = 0.0

    unique_sources = len({e.source for e in evidence})
    diversity      = min(1.0, unique_sources / 3.0)   # 3+ sources = full diversity

    overall = round(
        0.4 * retrieval_strength
        + 0.3 * graph_agreement
        + 0.2 * diversity
        + 0.1 * llm_confidence,
        3,
    )

    return ConfidenceBreakdown(
        overall            = overall,
        retrieval_strength = round(retrieval_strength, 3),
        graph_agreement    = round(graph_agreement, 3),
        evidence_diversity = round(diversity, 3),
    )


# ── public interface ──────────────────────────────────────────────────────────

def generate_decision(
    query:          str,
    evidence:       List[EvidenceItem],
    graph_edges:    List[GraphEdge],
    graph_agreement: float,
    domain:         Optional[str] = None,
) -> DecisionReport:
    settings = get_settings()
    client   = OpenAI(api_key=settings.openai_api_key)

    user_msg = _USER_TEMPLATE.format(
        question       = query,
        domain         = domain or "general",
        evidence_block = _format_evidence(evidence),
        graph_block    = _format_graph(graph_edges),
    )

    messages = [
        {"role": "system", "content": _SYSTEM_PROMPT},
        {"role": "user",   "content": user_msg},
    ]

    raw = _call_llm(messages, client, settings.openai_chat_model, settings.openai_max_tokens)

    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError as exc:
        logger.error("LLM returned invalid JSON: {} — raw: {:.200s}", exc, raw)
        parsed = {
            "data_insights": None,
            "tradeoffs": [],
            "final_recommendation": raw,
            "confidence_overall": 0.4,
        }

    tradeoffs = [
        Tradeoff(
            sign      = t.get("sign", "~"),
            text      = t.get("text", ""),
            citations = t.get("citations", []),
        )
        for t in parsed.get("tradeoffs", [])
    ]

    citations: Dict[str, str] = {item.citation_id: item.source for item in evidence}

    llm_conf = float(parsed.get("confidence_overall", 0.7))
    confidence = _compute_confidence(evidence, graph_agreement, llm_conf)

    report = DecisionReport(
        query                = query,
        domain               = domain,
        evidence_sources     = evidence,
        graph_edges          = graph_edges,
        data_insights        = parsed.get("data_insights"),
        tradeoffs            = tradeoffs,
        final_recommendation = parsed.get("final_recommendation", ""),
        citations            = citations,
        confidence           = confidence,
    )

    logger.info(
        "Decision generated | confidence={} | evidence={} | edges={}",
        confidence.overall, len(evidence), len(graph_edges),
    )
    return report
