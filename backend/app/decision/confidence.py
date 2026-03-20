"""
app/decision/confidence.py
Standalone confidence scorer + explainability text generator.

Used by the decision engine and can be called independently for
the /confidence endpoint or UI panel.
"""
from __future__ import annotations

from typing import List, Optional

from app.core.models import ConfidenceBreakdown, EvidenceItem, GraphEdge


def score_confidence(
    evidence:         List[EvidenceItem],
    graph_edges:      List[GraphEdge],
    graph_agreement:  float,
    llm_self_score:   float = 0.7,
) -> ConfidenceBreakdown:
    """
    Weighted confidence score from four signals:
      40% — retrieval strength  (avg hybrid score of evidence items)
      30% — graph agreement     (edge convergence on same conclusion)
      20% — evidence diversity  (unique source count, capped at 3)
      10% — LLM self-reported   (from the LLM's own confidence_overall field)
    """
    retrieval_strength = (
        sum(e.score for e in evidence) / len(evidence) if evidence else 0.0
    )
    unique_sources = len({e.source for e in evidence})
    diversity      = min(1.0, unique_sources / 3.0)

    overall = (
        0.40 * retrieval_strength
        + 0.30 * graph_agreement
        + 0.20 * diversity
        + 0.10 * llm_self_score
    )

    return ConfidenceBreakdown(
        overall             = round(min(overall, 1.0), 3),
        retrieval_strength  = round(retrieval_strength, 3),
        graph_agreement     = round(graph_agreement, 3),
        evidence_diversity  = round(diversity, 3),
    )


def explain_confidence(breakdown: ConfidenceBreakdown, evidence: List[EvidenceItem]) -> str:
    """
    Returns a human-readable explanation of why the confidence score is what it is.
    Used in the UI explainability panel.
    """
    lines = [f"Overall confidence: {breakdown.overall * 100:.0f}%\n"]

    # retrieval
    r = breakdown.retrieval_strength
    if r >= 0.8:
        lines.append(f"• Strong retrieval ({r*100:.0f}%): highly relevant evidence found.")
    elif r >= 0.5:
        lines.append(f"• Moderate retrieval ({r*100:.0f}%): some relevant evidence found.")
    else:
        lines.append(f"• Weak retrieval ({r*100:.0f}%): limited relevant evidence found.")

    # graph
    g = breakdown.graph_agreement
    if g >= 0.7:
        lines.append(f"• High graph agreement ({g*100:.0f}%): multiple knowledge edges converge on the same conclusion.")
    elif g >= 0.4:
        lines.append(f"• Moderate graph agreement ({g*100:.0f}%): some supporting graph edges found.")
    else:
        lines.append(f"• Low graph agreement ({g*100:.0f}%): graph edges are sparse or contradictory.")

    # diversity
    d = breakdown.evidence_diversity
    unique = len({e.source for e in evidence})
    if d >= 0.9:
        lines.append(f"• High source diversity ({unique} independent sources).")
    elif d >= 0.5:
        lines.append(f"• Moderate source diversity ({unique} sources).")
    else:
        lines.append(f"• Low source diversity (only {unique} source). Consider adding more documents.")

    return "\n".join(lines)
