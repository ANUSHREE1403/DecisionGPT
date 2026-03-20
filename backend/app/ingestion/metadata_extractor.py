"""
app/ingestion/metadata_extractor.py
Heuristic metadata extraction from filename + text snippet.
Detects: domain, year, dataset_type.
Can be overridden by caller-supplied values.
"""
from __future__ import annotations

import re
from pathlib import Path
from typing import Optional

from app.core.models import DatasetType

# ── domain keyword map ────────────────────────────────────────────────────────

_DOMAIN_KEYWORDS: dict[str, list[str]] = {
    "transport":  ["bus", "vehicle", "transit", "fleet", "diesel", "electric vehicle", "ev", "road"],
    "energy":     ["solar", "wind", "grid", "renewable", "coal", "nuclear", "power plant", "kwh"],
    "healthcare": ["hospital", "patient", "drug", "clinical", "treatment", "mortality", "nhs", "fda"],
    "finance":    ["gdp", "inflation", "interest rate", "investment", "revenue", "capex", "opex"],
    "policy":     ["regulation", "legislation", "law", "act", "compliance", "government", "policy"],
    "climate":    ["co2", "carbon", "emission", "greenhouse", "ipcc", "net zero", "temperature"],
    "education":  ["school", "student", "university", "curriculum", "learning", "exam"],
}

# ── dataset type keywords ─────────────────────────────────────────────────────

_TYPE_KEYWORDS: dict[DatasetType, list[str]] = {
    DatasetType.research: ["study", "research", "journal", "findings", "methodology", "abstract"],
    DatasetType.report:   ["report", "annual", "whitepaper", "review", "assessment"],
    DatasetType.dataset:  [".csv", "dataset", "data table", "statistics", "survey results"],
    DatasetType.policy:   ["policy", "regulation", "directive", "guidance", "legislation"],
    DatasetType.news:     ["news", "press release", "announcement", "breaking"],
}

_YEAR_PATTERN = re.compile(r"\b(19[8-9]\d|20[0-2]\d)\b")


# ── public functions ──────────────────────────────────────────────────────────

def detect_domain(text: str, filename: str = "") -> Optional[str]:
    combined = (filename + " " + text[:2000]).lower()
    scores: dict[str, int] = {}
    for domain, keywords in _DOMAIN_KEYWORDS.items():
        score = sum(combined.count(kw) for kw in keywords)
        if score > 0:
            scores[domain] = score
    return max(scores, key=scores.get) if scores else None


def detect_year(text: str, filename: str = "") -> Optional[int]:
    combined = filename + " " + text[:3000]
    matches  = _YEAR_PATTERN.findall(combined)
    if not matches:
        return None
    # return the most-frequently-appearing year
    from collections import Counter
    return int(Counter(matches).most_common(1)[0][0])


def detect_dataset_type(text: str, filename: str = "") -> DatasetType:
    combined = (filename + " " + text[:1000]).lower()
    for dtype, keywords in _TYPE_KEYWORDS.items():
        if any(kw in combined for kw in keywords):
            return dtype
    return DatasetType.other


def extract_metadata(
    text:     str,
    filename: str,
    domain:   Optional[str]  = None,
    year:     Optional[int]  = None,
    dataset_type: Optional[DatasetType] = None,
) -> dict:
    """
    Return enriched metadata dict, using auto-detected values as fallbacks.
    Caller-supplied values always take precedence.
    """
    return {
        "domain":       domain       or detect_domain(text, filename),
        "year":         year         or detect_year(text, filename),
        "dataset_type": dataset_type or detect_dataset_type(text, filename),
    }
