"""
app/tools/dataset_query.py
Dataset query tool — run safe SQL-like queries over ingested CSV files using DuckDB.
Falls back to pandas describe() if DuckDB unavailable.

LLM tool schema:
  name: "dataset_query"
  parameters:
    sql:  string  — SELECT query against table named 'data'
    file: string  — filename in data/raw/ or data/processed/
"""
from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Dict, Optional

import pandas as pd
from loguru import logger

from app.core.config import get_settings

# Optional DuckDB — graceful fallback
try:
    import duckdb
    _DUCKDB_AVAILABLE = True
except ImportError:
    _DUCKDB_AVAILABLE = False
    logger.warning("DuckDB not installed — falling back to pandas for dataset queries")

_ALLOWED_SQL = re.compile(r"^\s*SELECT\b", re.IGNORECASE)


def _resolve_path(filename: str) -> Optional[Path]:
    settings = get_settings()
    for base in [settings.raw_data_path, settings.processed_data_path]:
        p = Path(base) / filename
        if p.exists():
            return p
    return None


def dataset_query(sql: str, file: str) -> Dict[str, Any]:
    """
    Execute a SELECT query over a CSV file.

    Returns:
        {"columns": [...], "rows": [...], "row_count": N, "error": None}
        or
        {"columns": [], "rows": [], "row_count": 0, "error": <message>}
    """
    if not _ALLOWED_SQL.match(sql):
        return {"columns": [], "rows": [], "row_count": 0, "error": "Only SELECT queries allowed"}

    path = _resolve_path(file)
    if not path:
        return {"columns": [], "rows": [], "row_count": 0, "error": f"File not found: {file}"}
    if path.suffix.lower() != ".csv":
        return {"columns": [], "rows": [], "row_count": 0, "error": "Only CSV files supported"}

    try:
        if _DUCKDB_AVAILABLE:
            return _query_duckdb(sql, path)
        else:
            return _query_pandas(sql, path)
    except Exception as exc:
        logger.error("Dataset query failed: {}", exc)
        return {"columns": [], "rows": [], "row_count": 0, "error": str(exc)}


def _query_duckdb(sql: str, path: Path) -> Dict[str, Any]:
    con = duckdb.connect(database=":memory:")
    # Register CSV as 'data' table
    con.execute(f"CREATE TABLE data AS SELECT * FROM read_csv_auto('{path}')")
    rel     = con.execute(sql)
    columns = [desc[0] for desc in rel.description]
    rows    = rel.fetchall()
    con.close()
    logger.debug("DuckDB query OK | rows={} | file={}", len(rows), path.name)
    return {
        "columns":   columns,
        "rows":      [list(r) for r in rows[:500]],   # cap at 500 rows
        "row_count": len(rows),
        "error":     None,
    }


def _query_pandas(sql: str, path: Path) -> Dict[str, Any]:
    """Minimal fallback: just return describe() stats — no real SQL."""
    df      = pd.read_csv(path)
    stats   = df.describe(include="all").reset_index()
    columns = stats.columns.tolist()
    rows    = stats.values.tolist()
    logger.debug("Pandas fallback describe | file={}", path.name)
    return {
        "columns":   columns,
        "rows":      [[str(v) for v in row] for row in rows[:50]],
        "row_count": len(df),
        "note":      "DuckDB unavailable — returning describe() stats, not raw SQL result",
        "error":     None,
    }
