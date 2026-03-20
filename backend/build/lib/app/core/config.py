"""
app/core/config.py
Central settings — loaded once at startup from .env
"""
from functools import lru_cache
from typing import List

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # ── LLM + Embeddings ──────────────────────────────────────
    openai_api_key: str = Field(..., description="OpenAI API key")
    openai_embedding_model: str = "text-embedding-3-small"
    openai_chat_model: str = "gpt-4o"
    openai_max_tokens: int = 2048

    # ── Vector Store ─────────────────────────────────────────
    faiss_index_path: str = "data/indexes/faiss.index"
    faiss_metadata_path: str = "data/indexes/metadata.json"

    # ── Redis (CAG) ───────────────────────────────────────────
    redis_url: str = "redis://localhost:6379/0"
    cache_ttl_seconds: int = 3600
    semantic_cache_threshold: float = 0.92

    # ── Neo4j (KAG) ───────────────────────────────────────────
    neo4j_uri: str = "bolt://localhost:7687"
    neo4j_user: str = "neo4j"
    neo4j_password: str = "decisiongpt"

    # ── Ingestion ─────────────────────────────────────────────
    chunk_size_tokens: int = 700
    chunk_overlap_tokens: int = 70
    raw_data_path: str = "data/raw"
    processed_data_path: str = "data/processed"

    # ── Retrieval ─────────────────────────────────────────────
    retrieval_top_k: int = 8
    hybrid_alpha: float = 0.6          # 0 = pure BM25, 1 = pure vector

    # ── App ───────────────────────────────────────────────────
    app_env: str = "development"
    log_level: str = "DEBUG"
    cors_origins: List[str] = ["http://localhost:3000", "http://localhost:5173"]


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Cached singleton — import and call this everywhere."""
    return Settings()
