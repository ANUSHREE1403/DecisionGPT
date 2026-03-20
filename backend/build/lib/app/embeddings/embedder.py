"""
app/embeddings/embedder.py
OpenAI embedding wrapper with batching + tenacity retry.
"""
from __future__ import annotations

from typing import List

import numpy as np
from loguru import logger
from openai import OpenAI
from tenacity import retry, stop_after_attempt, wait_exponential

from app.core.config import get_settings


class Embedder:
    def __init__(self) -> None:
        settings        = get_settings()
        self._client    = OpenAI(api_key=settings.openai_api_key)
        self._model     = settings.openai_embedding_model
        self._batch     = 256          # OpenAI allows up to 2048

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(min=1, max=10))
    def embed_batch(self, texts: List[str]) -> np.ndarray:
        """Embed a list of strings → float32 array shape (N, D)."""
        if not texts:
            return np.empty((0, 1536), dtype=np.float32)

        # clean inputs
        texts = [t.replace("\n", " ").strip() for t in texts]

        response = self._client.embeddings.create(model=self._model, input=texts)
        vecs     = np.array([d.embedding for d in response.data], dtype=np.float32)
        logger.debug("Embedded {} texts → shape {}", len(texts), vecs.shape)
        return vecs

    def embed_one(self, text: str) -> np.ndarray:
        """Embed a single string → 1-D float32 vector."""
        return self.embed_batch([text])[0]

    def embed_many(self, texts: List[str]) -> np.ndarray:
        """Embed large list with automatic batching."""
        all_vecs: List[np.ndarray] = []
        for i in range(0, len(texts), self._batch):
            batch = texts[i : i + self._batch]
            all_vecs.append(self.embed_batch(batch))
        return np.vstack(all_vecs) if all_vecs else np.empty((0,), dtype=np.float32)


# singleton
_embedder: Embedder | None = None


def get_embedder() -> Embedder:
    global _embedder
    if _embedder is None:
        _embedder = Embedder()
    return _embedder
