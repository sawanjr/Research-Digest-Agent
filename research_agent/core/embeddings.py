from __future__ import annotations

import asyncio
from typing import Any, Sequence

import structlog

from research_agent.config import Settings
from research_agent.models import ClaimRecord

try:
    from sentence_transformers import SentenceTransformer
except Exception:  # pragma: no cover - optional dependency
    SentenceTransformer = None

try:
    import torch
except Exception:  # pragma: no cover - optional dependency
    torch = None


class EmbeddingService:
    def __init__(self, settings: Settings, logger: structlog.stdlib.BoundLogger):
        self.settings = settings
        self.logger = logger
        self._model: Any = None

    async def _ensure_model(self) -> Any:
        if self._model is not None:
            return self._model

        sentence_transformer_cls = SentenceTransformer
        if sentence_transformer_cls is None:
            self.logger.warning("embeddings.unavailable", reason="sentence_transformers_not_installed")
            return None

        device = "cpu"
        if torch is not None and torch.cuda.is_available():
            device = "cuda"
        elif torch is not None and hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            device = "mps"

        def _load() -> Any:
            return sentence_transformer_cls(self.settings.embedding_model_name, device=device)

        try:
            self._model = await asyncio.to_thread(_load)
            self.logger.info(
                "embeddings.model_ready",
                model=self.settings.embedding_model_name,
                device=device,
            )
            return self._model
        except Exception as exc:  # noqa: BLE001
            self.logger.warning("embeddings.load_failed", error=str(exc))
            return None

    async def embed_texts(self, texts: Sequence[str]) -> list[list[float] | None]:
        model = await self._ensure_model()
        if model is None:
            return [None for _ in texts]

        batch_size = self.settings.embedding_batch_size
        vectors: list[list[float] | None] = [None for _ in texts]

        for start in range(0, len(texts), batch_size):
            batch = list(texts[start : start + batch_size])
            try:
                batch_vectors = await asyncio.to_thread(
                    model.encode,
                    batch,
                    convert_to_numpy=True,
                    normalize_embeddings=True,
                    show_progress_bar=False,
                )
            except Exception as exc:  # noqa: BLE001
                self.logger.warning("embeddings.batch_failed", start=start, error=str(exc))
                continue

            for offset, vector in enumerate(batch_vectors):
                vectors[start + offset] = [float(value) for value in vector.tolist()]

        return vectors

    async def embed_claims(self, claims: list[ClaimRecord]) -> list[ClaimRecord]:
        texts = [claim.claim_text for claim in claims]
        vectors = await self.embed_texts(texts)

        embedded_claims: list[ClaimRecord] = []
        for claim, vector in zip(claims, vectors, strict=False):
            embedded_claims.append(claim.model_copy(update={"embedding": vector}))

        embedded_count = sum(1 for item in embedded_claims if item.embedding is not None)
        self.logger.info("embeddings.completed", claims=len(claims), embedded=embedded_count)
        return embedded_claims
