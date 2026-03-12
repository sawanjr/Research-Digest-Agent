from __future__ import annotations

import logging
import os
import sys
from functools import lru_cache
from typing import Literal

import structlog
from pydantic import BaseModel, Field


ProviderName = Literal["openai", "groq", "anthropic"]


class Settings(BaseModel):
    request_timeout_seconds: int = Field(default=15, ge=1, le=120)
    max_chars_per_source: int = Field(default=12000, ge=512)
    max_claims_per_source: int = Field(default=6, ge=1, le=20)
    embedding_batch_size: int = Field(default=32, ge=1, le=512)
    embedding_model_name: str = "sentence-transformers/all-MiniLM-L6-v2"
    nli_model_name: str = "facebook/bart-large-mnli"
    clustering_distance_threshold: float = Field(default=0.4, gt=0.0, le=1.0)
    default_provider: ProviderName = "groq"
    default_model_name: str = "openai/gpt-oss-120b"
    openai_api_key: str | None = None
    groq_api_key: str | None = None
    anthropic_api_key: str | None = None
    use_vector_store_default: bool = False
    chroma_collection_name: str = "research_claim_clusters"

    @classmethod
    def from_env(cls) -> "Settings":
        default_provider = os.getenv("DEFAULT_LLM_PROVIDER", "groq").strip().lower() or "groq"
        if default_provider not in {"openai", "groq", "anthropic"}:
            default_provider = "groq"

        return cls(
            request_timeout_seconds=int(os.getenv("REQUEST_TIMEOUT_SECONDS", "15")),
            max_chars_per_source=int(os.getenv("MAX_CHARS_PER_SOURCE", "12000")),
            max_claims_per_source=int(os.getenv("MAX_CLAIMS_PER_SOURCE", "6")),
            embedding_batch_size=int(os.getenv("EMBEDDING_BATCH_SIZE", "32")),
            embedding_model_name=os.getenv(
                "EMBEDDING_MODEL_NAME",
                "sentence-transformers/all-MiniLM-L6-v2",
            ),
            nli_model_name=os.getenv("NLI_MODEL_NAME", "facebook/bart-large-mnli"),
            clustering_distance_threshold=float(os.getenv("CLUSTERING_DISTANCE_THRESHOLD", "0.4")),
            default_provider=default_provider,
            default_model_name=os.getenv("DEFAULT_MODEL_NAME", "openai/gpt-oss-120b"),
            openai_api_key=os.getenv("OPENAI_API_KEY"),
            groq_api_key=os.getenv("GROQ_API_KEY"),
            anthropic_api_key=os.getenv("ANTHROPIC_API_KEY"),
            use_vector_store_default=os.getenv("USE_VECTOR_STORE", "false").strip().lower() in {"1", "true", "yes"},
            chroma_collection_name=os.getenv("CHROMA_COLLECTION_NAME", "research_claim_clusters"),
        )

    def fallback_api_key(self, provider: ProviderName) -> str | None:
        if provider == "openai":
            return self.openai_api_key
        if provider == "anthropic":
            return self.anthropic_api_key
        return self.groq_api_key


class RuntimeLLMConfig(BaseModel):
    provider: ProviderName = "groq"
    api_key: str | None = None
    model_name: str | None = None


def configure_logging() -> None:
    timestamper = structlog.processors.TimeStamper(fmt="iso", utc=True)
    pre_chain = [
        structlog.contextvars.merge_contextvars,
        structlog.processors.add_log_level,
        timestamper,
    ]

    logging.basicConfig(
        format="%(message)s",
        stream=sys.stdout,
        level=logging.INFO,
    )

    structlog.configure(
        processors=[
            *pre_chain,
            structlog.processors.format_exc_info,
            structlog.processors.JSONRenderer(),
        ],
        logger_factory=structlog.stdlib.LoggerFactory(),
        wrapper_class=structlog.stdlib.BoundLogger,
        cache_logger_on_first_use=True,
    )


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings.from_env()
