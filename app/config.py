from __future__ import annotations

import os
from dataclasses import dataclass


@dataclass(frozen=True)
class AgentConfig:
    groq_api_key: str | None
    model_name: str
    max_chars_per_source: int
    max_claims_per_source: int
    grouping_threshold: float
    request_timeout_seconds: int

    @staticmethod
    def from_env() -> "AgentConfig":
        return AgentConfig(
            groq_api_key=os.getenv("GROQ_API_KEY"),
            model_name=os.getenv("MODEL_NAME", "openai/gpt-oss-120b"),
            max_chars_per_source=int(os.getenv("MAX_CHARS_PER_SOURCE", "12000")),
            max_claims_per_source=int(os.getenv("MAX_CLAIMS_PER_SOURCE", "6")),
            grouping_threshold=float(os.getenv("GROUPING_THRESHOLD", "0.68")),
            request_timeout_seconds=int(os.getenv("REQUEST_TIMEOUT_SECONDS", "15")),
        )
