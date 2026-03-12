from __future__ import annotations

from typing import Literal

from fastapi import Depends, Request, Security
from fastapi.security import APIKeyHeader
from pydantic import BaseModel

from research_agent.config import ProviderName, Settings, get_settings


api_key_header = APIKeyHeader(name="X-API-KEY", auto_error=False)
api_provider_header = APIKeyHeader(name="X-API-PROVIDER", auto_error=False)


class ResolvedAPIKey(BaseModel):
    provider: ProviderName
    api_key: str | None
    source: Literal["header", "body", "config", "none"]


def _normalize_provider(raw_provider: str | None, fallback: ProviderName) -> ProviderName:
    candidate = (raw_provider or "").strip().lower()
    if candidate in {"openai", "groq", "anthropic"}:
        return candidate  # type: ignore[return-value]
    return fallback


def get_settings_dependency() -> Settings:
    return get_settings()


async def resolve_runtime_api_key(
    request: Request,
    header_api_key: str | None = Security(api_key_header),
    header_provider: str | None = Security(api_provider_header),
    settings: Settings = Depends(get_settings_dependency),
) -> ResolvedAPIKey:
    body_api_key: str | None = None
    body_provider: str | None = None

    content_type = request.headers.get("content-type", "")
    if "application/json" in content_type:
        try:
            body = await request.json()
            if isinstance(body, dict):
                if isinstance(body.get("api_key"), str):
                    body_api_key = body["api_key"].strip() or None
                if isinstance(body.get("api_provider"), str):
                    body_provider = body["api_provider"]
        except Exception:
            body_api_key = None
            body_provider = None

    provider = _normalize_provider(header_provider or body_provider, settings.default_provider)

    if header_api_key and header_api_key.strip():
        return ResolvedAPIKey(provider=provider, api_key=header_api_key.strip(), source="header")

    if body_api_key and body_api_key.strip():
        return ResolvedAPIKey(provider=provider, api_key=body_api_key.strip(), source="body")

    fallback_key = settings.fallback_api_key(provider)
    if fallback_key:
        return ResolvedAPIKey(provider=provider, api_key=fallback_key, source="config")

    return ResolvedAPIKey(provider=provider, api_key=None, source="none")
