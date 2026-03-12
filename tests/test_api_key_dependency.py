from __future__ import annotations

import asyncio
import json

from starlette.requests import Request

from research_agent.api.dependencies import resolve_runtime_api_key
from research_agent.config import Settings


def _build_request(payload: dict) -> Request:
    body = json.dumps(payload).encode("utf-8")

    async def receive() -> dict:
        return {"type": "http.request", "body": body, "more_body": False}

    scope = {
        "type": "http",
        "http_version": "1.1",
        "method": "POST",
        "scheme": "http",
        "path": "/api/run",
        "raw_path": b"/api/run",
        "query_string": b"",
        "headers": [(b"content-type", b"application/json")],
        "client": ("127.0.0.1", 8000),
        "server": ("127.0.0.1", 8000),
    }
    return Request(scope, receive)


def test_header_api_key_takes_precedence() -> None:
    request = _build_request({"api_key": "body-key", "api_provider": "openai"})
    settings = Settings(default_provider="groq", groq_api_key="fallback-key")

    result = asyncio.run(
        resolve_runtime_api_key(
            request,
            header_api_key="header-key",
            header_provider="anthropic",
            settings=settings,
        )
    )

    assert result.source == "header"
    assert result.api_key == "header-key"
    assert result.provider == "anthropic"


def test_config_fallback_used_when_no_header_or_body_key() -> None:
    request = _build_request({"api_provider": "groq"})
    settings = Settings(default_provider="groq", groq_api_key="fallback-key")

    result = asyncio.run(
        resolve_runtime_api_key(
            request,
            header_api_key=None,
            header_provider=None,
            settings=settings,
        )
    )

    assert result.source == "config"
    assert result.api_key == "fallback-key"
    assert result.provider == "groq"
