from __future__ import annotations

import asyncio
import hashlib
import re
from pathlib import Path
from typing import Literal, Sequence
from urllib.parse import urlsplit, urlunsplit

import httpx
import structlog

from research_agent.models import IngestionError, SourceDocument, UploadedSource

try:
    import trafilatura  # type: ignore[import-not-found]
except Exception:  # pragma: no cover - optional dependency
    trafilatura = None

try:
    from bs4 import BeautifulSoup
except Exception:  # pragma: no cover - optional dependency
    BeautifulSoup = None


_WS_RE = re.compile(r"\s+")
_MIN_SOURCE_LENGTH = 60


def _clean_text(text: str) -> str:
    return _WS_RE.sub(" ", text).strip()


def _sha256(content: str) -> str:
    return hashlib.sha256(content.encode("utf-8", errors="ignore")).hexdigest()


def canonicalize_url(raw_url: str) -> str:
    candidate = raw_url.strip()
    if "://" not in candidate:
        candidate = f"https://{candidate}"
    split = urlsplit(candidate)
    normalized_path = split.path.rstrip("/") or "/"
    return urlunsplit((split.scheme.lower(), split.netloc.lower(), normalized_path, split.query, ""))


def _extract_html_with_bs4(raw_html: str, fallback_title: str) -> tuple[str, str]:
    if BeautifulSoup is None:
        text = re.sub(r"<[^>]+>", " ", raw_html)
        return fallback_title, _clean_text(text)

    soup = BeautifulSoup(raw_html, "html.parser")
    for tag in soup(["script", "style", "noscript"]):
        tag.decompose()
    title = fallback_title
    if soup.title and soup.title.get_text(strip=True):
        title = _clean_text(soup.title.get_text(" "))
    text = _clean_text(soup.get_text(" "))
    return title, text


def _extract_with_trafilatura(raw_html: str, fallback_title: str) -> tuple[str, str]:
    if trafilatura is None:
        return _extract_html_with_bs4(raw_html, fallback_title)

    title = fallback_title
    try:
        metadata = trafilatura.extract_metadata(raw_html)
        if metadata is not None and getattr(metadata, "title", None):
            title = _clean_text(str(metadata.title))
    except Exception:
        pass

    try:
        extracted_text = trafilatura.extract(
            raw_html,
            include_comments=False,
            include_tables=False,
            no_fallback=False,
            favor_precision=True,
        )
    except Exception:
        extracted_text = None

    if not extracted_text:
        return _extract_html_with_bs4(raw_html, title)

    return title, _clean_text(extracted_text)


async def _fetch_url(
    client: httpx.AsyncClient,
    url: str,
    logger: structlog.stdlib.BoundLogger,
) -> tuple[str, str] | IngestionError:
    try:
        response = await client.get(url)
        response.raise_for_status()
    except Exception as exc:  # noqa: BLE001
        logger.warning("ingestion.url_fetch_failed", source=url, error=str(exc))
        return IngestionError(source=url, reason=f"fetch_failed:{exc}")

    content_type = response.headers.get("content-type", "").lower()
    body_text = response.text
    fallback_title = url
    if "html" in content_type or "<html" in body_text.lower():
        title, text = _extract_with_trafilatura(body_text, fallback_title)
    else:
        title, text = fallback_title, _clean_text(body_text)
    return title, text


async def ingest_sources(
    urls: Sequence[str] | None,
    folder_path: str | None,
    uploaded_sources: Sequence[UploadedSource] | None,
    timeout_seconds: int,
    logger: structlog.stdlib.BoundLogger,
) -> tuple[list[SourceDocument], list[IngestionError]]:
    sources: list[SourceDocument] = []
    errors: list[IngestionError] = []
    seen_references: set[str] = set()
    seen_hash_to_source_id: dict[str, str] = {}
    next_source_number = 1

    def register_source(
        source_ref: str,
        source_type: Literal["url", "file", "uploaded"],
        title: str,
        content: str,
    ) -> None:
        nonlocal next_source_number

        normalized_content = _clean_text(content)
        if len(normalized_content) < _MIN_SOURCE_LENGTH:
            errors.append(IngestionError(source=source_ref, reason="empty_or_unclear_content"))
            logger.info("ingestion.source_skipped_empty", source=source_ref)
            return

        content_hash = _sha256(normalized_content.lower())
        if content_hash in seen_hash_to_source_id:
            duplicate_of = seen_hash_to_source_id[content_hash]
            errors.append(IngestionError(source=source_ref, reason=f"duplicate_content_of_{duplicate_of}"))
            logger.info(
                "ingestion.source_skipped_duplicate",
                source=source_ref,
                duplicate_of=duplicate_of,
            )
            return

        source_id = f"S{next_source_number}"
        next_source_number += 1
        seen_hash_to_source_id[content_hash] = source_id

        sources.append(
            SourceDocument(
                source_id=source_id,
                source_url=source_ref,
                source_type=source_type,
                title=_clean_text(title) or "Untitled Source",
                content=normalized_content,
                length=len(normalized_content),
                content_hash=content_hash,
            )
        )

    cleaned_urls: list[str] = []
    for raw_url in urls or []:
        if not raw_url.strip():
            continue
        try:
            normalized = canonicalize_url(raw_url)
        except Exception as exc:  # noqa: BLE001
            errors.append(IngestionError(source=raw_url, reason=f"invalid_url:{exc}"))
            logger.warning("ingestion.invalid_url", source=raw_url, error=str(exc))
            continue

        if normalized in seen_references:
            errors.append(IngestionError(source=normalized, reason="duplicate_source_reference"))
            logger.info("ingestion.duplicate_reference", source=normalized)
            continue

        seen_references.add(normalized)
        cleaned_urls.append(normalized)

    if cleaned_urls:
        headers = {
            "User-Agent": "ResearchDigestAgent/2.0",
            "Accept": "text/html,text/plain;q=0.9,*/*;q=0.5",
        }
        timeout = httpx.Timeout(timeout_seconds)
        async with httpx.AsyncClient(timeout=timeout, headers=headers, follow_redirects=True) as client:
            fetches = [_fetch_url(client, url, logger) for url in cleaned_urls]
            results = await asyncio.gather(*fetches, return_exceptions=False)

        for source_ref, result in zip(cleaned_urls, results, strict=False):
            if isinstance(result, IngestionError):
                errors.append(result)
                continue
            title, content = result
            register_source(source_ref=source_ref, source_type="url", title=title, content=content)

    for uploaded in uploaded_sources or []:
        source_ref = f"uploaded::{uploaded.name.strip() or 'unnamed_file'}"
        if source_ref in seen_references:
            errors.append(IngestionError(source=source_ref, reason="duplicate_source_reference"))
            continue
        seen_references.add(source_ref)

        title = Path(uploaded.name).stem or "Uploaded Source"
        text = uploaded.content
        if uploaded.name.lower().endswith((".html", ".htm")):
            title, text = _extract_with_trafilatura(uploaded.content, title)
        register_source(source_ref=source_ref, source_type="uploaded", title=title, content=text)

    if folder_path:
        root = Path(folder_path)
        if not root.exists() or not root.is_dir():
            errors.append(IngestionError(source=folder_path, reason="invalid_folder_path"))
        else:
            files = sorted(
                [*root.rglob("*.txt"), *root.rglob("*.html"), *root.rglob("*.htm")],
                key=lambda file_path: str(file_path).lower(),
            )
            if not files:
                errors.append(IngestionError(source=folder_path, reason="no_supported_files_found"))

            for file_path in files:
                source_ref = str(file_path.resolve())
                if source_ref in seen_references:
                    errors.append(IngestionError(source=source_ref, reason="duplicate_source_reference"))
                    continue
                seen_references.add(source_ref)

                try:
                    raw_text = await asyncio.to_thread(
                        file_path.read_text,
                        encoding="utf-8",
                        errors="ignore",
                    )
                except Exception as exc:  # noqa: BLE001
                    errors.append(IngestionError(source=source_ref, reason=f"read_failed:{exc}"))
                    logger.warning("ingestion.file_read_failed", source=source_ref, error=str(exc))
                    continue

                title = file_path.stem
                text = raw_text
                if file_path.suffix.lower() in {".html", ".htm"}:
                    title, text = _extract_with_trafilatura(raw_text, title)
                register_source(source_ref=source_ref, source_type="file", title=title, content=text)

    return sources, errors
