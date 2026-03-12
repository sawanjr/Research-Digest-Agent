from __future__ import annotations

import hashlib
import re
from pathlib import Path
from typing import Callable
from urllib.parse import urlsplit, urlunsplit
from urllib.request import Request, urlopen

from app.models import IngestionError, SourceDocument, UploadedSource

try:
    from bs4 import BeautifulSoup
except Exception:  # pragma: no cover - optional dependency fallback
    BeautifulSoup = None


HtmlReader = Callable[[str], tuple[str, str]]

_MIN_SOURCE_LENGTH = 60
_WS_RE = re.compile(r"\s+")


def clean_text(text: str) -> str:
    return _WS_RE.sub(" ", text).strip()


def _simple_html_to_text(html: str) -> tuple[str, str]:
    title_match = re.search(r"<title[^>]*>(.*?)</title>", html, flags=re.IGNORECASE | re.DOTALL)
    title = clean_text(title_match.group(1)) if title_match else "Untitled HTML Source"
    stripped = re.sub(r"<script[\s\S]*?</script>", " ", html, flags=re.IGNORECASE)
    stripped = re.sub(r"<style[\s\S]*?</style>", " ", stripped, flags=re.IGNORECASE)
    stripped = re.sub(r"<[^>]+>", " ", stripped)
    return title, clean_text(stripped)


def parse_html(html: str) -> tuple[str, str]:
    if BeautifulSoup is None:
        return _simple_html_to_text(html)

    soup = BeautifulSoup(html, "html.parser")
    for tag in soup(["script", "style", "noscript"]):
        tag.decompose()
    title = clean_text(soup.title.get_text(" ")) if soup.title else "Untitled HTML Source"
    text = clean_text(soup.get_text(" "))
    return title, text


def canonicalize_url(raw_url: str) -> str:
    candidate = raw_url.strip()
    if "://" not in candidate:
        candidate = f"https://{candidate}"
    parts = urlsplit(candidate)
    normalized_path = parts.path.rstrip("/") or "/"
    return urlunsplit((parts.scheme.lower(), parts.netloc.lower(), normalized_path, parts.query, ""))


def read_url_default(url: str, timeout_seconds: int) -> tuple[str, str]:
    request = Request(
        url,
        headers={
            "User-Agent": "ResearchDigestAgent/1.0 (+https://example.local)",
            "Accept": "text/html,text/plain;q=0.9,*/*;q=0.5",
        },
    )
    with urlopen(request, timeout=timeout_seconds) as response:
        content_type = response.headers.get("Content-Type", "").lower()
        charset = response.headers.get_content_charset() or "utf-8"
        raw = response.read()
    decoded = raw.decode(charset, errors="ignore")
    if "html" in content_type or "<html" in decoded.lower():
        return parse_html(decoded)
    return url, clean_text(decoded)


def _content_hash(text: str) -> str:
    return hashlib.sha256(text.lower().encode("utf-8", errors="ignore")).hexdigest()


def ingest_sources(
    urls: list[str] | None,
    folder_path: str | None,
    uploaded_sources: list[UploadedSource] | None = None,
    timeout_seconds: int = 15,
    url_reader: HtmlReader | None = None,
) -> tuple[list[SourceDocument], list[IngestionError]]:
    sources: list[SourceDocument] = []
    errors: list[IngestionError] = []
    seen_references: set[str] = set()
    seen_hash_to_source_id: dict[str, str] = {}
    next_source_number = 1

    def register_source(source: str, source_type: str, title: str, content: str) -> None:
        nonlocal next_source_number

        normalized_content = clean_text(content)
        if len(normalized_content) < _MIN_SOURCE_LENGTH:
            errors.append(IngestionError(source=source, reason="empty_or_unclear_content"))
            return

        digest = _content_hash(normalized_content)
        if digest in seen_hash_to_source_id:
            errors.append(
                IngestionError(
                    source=source,
                    reason=f"duplicate_content_of_{seen_hash_to_source_id[digest]}",
                )
            )
            return

        source_id = f"S{next_source_number}"
        next_source_number += 1

        seen_hash_to_source_id[digest] = source_id
        sources.append(
            SourceDocument(
                source_id=source_id,
                source=source,
                source_type="url" if source_type == "url" else "file",
                title=clean_text(title) or "Untitled Source",
                content=normalized_content,
                length=len(normalized_content),
                content_hash=digest,
            )
        )

    reader = url_reader or (lambda u: read_url_default(u, timeout_seconds))

    for raw_url in urls or []:
        if not raw_url.strip():
            continue
        try:
            normalized_url = canonicalize_url(raw_url)
        except Exception as exc:
            errors.append(IngestionError(source=raw_url, reason=f"invalid_url:{exc}"))
            continue

        if normalized_url in seen_references:
            errors.append(IngestionError(source=normalized_url, reason="duplicate_source_reference"))
            continue
        seen_references.add(normalized_url)

        try:
            title, content = reader(normalized_url)
            register_source(source=normalized_url, source_type="url", title=title, content=content)
        except Exception as exc:
            errors.append(IngestionError(source=normalized_url, reason=f"fetch_failed:{exc}"))

    for uploaded in uploaded_sources or []:
        source_ref = f"uploaded::{uploaded.name.strip() or 'unnamed_file'}"
        if source_ref in seen_references:
            errors.append(IngestionError(source=source_ref, reason="duplicate_source_reference"))
            continue
        seen_references.add(source_ref)

        title = Path(uploaded.name).stem or "Uploaded Source"
        raw_text = uploaded.content
        if uploaded.name.lower().endswith((".html", ".htm")):
            title, raw_text = parse_html(uploaded.content)
        register_source(source=source_ref, source_type="file", title=title, content=raw_text)

    if folder_path:
        root = Path(folder_path)
        if not root.exists() or not root.is_dir():
            errors.append(IngestionError(source=folder_path, reason="invalid_folder_path"))
        else:
            files = sorted(
                [*root.rglob("*.txt"), *root.rglob("*.html"), *root.rglob("*.htm")],
                key=lambda p: str(p).lower(),
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
                    raw = file_path.read_text(encoding="utf-8", errors="ignore")
                except Exception as exc:
                    errors.append(IngestionError(source=source_ref, reason=f"read_failed:{exc}"))
                    continue

                if file_path.suffix.lower() in {".html", ".htm"}:
                    title, content = parse_html(raw)
                else:
                    title, content = file_path.stem, raw

                register_source(source=source_ref, source_type="file", title=title, content=content)

    return sources, errors
