from __future__ import annotations

import asyncio
from typing import Any

from research_agent.core.graph import run_research_digest as run_research_digest_async
from research_agent.models import RuntimeConfig, UploadedSource


def run_research_digest(
    topic: str,
    urls: list[str] | None,
    folder_path: str | None,
    output_dir: str,
    uploaded_sources: list[Any] | None = None,
    grouping_threshold: float | None = None,
    config: Any | None = None,
):
    runtime = RuntimeConfig(
        grouping_distance_threshold=grouping_threshold,
    )

    normalized_uploads: list[UploadedSource] = []
    for item in uploaded_sources or []:
        name = getattr(item, "name", "uploaded.txt")
        content = getattr(item, "content", "")
        normalized_uploads.append(UploadedSource(name=name, content=content))

    result = asyncio.run(
        run_research_digest_async(
            topic=topic,
            urls=urls,
            folder_path=folder_path,
            output_dir=output_dir,
            uploaded_sources=normalized_uploads,
            runtime=runtime,
        )
    )

    return {
        "topic": result.topic,
        "urls": result.urls,
        "folder_path": result.folder_path,
        "uploaded_sources": result.uploaded_sources,
        "output_dir": result.output_dir,
        "grouping_threshold": result.runtime.grouping_distance_threshold,
        "sources": result.sources,
        "ingestion_errors": result.ingestion_errors,
        "claims": result.claims,
        "groups": result.clusters,
        "digest_markdown": result.digest_markdown,
        "sources_payload": result.sources_payload,
        "digest_path": result.digest_path,
        "sources_json_path": result.sources_json_path,
    }
