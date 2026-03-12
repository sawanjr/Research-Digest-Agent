from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field, model_validator

from research_agent.api.dependencies import ResolvedAPIKey, resolve_runtime_api_key
from research_agent.core.graph import run_research_digest
from research_agent.models import RuntimeConfig, UploadedSource


router = APIRouter()


class LocalFileInput(BaseModel):
    name: str = Field(..., min_length=1)
    content: str = ""


class RunRequest(BaseModel):
    topic: str | None = Field(default="General research topic")
    urls: list[str] = Field(default_factory=list)
    folder_path: str | None = None
    local_files: list[LocalFileInput] = Field(default_factory=list)
    output_dir: str = "outputs/latest"
    grouping_threshold: float | None = None

    api_provider: str | None = None
    api_key: str | None = None

    use_vector_store: bool = False
    vector_store_collection: str | None = None

    @model_validator(mode="after")
    def validate_sources(self) -> "RunRequest":
        cleaned_urls = [url.strip() for url in self.urls if url.strip()]
        cleaned_local_files = [file for file in self.local_files if file.content.strip()]
        has_folder = bool(self.folder_path and self.folder_path.strip())
        if not cleaned_urls and not has_folder and not cleaned_local_files:
            raise ValueError("Provide at least one URL, local folder, or uploaded source file.")

        self.urls = cleaned_urls
        self.folder_path = self.folder_path.strip() if self.folder_path else None
        self.local_files = cleaned_local_files
        topic_value = (self.topic or "").strip()
        self.topic = topic_value if topic_value else "General research topic"

        if self.api_key is not None:
            self.api_key = self.api_key.strip() or None
        if self.api_provider is not None:
            self.api_provider = self.api_provider.strip().lower() or None
        return self


@router.get("/api/health")
async def health() -> dict:
    return {"status": "ok"}


@router.post("/api/run")
async def run(
    payload: RunRequest,
    resolved_api_key: ResolvedAPIKey = Depends(resolve_runtime_api_key),
) -> dict:
    try:
        provider = resolved_api_key.provider
        api_key = resolved_api_key.api_key or payload.api_key

        runtime = RuntimeConfig(
            api_provider=provider,
            api_key=api_key,
            grouping_distance_threshold=payload.grouping_threshold,
            use_vector_store=payload.use_vector_store,
            vector_store_collection=payload.vector_store_collection,
        )

        uploaded_sources = [
            UploadedSource(name=file.name.strip(), content=file.content)
            for file in payload.local_files
        ]

        result = await run_research_digest(
            topic=payload.topic or "General research topic",
            urls=payload.urls,
            folder_path=payload.folder_path,
            uploaded_sources=uploaded_sources,
            output_dir=payload.output_dir,
            runtime=runtime,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=500, detail=f"Agent run failed: {exc}") from exc

    return {
        "topic": payload.topic,
        "sources_processed": len(result.sources),
        "claims_extracted": len(result.claims),
        "claim_groups": len(result.clusters),
        "skipped_sources": [error.model_dump() for error in result.ingestion_errors],
        "processing_errors": [error.model_dump() for error in result.processing_errors],
        "digest_path": result.digest_path,
        "sources_json_path": result.sources_json_path,
        "digest_markdown": result.digest_markdown,
        "sources_payload": result.sources_payload,
        "api_key_source": resolved_api_key.source,
    }
