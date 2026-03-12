from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field

StanceLabel = Literal["supporting", "contradicting", "neutral"]


class UploadedSource(BaseModel):
    name: str = Field(min_length=1)
    content: str = ""


class IngestionError(BaseModel):
    source: str
    reason: str


class ProcessingError(BaseModel):
    component: str
    reason: str


class SourceDocument(BaseModel):
    source_id: str
    source_url: str
    source_type: Literal["url", "file", "uploaded"]
    title: str
    content: str
    length: int
    content_hash: str


class ClaimRecord(BaseModel):
    claim_id: str = ""
    claim_text: str
    source_url: str
    source_id: str
    source_title: str = ""
    evidence: str = ""
    stance: StanceLabel = "neutral"
    embedding: list[float] | None = None


class ClaimCluster(BaseModel):
    canonical_claim: str
    supporting_sources: list[str] = Field(default_factory=list)
    contradicting_sources: list[str] = Field(default_factory=list)
    confidence_score: float = 0.0
    claims: list[ClaimRecord] = Field(default_factory=list)


class RuntimeConfig(BaseModel):
    api_provider: Literal["openai", "groq", "anthropic"] = "groq"
    api_key: str | None = None
    model_name: str | None = None
    grouping_distance_threshold: float | None = None
    use_vector_store: bool = False
    vector_store_collection: str | None = None


class ResearchState(BaseModel):
    topic: str
    urls: list[str] = Field(default_factory=list)
    folder_path: str | None = None
    uploaded_sources: list[UploadedSource] = Field(default_factory=list)
    output_dir: str = "outputs/latest"
    runtime: RuntimeConfig = Field(default_factory=RuntimeConfig)

    sources: list[SourceDocument] = Field(default_factory=list)
    ingestion_errors: list[IngestionError] = Field(default_factory=list)
    processing_errors: list[ProcessingError] = Field(default_factory=list)

    claims: list[ClaimRecord] = Field(default_factory=list)
    clustered_claim_groups: list[list[ClaimRecord]] = Field(default_factory=list)
    clusters: list[ClaimCluster] = Field(default_factory=list)

    digest_markdown: str = ""
    sources_payload: dict[str, Any] = Field(default_factory=dict)
    digest_path: str = ""
    sources_json_path: str = ""
