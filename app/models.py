from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal, TypedDict

from pydantic import BaseModel, Field


Stance = Literal["supporting", "opposing", "neutral"]


@dataclass
class SourceDocument:
    source_id: str
    source: str
    source_type: Literal["url", "file"]
    title: str
    content: str
    length: int
    content_hash: str


@dataclass
class IngestionError:
    source: str
    reason: str


class ExtractedClaim(BaseModel):
    claim: str
    evidence: str
    stance: Stance = "neutral"


class ExtractedClaimBatch(BaseModel):
    claims: list[ExtractedClaim] = Field(default_factory=list)


@dataclass
class ClaimRecord:
    claim_id: str
    source_id: str
    source: str
    source_title: str
    text: str
    evidence: str
    stance: Stance


@dataclass
class ClaimGroup:
    group_id: str
    theme_title: str
    summary: str
    claims: list[ClaimRecord] = field(default_factory=list)
    source_ids: list[str] = field(default_factory=list)
    has_conflict: bool = False


@dataclass
class UploadedSource:
    name: str
    content: str


class AgentState(TypedDict):
    topic: str
    urls: list[str]
    folder_path: str | None
    uploaded_sources: list[UploadedSource]
    output_dir: str
    grouping_threshold: float
    sources: list[SourceDocument]
    ingestion_errors: list[IngestionError]
    claims: list[ClaimRecord]
    groups: list[ClaimGroup]
    digest_markdown: str
    sources_payload: dict
    digest_path: str
    sources_json_path: str
