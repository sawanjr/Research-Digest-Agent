from __future__ import annotations

import json
from pathlib import Path

import structlog
from langgraph.graph import END, StateGraph

from research_agent.config import RuntimeLLMConfig, Settings, get_settings
from research_agent.core.clustering import cluster_claims_semantically
from research_agent.core.digest import build_digest_markdown, build_sources_payload
from research_agent.core.embeddings import EmbeddingService
from research_agent.core.extraction import ClaimExtractionService
from research_agent.core.ingestion import ingest_sources
from research_agent.core.stance import StanceService
from research_agent.core.vector_store import OptionalChromaStore
from research_agent.models import ProcessingError, ResearchState, RuntimeConfig, UploadedSource


def _append_processing_error(
    state: ResearchState,
    component: str,
    reason: str,
) -> list[ProcessingError]:
    return [*state.processing_errors, ProcessingError(component=component, reason=reason)]


def build_research_graph(settings: Settings):
    logger = structlog.get_logger("research_agent")
    extraction_service = ClaimExtractionService(settings, logger)
    embedding_service = EmbeddingService(settings, logger)
    stance_service = StanceService(settings, logger)

    def as_state(raw_state: ResearchState | dict) -> ResearchState:
        if isinstance(raw_state, ResearchState):
            return raw_state
        return ResearchState.model_validate(raw_state)

    async def ingest_node(state: ResearchState | dict) -> dict:
        current = as_state(state)
        try:
            sources, errors = await ingest_sources(
                urls=current.urls,
                folder_path=current.folder_path,
                uploaded_sources=current.uploaded_sources,
                timeout_seconds=settings.request_timeout_seconds,
                logger=logger,
            )
            return {
                "sources": sources,
                "ingestion_errors": errors,
            }
        except Exception as exc:  # noqa: BLE001
            logger.exception("graph.ingest_failed", error=str(exc))
            return {
                "sources": [],
                "ingestion_errors": current.ingestion_errors,
                "processing_errors": _append_processing_error(current, "ingestion", str(exc)),
            }

    async def extract_node(state: ResearchState | dict) -> dict:
        current = as_state(state)
        try:
            runtime = RuntimeLLMConfig(
                provider=current.runtime.api_provider,
                api_key=current.runtime.api_key or settings.fallback_api_key(current.runtime.api_provider),
                model_name=current.runtime.model_name,
            )
            claims = await extraction_service.extract_claims(current.topic, current.sources, runtime)
            return {"claims": claims}
        except Exception as exc:  # noqa: BLE001
            logger.exception("graph.extract_failed", error=str(exc))
            return {
                "claims": [],
                "processing_errors": _append_processing_error(current, "claim_extraction", str(exc)),
            }

    async def embed_node(state: ResearchState | dict) -> dict:
        current = as_state(state)
        try:
            claims = await embedding_service.embed_claims(current.claims)
            return {"claims": claims}
        except Exception as exc:  # noqa: BLE001
            logger.exception("graph.embedding_failed", error=str(exc))
            return {
                "processing_errors": _append_processing_error(current, "embeddings", str(exc)),
            }

    async def cluster_node(state: ResearchState | dict) -> dict:
        current = as_state(state)
        threshold = current.runtime.grouping_distance_threshold or settings.clustering_distance_threshold
        try:
            grouped_claims = cluster_claims_semantically(current.claims, threshold, logger)
            return {"clustered_claim_groups": grouped_claims}
        except Exception as exc:  # noqa: BLE001
            logger.exception("graph.clustering_failed", error=str(exc))
            return {
                "clustered_claim_groups": [[claim] for claim in current.claims],
                "processing_errors": _append_processing_error(current, "clustering", str(exc)),
            }

    async def stance_node(state: ResearchState | dict) -> dict:
        current = as_state(state)
        try:
            runtime = RuntimeLLMConfig(
                provider=current.runtime.api_provider,
                api_key=current.runtime.api_key or settings.fallback_api_key(current.runtime.api_provider),
                model_name=current.runtime.model_name,
            )
            clusters = await stance_service.build_clusters(current.clustered_claim_groups, runtime=runtime)
            flattened_claims = [claim for cluster in clusters for claim in cluster.claims]
            return {
                "clusters": clusters,
                "claims": flattened_claims,
            }
        except Exception as exc:  # noqa: BLE001
            logger.exception("graph.stance_failed", error=str(exc))
            return {
                "clusters": [],
                "processing_errors": _append_processing_error(current, "stance", str(exc)),
            }

    async def vector_store_node(state: ResearchState | dict) -> dict:
        current = as_state(state)
        use_vector_store = current.runtime.use_vector_store or settings.use_vector_store_default
        if not use_vector_store or not current.clusters:
            return {}

        collection_name = current.runtime.vector_store_collection or settings.chroma_collection_name
        vector_store = OptionalChromaStore(collection_name=collection_name, logger=logger)

        stored = await vector_store.upsert_clusters(current.clusters)
        if not stored:
            return {
                "processing_errors": _append_processing_error(
                    current,
                    "vector_store",
                    "vector_store_disabled_or_upsert_failed",
                )
            }
        return {}

    async def compile_node(state: ResearchState | dict) -> dict:
        current = as_state(state)
        digest_markdown = build_digest_markdown(
            topic=current.topic,
            sources=current.sources,
            claims=current.claims,
            clusters=current.clusters,
            ingestion_errors=current.ingestion_errors,
            processing_errors=current.processing_errors,
        )
        payload = build_sources_payload(
            topic=current.topic,
            sources=current.sources,
            claims=current.claims,
            clusters=current.clusters,
            ingestion_errors=current.ingestion_errors,
            processing_errors=current.processing_errors,
        )
        return {
            "digest_markdown": digest_markdown,
            "sources_payload": payload,
        }

    async def persist_node(state: ResearchState | dict) -> dict:
        current = as_state(state)
        output_dir = Path(current.output_dir).resolve()
        output_dir.mkdir(parents=True, exist_ok=True)

        digest_path = output_dir / "digest.md"
        sources_path = output_dir / "sources.json"

        digest_path.write_text(current.digest_markdown, encoding="utf-8")
        sources_path.write_text(json.dumps(current.sources_payload, indent=2), encoding="utf-8")

        return {
            "digest_path": str(digest_path),
            "sources_json_path": str(sources_path),
        }

    workflow = StateGraph(ResearchState)
    workflow.add_node("ingest", ingest_node)
    workflow.add_node("extract", extract_node)
    workflow.add_node("embed", embed_node)
    workflow.add_node("cluster", cluster_node)
    workflow.add_node("stance", stance_node)
    workflow.add_node("vector_store", vector_store_node)
    workflow.add_node("compile", compile_node)
    workflow.add_node("persist", persist_node)

    workflow.set_entry_point("ingest")
    workflow.add_edge("ingest", "extract")
    workflow.add_edge("extract", "embed")
    workflow.add_edge("embed", "cluster")
    workflow.add_edge("cluster", "stance")
    workflow.add_edge("stance", "vector_store")
    workflow.add_edge("vector_store", "compile")
    workflow.add_edge("compile", "persist")
    workflow.add_edge("persist", END)

    return workflow.compile()


async def run_research_digest(
    *,
    topic: str,
    urls: list[str] | None,
    folder_path: str | None,
    output_dir: str,
    uploaded_sources: list[UploadedSource] | None = None,
    runtime: RuntimeConfig | None = None,
    settings: Settings | None = None,
) -> ResearchState:
    cleaned_urls = [url.strip() for url in (urls or []) if url.strip()]
    cleaned_uploads = [item for item in (uploaded_sources or []) if item.content.strip()]
    cleaned_folder = folder_path.strip() if folder_path else None

    if not cleaned_urls and not cleaned_folder and not cleaned_uploads:
        raise ValueError("Provide at least one URL, folder path, or uploaded file.")

    active_settings = settings or get_settings()
    graph = build_research_graph(active_settings)
    initial_state = ResearchState(
        topic=topic.strip(),
        urls=cleaned_urls,
        folder_path=cleaned_folder,
        uploaded_sources=cleaned_uploads,
        output_dir=output_dir.strip(),
        runtime=runtime or RuntimeConfig(),
    )

    final_state = await graph.ainvoke(initial_state.model_dump())
    return ResearchState.model_validate(final_state)
