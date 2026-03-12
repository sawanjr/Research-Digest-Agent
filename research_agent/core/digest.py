from __future__ import annotations

from datetime import datetime, timezone

from research_agent.models import ClaimCluster, ClaimRecord, IngestionError, ProcessingError, SourceDocument


def build_sources_payload(
    topic: str,
    sources: list[SourceDocument],
    claims: list[ClaimRecord],
    clusters: list[ClaimCluster],
    ingestion_errors: list[IngestionError],
    processing_errors: list[ProcessingError],
) -> dict:
    claims_by_source: dict[str, list[dict]] = {}
    for claim in claims:
        claims_by_source.setdefault(claim.source_id, []).append(
            {
                "claim_id": claim.claim_id,
                "claim_text": claim.claim_text,
                "evidence": claim.evidence,
                "stance": claim.stance,
            }
        )

    cluster_payload = []
    for cluster in clusters:
        cluster_payload.append(
            {
                "canonical_claim": cluster.canonical_claim,
                "supporting_sources": cluster.supporting_sources,
                "contradicting_sources": cluster.contradicting_sources,
                "confidence_score": cluster.confidence_score,
                "claims": [
                    {
                        "claim_id": claim.claim_id,
                        "claim_text": claim.claim_text,
                        "source_id": claim.source_id,
                        "source_url": claim.source_url,
                        "stance": claim.stance,
                    }
                    for claim in cluster.claims
                ],
            }
        )

    return {
        "topic": topic,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "sources": [
            {
                "source_id": source.source_id,
                "source_url": source.source_url,
                "source_type": source.source_type,
                "title": source.title,
                "length": source.length,
                "claims": claims_by_source.get(source.source_id, []),
            }
            for source in sources
        ],
        "clusters": cluster_payload,
        "ingestion_errors": [error.model_dump() for error in ingestion_errors],
        "processing_errors": [error.model_dump() for error in processing_errors],
    }


def build_digest_markdown(
    topic: str,
    sources: list[SourceDocument],
    claims: list[ClaimRecord],
    clusters: list[ClaimCluster],
    ingestion_errors: list[IngestionError],
    processing_errors: list[ProcessingError],
) -> str:
    lines = [
        f"# Semantic Research Digest: {topic}",
        "",
        "## Run Summary",
        f"- Sources processed: {len(sources)}",
        f"- Claims extracted: {len(claims)}",
        f"- Semantic clusters: {len(clusters)}",
        f"- Ingestion issues: {len(ingestion_errors)}",
        f"- Processing issues: {len(processing_errors)}",
        "",
        "## Source Index",
    ]

    if sources:
        for source in sources:
            lines.append(f"- [{source.source_id}] {source.title} ({source.source_url})")
    else:
        lines.append("- No valid sources were processed.")

    if ingestion_errors:
        lines.extend(["", "## Ingestion Issues"])
        for error in ingestion_errors:
            lines.append(f"- {error.source}: {error.reason}")

    if processing_errors:
        lines.extend(["", "## Processing Issues"])
        for error in processing_errors:
            lines.append(f"- {error.component}: {error.reason}")

    lines.extend(["", "## Clustered Claims"])
    if not clusters:
        lines.append("- No clusters available. The pipeline could not derive grounded claims.")
        return "\n".join(lines)

    for index, cluster in enumerate(clusters, start=1):
        confidence_pct = round(cluster.confidence_score * 100, 2)
        lines.extend(
            [
                "",
                f"### Cluster {index}",
                f"- Canonical claim: {cluster.canonical_claim}",
                f"- Confidence score: {confidence_pct}%",
                f"- Supporting sources: {', '.join(cluster.supporting_sources) or 'none'}",
                f"- Contradicting sources: {', '.join(cluster.contradicting_sources) or 'none'}",
                "- Evidence claims:",
            ]
        )
        for claim in cluster.claims:
            snippet = claim.evidence.replace("\n", " ").strip() or claim.claim_text
            lines.append(f"  - [{claim.source_id} | {claim.stance}] {claim.claim_text}")
            if snippet and snippet != claim.claim_text:
                lines.append(f"    - \"{snippet}\"")

    return "\n".join(lines)
