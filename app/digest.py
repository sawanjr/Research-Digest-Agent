from __future__ import annotations

from datetime import datetime, timezone

from app.models import ClaimGroup, ClaimRecord, IngestionError, SourceDocument


def build_sources_payload(
    topic: str,
    sources: list[SourceDocument],
    claims: list[ClaimRecord],
    groups: list[ClaimGroup],
    errors: list[IngestionError],
) -> dict:
    claims_by_source: dict[str, list[dict]] = {}
    for claim in claims:
        claims_by_source.setdefault(claim.source_id, []).append(
            {
                "claim_id": claim.claim_id,
                "claim": claim.text,
                "evidence": claim.evidence,
                "stance": claim.stance,
            }
        )

    source_payload = []
    for source in sources:
        source_payload.append(
            {
                "source_id": source.source_id,
                "source": source.source,
                "source_type": source.source_type,
                "title": source.title,
                "length": source.length,
                "claims": claims_by_source.get(source.source_id, []),
            }
        )

    group_payload = []
    for group in groups:
        group_payload.append(
            {
                "group_id": group.group_id,
                "theme_title": group.theme_title,
                "summary": group.summary,
                "source_ids": group.source_ids,
                "has_conflict": group.has_conflict,
                "claims": [
                    {
                        "claim_id": claim.claim_id,
                        "source_id": claim.source_id,
                        "claim": claim.text,
                        "evidence": claim.evidence,
                        "stance": claim.stance,
                    }
                    for claim in group.claims
                ],
            }
        )

    error_payload = [{"source": error.source, "reason": error.reason} for error in errors]

    return {
        "topic": topic,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "sources": source_payload,
        "groups": group_payload,
        "errors": error_payload,
    }


def build_digest_markdown(
    topic: str,
    sources: list[SourceDocument],
    claims: list[ClaimRecord],
    groups: list[ClaimGroup],
    errors: list[IngestionError],
) -> str:
    lines = [
        f"# Research Digest: {topic}",
        "",
        "## Run Summary",
        f"- Sources processed: {len(sources)}",
        f"- Claims extracted: {len(claims)}",
        f"- Claim groups: {len(groups)}",
        f"- Skipped or failed sources: {len(errors)}",
        "",
        "## Source Index",
    ]

    if sources:
        for source in sources:
            lines.append(f"- [{source.source_id}] {source.title} ({source.source})")
    else:
        lines.append("- No valid sources were processed.")

    if errors:
        lines.extend(["", "## Skipped Sources"])
        for error in errors:
            lines.append(f"- {error.source}: {error.reason}")

    lines.extend(["", "## Thematic Digest"])
    if not groups:
        lines.append("- No grounded claims were extracted from the provided sources.")
        return "\n".join(lines)

    for index, group in enumerate(groups, start=1):
        lines.extend(
            [
                "",
                f"### Theme {index}: {group.theme_title}",
                f"- Core insight: {group.summary}",
                f"- Supporting sources: {', '.join(f'[{sid}]' for sid in group.source_ids)}",
                f"- Conflicting viewpoints: {'yes' if group.has_conflict else 'no'}",
                "- Evidence:",
            ]
        )
        for claim in group.claims:
            claim_text = claim.text.replace("\n", " ").strip()
            evidence = claim.evidence.replace("\n", " ").strip()
            lines.append(f"  - [{claim.source_id} | {claim.stance}] {claim_text}")
            lines.append(f"    - \"{evidence}\"")

    return "\n".join(lines)
