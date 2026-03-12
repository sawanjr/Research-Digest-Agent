from __future__ import annotations

from app.digest import build_digest_markdown
from app.grouping import group_claims
from app.models import ClaimRecord, SourceDocument


def test_conflicting_claims_are_preserved_in_digest() -> None:
    sources = [
        SourceDocument(
            source_id="S1",
            source="source_a.txt",
            source_type="file",
            title="Source A",
            content="Mandatory audits are necessary for high-risk AI.",
            length=52,
            content_hash="a",
        ),
        SourceDocument(
            source_id="S2",
            source="source_b.txt",
            source_type="file",
            title="Source B",
            content="Mandatory audits are unnecessary for high-risk AI.",
            length=54,
            content_hash="b",
        ),
    ]

    claims = [
        ClaimRecord(
            claim_id="C1",
            source_id="S1",
            source="source_a.txt",
            source_title="Source A",
            text="Mandatory pre-release audits are necessary for high-risk AI systems.",
            evidence="Mandatory audits are necessary for high-risk AI.",
            stance="supporting",
        ),
        ClaimRecord(
            claim_id="C2",
            source_id="S2",
            source="source_b.txt",
            source_title="Source B",
            text="Mandatory pre-release audits are unnecessary for high-risk AI systems.",
            evidence="Mandatory audits are unnecessary for high-risk AI.",
            stance="opposing",
        ),
    ]

    groups = group_claims(claims, threshold=0.5)
    assert len(groups) == 1
    assert groups[0].has_conflict is True

    digest = build_digest_markdown(
        topic="AI regulation",
        sources=sources,
        claims=claims,
        groups=groups,
        errors=[],
    )

    assert "Conflicting viewpoints: yes" in digest
    assert "[S1 | supporting]" in digest
    assert "[S2 | opposing]" in digest
