from __future__ import annotations

import asyncio
from typing import Any

import structlog

from research_agent.config import Settings
from research_agent.core.stance import StanceService
from research_agent.models import ClaimRecord


class _MockClassifier:
    def __call__(self, candidate_claim: str, labels: list[str], hypothesis_template: str) -> dict[str, Any]:
        lowered = candidate_claim.lower()
        if "not necessary" in lowered:
            return {"labels": ["contradicts", "neutral", "supports"]}
        if "necessary" in lowered:
            return {"labels": ["supports", "neutral", "contradicts"]}
        return {"labels": ["neutral", "supports", "contradicts"]}


def test_confidence_uses_support_over_support_plus_contradict() -> None:
    logger = structlog.get_logger("test")
    service = StanceService(Settings(), logger)
    service._classifier = _MockClassifier()

    claims = [
        ClaimRecord(
            claim_id="C1",
            claim_text="Mandatory pre-release audits are necessary for high-risk AI systems.",
            source_url="source-a",
            source_id="S1",
        ),
        ClaimRecord(
            claim_id="C2",
            claim_text="Mandatory pre-release audits are not necessary for high-risk AI systems.",
            source_url="source-b",
            source_id="S2",
        ),
        ClaimRecord(
            claim_id="C3",
            claim_text="Regional procurement policies changed in 2024.",
            source_url="source-c",
            source_id="S3",
        ),
    ]

    clusters = asyncio.run(service.build_clusters([claims]))

    assert len(clusters) == 1
    assert clusters[0].confidence_score == 0.5

    supporting = set(clusters[0].supporting_sources)
    contradicting = set(clusters[0].contradicting_sources)
    assert supporting.isdisjoint(contradicting)
    assert supporting | contradicting == {"source-a", "source-b"}
