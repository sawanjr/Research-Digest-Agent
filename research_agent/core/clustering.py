from __future__ import annotations

from collections import defaultdict

import structlog

from research_agent.models import ClaimRecord

try:
    import numpy as np
except Exception:  # pragma: no cover - optional dependency
    np = None

try:
    from sklearn.cluster import AgglomerativeClustering
except Exception:  # pragma: no cover - optional dependency
    AgglomerativeClustering = None


def _cosine_similarity(left: list[float], right: list[float]) -> float:
    if np is None:
        return 0.0
    left_vector = np.asarray(left)
    right_vector = np.asarray(right)
    denominator = np.linalg.norm(left_vector) * np.linalg.norm(right_vector)
    if denominator == 0:
        return 0.0
    return float(np.dot(left_vector, right_vector) / denominator)


def select_canonical_claim(claims: list[ClaimRecord]) -> str:
    candidates = [claim for claim in claims if claim.embedding is not None]
    if not candidates:
        longest = max(claims, key=lambda claim: len(claim.claim_text), default=None)
        return longest.claim_text if longest is not None else ""

    if len(candidates) == 1:
        return candidates[0].claim_text

    centroids: list[float] | None = None
    if np is not None:
        matrix = np.asarray([claim.embedding for claim in candidates], dtype=float)
        centroids = matrix.mean(axis=0).tolist()

    if centroids is None:
        return candidates[0].claim_text

    best_claim = candidates[0]
    best_score = -1.0
    for claim in candidates:
        score = _cosine_similarity(claim.embedding or [], centroids)
        if score > best_score:
            best_score = score
            best_claim = claim
    return best_claim.claim_text


def cluster_claims_semantically(
    claims: list[ClaimRecord],
    distance_threshold: float,
    logger: structlog.stdlib.BoundLogger,
) -> list[list[ClaimRecord]]:
    if not claims:
        return []

    embedded_claims = [(index, claim) for index, claim in enumerate(claims) if claim.embedding is not None]
    unembedded_claims = [claim for claim in claims if claim.embedding is None]

    if not embedded_claims or AgglomerativeClustering is None or np is None:
        logger.info(
            "clustering.fallback_singletons",
            reason="missing_embeddings_or_dependencies",
            claims=len(claims),
        )
        return [[claim] for claim in claims]

    embeddings = np.asarray([claim.embedding for _, claim in embedded_claims], dtype=float)
    if len(embeddings) == 1:
        clusters = [[embedded_claims[0][1]]]
    else:
        model = AgglomerativeClustering(
            n_clusters=None,
            metric="cosine",
            linkage="average",
            distance_threshold=distance_threshold,
        )
        labels = model.fit_predict(embeddings)
        grouped: dict[int, list[ClaimRecord]] = defaultdict(list)
        for (_, claim), label in zip(embedded_claims, labels, strict=False):
            grouped[int(label)].append(claim)
        clusters = list(grouped.values())

    for claim in unembedded_claims:
        clusters.append([claim])

    clusters.sort(key=len, reverse=True)
    logger.info("clustering.completed", clusters=len(clusters), claims=len(claims))
    return clusters
