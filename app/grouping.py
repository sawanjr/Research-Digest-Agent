from __future__ import annotations

import re
from collections import Counter
from difflib import SequenceMatcher

from app.models import ClaimGroup, ClaimRecord


_TOKEN_RE = re.compile(r"[a-z0-9]+")
_STOP_WORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "by",
    "for",
    "from",
    "has",
    "have",
    "in",
    "is",
    "it",
    "of",
    "on",
    "or",
    "that",
    "the",
    "to",
    "was",
    "were",
    "with",
}


def _tokenize(text: str) -> list[str]:
    return [token for token in _TOKEN_RE.findall(text.lower()) if token not in _STOP_WORDS]


def _similarity(left: str, right: str) -> float:
    left_normalized = " ".join(_tokenize(left))
    right_normalized = " ".join(_tokenize(right))

    sequence_score = SequenceMatcher(None, left_normalized, right_normalized).ratio()
    left_tokens = set(left_normalized.split())
    right_tokens = set(right_normalized.split())

    if not left_tokens and not right_tokens:
        jaccard_score = 1.0
    elif not left_tokens or not right_tokens:
        jaccard_score = 0.0
    else:
        jaccard_score = len(left_tokens & right_tokens) / len(left_tokens | right_tokens)

    return (0.55 * sequence_score) + (0.45 * jaccard_score)


def _group_title(claims: list[ClaimRecord]) -> str:
    tokens = Counter()
    for claim in claims:
        tokens.update(_tokenize(claim.text))
    top = [token for token, _ in tokens.most_common(4)]
    if not top:
        return "General Findings"
    return " ".join(top).title()


def _group_summary(claims: list[ClaimRecord], title: str) -> str:
    if len(claims) == 1:
        return claims[0].text
    return f"{len(claims)} related claims discuss {title.lower()}."


def _has_conflict(claims: list[ClaimRecord]) -> bool:
    stances = {claim.stance for claim in claims}
    return "supporting" in stances and "opposing" in stances


def group_claims(claims: list[ClaimRecord], threshold: float) -> list[ClaimGroup]:
    if not claims:
        return []

    raw_groups: list[dict] = []
    for claim in claims:
        best_group_index = -1
        best_score = 0.0

        for idx, group in enumerate(raw_groups):
            representative = group["representative"]
            score = _similarity(claim.text, representative)
            if score > best_score:
                best_score = score
                best_group_index = idx

        if best_group_index >= 0 and best_score >= threshold:
            raw_groups[best_group_index]["claims"].append(claim)
            if len(claim.text) > len(raw_groups[best_group_index]["representative"]):
                raw_groups[best_group_index]["representative"] = claim.text
        else:
            raw_groups.append({"representative": claim.text, "claims": [claim]})

    raw_groups.sort(key=lambda group: len(group["claims"]), reverse=True)

    result: list[ClaimGroup] = []
    for index, group in enumerate(raw_groups, start=1):
        claim_list: list[ClaimRecord] = group["claims"]
        title = _group_title(claim_list)
        source_ids = sorted({claim.source_id for claim in claim_list})
        result.append(
            ClaimGroup(
                group_id=f"G{index}",
                theme_title=title,
                summary=_group_summary(claim_list, title),
                claims=claim_list,
                source_ids=source_ids,
                has_conflict=_has_conflict(claim_list),
            )
        )

    return result
