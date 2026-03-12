from __future__ import annotations

import re
from typing import Iterable

from app.config import AgentConfig
from app.models import ExtractedClaim, ExtractedClaimBatch, SourceDocument, Stance

try:
    from langchain_core.prompts import ChatPromptTemplate
    from langchain_groq import ChatGroq
except Exception:  # pragma: no cover - optional dependency fallback
    ChatPromptTemplate = None
    ChatGroq = None


_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+")


def _normalize(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def _heuristic_stance(sentence: str) -> Stance:
    s = sentence.lower()
    positive_markers = {"necessary", "benefit", "improve", "support", "protect", "increase"}
    negative_markers = {"risk", "harm", "burden", "slow", "unnecessary", "oppose", "cost"}

    has_positive = any(word in s for word in positive_markers)
    has_negative = any(word in s for word in negative_markers)

    if has_positive and not has_negative:
        return "supporting"
    if has_negative and not has_positive:
        return "opposing"
    return "neutral"


class ClaimExtractor:
    def __init__(self, config: AgentConfig):
        self.config = config
        self._chain = None

        if not (config.groq_api_key and ChatGroq and ChatPromptTemplate):
            return

        llm = ChatGroq(
            model=config.model_name,
            api_key=config.groq_api_key,
            temperature=0,
        )
        prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    (
                        "You extract grounded claims from one source. "
                        "Return only claims supported by the provided source text. "
                        "Each claim must include a short direct evidence quote from the source. "
                        "Use stance values: supporting, opposing, or neutral."
                    ),
                ),
                (
                    "human",
                    (
                        "Topic: {topic}\n"
                        "Source title: {title}\n"
                        "Source id: {source_id}\n"
                        "Source text:\n{source_text}\n\n"
                        "Extract up to {max_claims} high-value claims. "
                        "Skip uncertain claims."
                    ),
                ),
            ]
        )
        self._chain = prompt | llm.with_structured_output(ExtractedClaimBatch)

    def extract_for_source(self, topic: str, source: SourceDocument) -> list[ExtractedClaim]:
        clipped_text = source.content[: self.config.max_chars_per_source]
        if not clipped_text:
            return []

        if self._chain is not None:
            try:
                result = self._chain.invoke(
                    {
                        "topic": topic,
                        "title": source.title,
                        "source_id": source.source_id,
                        "source_text": clipped_text,
                        "max_claims": self.config.max_claims_per_source,
                    }
                )
                if isinstance(result, ExtractedClaimBatch):
                    cleaned = self._clean_claims(result.claims)
                    if cleaned:
                        return cleaned
            except Exception:
                pass

        return self._heuristic_extract(clipped_text)

    def _clean_claims(self, claims: Iterable[ExtractedClaim]) -> list[ExtractedClaim]:
        cleaned: list[ExtractedClaim] = []
        for claim in claims:
            claim_text = _normalize(claim.claim)
            evidence = _normalize(claim.evidence)
            if not claim_text or not evidence:
                continue
            cleaned.append(
                ExtractedClaim(
                    claim=claim_text,
                    evidence=evidence,
                    stance=claim.stance,
                )
            )
            if len(cleaned) >= self.config.max_claims_per_source:
                break
        return cleaned

    def _heuristic_extract(self, source_text: str) -> list[ExtractedClaim]:
        sentences = [_normalize(s) for s in _SENTENCE_SPLIT_RE.split(source_text) if _normalize(s)]
        candidates = [s for s in sentences if len(s) >= 45]
        if not candidates:
            return []

        selected = candidates[: self.config.max_claims_per_source]
        return [
            ExtractedClaim(
                claim=sentence,
                evidence=sentence,
                stance=_heuristic_stance(sentence),
            )
            for sentence in selected
        ]
