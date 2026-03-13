from __future__ import annotations

import asyncio
import importlib
import re
from typing import Iterable, cast

import structlog
from pydantic import BaseModel, Field

from research_agent.config import RuntimeLLMConfig, Settings
from research_agent.models import ClaimRecord, SourceDocument
from research_agent.models.claim_models import StanceLabel

try:
    from langchain_core.prompts import ChatPromptTemplate
except Exception:  # pragma: no cover - optional dependency
    ChatPromptTemplate = None

try:
    from langchain_groq import ChatGroq
except Exception:  # pragma: no cover - optional dependency
    ChatGroq = None

try:
    from langchain_openai import ChatOpenAI
except Exception:  # pragma: no cover - optional dependency
    ChatOpenAI = None

try:
    ChatAnthropic = getattr(importlib.import_module("langchain_anthropic"), "ChatAnthropic")
except Exception:  # pragma: no cover - optional dependency
    ChatAnthropic = None


_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+")
_SPACE_RE = re.compile(r"\s+")


class ExtractedClaim(BaseModel):
    claim_text: str
    evidence: str


class ExtractedClaimBatch(BaseModel):
    claims: list[ExtractedClaim] = Field(default_factory=list)


def _normalize(text: str) -> str:
    return _SPACE_RE.sub(" ", text).strip()


def _heuristic_stance(sentence: str) -> StanceLabel:
    lowered = sentence.lower()
    support_markers = {"benefit", "improve", "support", "increase", "necessary", "effective"}
    contradict_markers = {"risk", "harm", "oppose", "unnecessary", "burden", "decrease"}
    has_support = any(word in lowered for word in support_markers)
    has_contradict = any(word in lowered for word in contradict_markers)
    if has_support and not has_contradict:
        return "supporting"
    if has_contradict and not has_support:
        return "contradicting"
    return "neutral"


class ClaimExtractionService:
    def __init__(self, settings: Settings, logger: structlog.stdlib.BoundLogger):
        self.settings = settings
        self.logger = logger

    def _build_chain(self, runtime: RuntimeLLMConfig):
        if not runtime.api_key or ChatPromptTemplate is None:
            return None

        model_name = runtime.model_name or self.settings.default_model_name
        provider = runtime.provider
        api_key = runtime.api_key

        if provider == "groq" and ChatGroq is not None:
            llm = ChatGroq(model=model_name, api_key=api_key, temperature=0)
        elif provider == "openai" and ChatOpenAI is not None:
            llm = ChatOpenAI(model=model_name or "gpt-4o-mini", api_key=api_key, temperature=0)  # pyright: ignore[reportArgumentType]
        elif provider == "anthropic" and ChatAnthropic is not None:
            llm = ChatAnthropic(
                model=model_name or "claude-3-5-haiku-latest",
                api_key=api_key,
                temperature=0,
            )
        else:
            self.logger.info("extraction.llm_provider_unavailable", provider=provider)
            return None

        prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    (
                        "Extract grounded claims from the provided source text. "
                        "Return concise claims with direct evidence snippets. "
                        "Do not invent facts outside the source text."
                    ),
                ),
                (
                    "human",
                    (
                        "Topic: {topic}\n"
                        "Source title: {title}\n"
                        "Source id: {source_id}\n"
                        "Source text:\n{source_text}\n\n"
                        "Return up to {max_claims} claims."
                    ),
                ),
            ]
        )
        return prompt | llm.with_structured_output(ExtractedClaimBatch)

    async def _extract_with_chain(
        self,
        chain,
        topic: str,
        source: SourceDocument,
    ) -> list[ExtractedClaim]:
        payload = {
            "topic": topic,
            "title": source.title,
            "source_id": source.source_id,
            "source_text": source.content[: self.settings.max_chars_per_source],
            "max_claims": self.settings.max_claims_per_source,
        }

        try:
            if hasattr(chain, "ainvoke"):
                result = await chain.ainvoke(
                    payload,
                    config={
                        "run_name": f"extract_claims:{source.source_id}",
                        "tags": ["research_agent", "extraction"],
                        "metadata": {
                            "topic": topic,
                            "source_id": source.source_id,
                            "source_url": source.source_url,
                            "source_title": source.title,
                        },
                    },
                )
            else:
                result = await asyncio.to_thread(chain.invoke, payload)
        except Exception as exc:  # noqa: BLE001
            self.logger.warning("extraction.llm_failed", source=source.source_url, error=str(exc))
            return []

        if not isinstance(result, ExtractedClaimBatch):
            return []
        return self._clean_claims(result.claims)

    def _clean_claims(self, claims: Iterable[ExtractedClaim]) -> list[ExtractedClaim]:
        cleaned: list[ExtractedClaim] = []
        for claim in claims:
            claim_text = _normalize(claim.claim_text)
            evidence = _normalize(claim.evidence)
            if not claim_text or not evidence:
                continue
            cleaned.append(ExtractedClaim(claim_text=claim_text, evidence=evidence))
            if len(cleaned) >= self.settings.max_claims_per_source:
                break
        return cleaned

    def _heuristic_extract(self, source: SourceDocument) -> list[ExtractedClaim]:
        clipped = source.content[: self.settings.max_chars_per_source]
        sentences = [_normalize(chunk) for chunk in _SENTENCE_SPLIT_RE.split(clipped) if _normalize(chunk)]
        candidates = [sentence for sentence in sentences if len(sentence) >= 45]
        selected = candidates[: self.settings.max_claims_per_source]
        return [ExtractedClaim(claim_text=sentence, evidence=sentence) for sentence in selected]

    async def extract_claims(
        self,
        topic: str,
        sources: list[SourceDocument],
        runtime: RuntimeLLMConfig,
    ) -> list[ClaimRecord]:
        chain = self._build_chain(runtime)

        claim_records: list[ClaimRecord] = []
        claim_number = 1
        ai_source_count = 0
        fallback_source_count = 0

        for source in sources:
            mode = "fallback"
            if chain is not None:
                extracted = await self._extract_with_chain(chain, topic, source)
                if extracted:
                    mode = "ai"
                    ai_source_count += 1
                else:
                    extracted = self._heuristic_extract(source)
                    fallback_source_count += 1
            else:
                extracted = self._heuristic_extract(source)
                fallback_source_count += 1

            self.logger.info(
                "extraction.source_mode",
                source_id=source.source_id,
                source=source.source_url,
                mode=mode,
                claims=len(extracted),
            )

            for item in extracted:
                claim_records.append(
                    ClaimRecord(
                        claim_id=f"C{claim_number}",
                        claim_text=item.claim_text,
                        source_url=source.source_url,
                        source_id=source.source_id,
                        source_title=source.title,
                        evidence=item.evidence,
                        stance=cast(StanceLabel, _heuristic_stance(item.claim_text)),
                    )
                )
                claim_number += 1

        self.logger.info(
            "extraction.mode_summary",
            llm_enabled=chain is not None,
            ai_sources=ai_source_count,
            fallback_sources=fallback_source_count,
            total_sources=len(sources),
        )
        self.logger.info("extraction.completed", claims=len(claim_records), sources=len(sources))
        return claim_records
