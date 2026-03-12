from __future__ import annotations

import asyncio
import importlib
from typing import Any

import structlog
from pydantic import BaseModel, Field

from research_agent.config import RuntimeLLMConfig, Settings
from research_agent.core.clustering import select_canonical_claim
from research_agent.models import ClaimCluster, ClaimRecord
from research_agent.models.claim_models import StanceLabel

try:
    from transformers import pipeline
except Exception:  # pragma: no cover - optional dependency
    pipeline = None

try:
    import torch
except Exception:  # pragma: no cover - optional dependency
    torch = None

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


class ClaimJudgment(BaseModel):
    claim_id: str
    stance: StanceLabel


class ClusterJudgment(BaseModel):
    canonical_claim: str
    confidence_score: float = Field(default=0.5, ge=0.0, le=1.0)
    supporting_source_ids: list[str] = Field(default_factory=list)
    contradicting_source_ids: list[str] = Field(default_factory=list)
    claim_judgments: list[ClaimJudgment] = Field(default_factory=list)


class StanceService:
    def __init__(self, settings: Settings, logger: structlog.stdlib.BoundLogger):
        self.settings = settings
        self.logger = logger
        self._classifier: Any = None

    def _build_llm_chain(self, runtime: RuntimeLLMConfig | None):
        if runtime is None or not runtime.api_key or ChatPromptTemplate is None:
            return None

        provider = runtime.provider
        model_name = runtime.model_name or self.settings.default_model_name
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
            self.logger.warning("stance.llm_provider_unavailable", provider=provider)
            return None

        prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    (
                        "You are a strict factual adjudicator for clustered claims. "
                        "Given a canonical claim and claim list, classify each claim as supporting, contradicting, or neutral. "
                        "Return valid JSON matching schema exactly. "
                        "Use source IDs exactly as provided. "
                        "Set confidence_score from 0 to 1, where 0.5 means mixed/uncertain and extremes are strong consensus."
                    ),
                ),
                (
                    "human",
                    (
                        "Canonical claim candidate:\n{canonical_claim}\n\n"
                        "Claims in cluster:\n{claims_text}\n\n"
                        "Return:\n"
                        "- improved canonical_claim\n"
                        "- confidence_score (0..1)\n"
                        "- supporting_source_ids\n"
                        "- contradicting_source_ids\n"
                        "- claim_judgments with one entry for every claim_id"
                    ),
                ),
            ]
        )
        return prompt | llm.with_structured_output(ClusterJudgment)

    async def _ensure_classifier(self):
        if self._classifier is not None:
            return self._classifier

        pipeline_factory = pipeline
        if pipeline_factory is None:
            self.logger.warning("stance.classifier_unavailable", reason="transformers_not_installed")
            return None

        device = -1
        if torch is not None and torch.cuda.is_available():
            device = 0

        def _load_pipeline() -> Any:
            return pipeline_factory(
                "zero-shot-classification",
                model=self.settings.nli_model_name,
                device=device,
            )

        try:
            self._classifier = await asyncio.to_thread(_load_pipeline)
            self.logger.info(
                "stance.classifier_ready",
                model=self.settings.nli_model_name,
                device="cuda" if device == 0 else "cpu",
            )
            return self._classifier
        except Exception as exc:  # noqa: BLE001
            self.logger.warning("stance.classifier_load_failed", error=str(exc))
            return None

    async def _classify_stance_fallback(self, canonical_claim: str, candidate_claim: str) -> StanceLabel:
        classifier = await self._ensure_classifier()
        if classifier is None:
            return "neutral"

        labels = ["supports", "contradicts", "neutral"]

        def _infer() -> StanceLabel:
            result = classifier(
                candidate_claim,
                labels,
                hypothesis_template=f"This statement {{}} the canonical claim: {canonical_claim}",
            )
            if not isinstance(result, dict):
                return "neutral"
            raw_labels = result.get("labels")
            if not isinstance(raw_labels, list) or not raw_labels:
                return "neutral"
            top_label = str(raw_labels[0]).lower()
            if "support" in top_label:
                return "supporting"
            if "contradict" in top_label:
                return "contradicting"
            return "neutral"

        try:
            return await asyncio.to_thread(_infer)
        except Exception as exc:  # noqa: BLE001
            self.logger.warning("stance.classification_failed", error=str(exc))
            return "neutral"

    async def _judge_cluster_with_llm(
        self,
        chain,
        canonical_claim: str,
        claims: list[ClaimRecord],
    ) -> ClusterJudgment | None:
        claims_lines: list[str] = []
        for claim in claims:
            claims_lines.append(
                (
                    f"- claim_id={claim.claim_id}; source_id={claim.source_id}; "
                    f"source_url={claim.source_url}; claim={claim.claim_text}; evidence={claim.evidence}"
                )
            )
        claims_text = "\n".join(claims_lines)

        payload = {
            "canonical_claim": canonical_claim,
            "claims_text": claims_text,
        }

        try:
            if hasattr(chain, "ainvoke"):
                result = await chain.ainvoke(payload)
            else:
                result = await asyncio.to_thread(chain.invoke, payload)
        except Exception as exc:  # noqa: BLE001
            self.logger.warning("stance.llm_judgment_failed", error=str(exc))
            return None

        if not isinstance(result, ClusterJudgment):
            return None
        return result

    async def build_clusters(
        self,
        grouped_claims: list[list[ClaimRecord]],
        runtime: RuntimeLLMConfig | None = None,
    ) -> list[ClaimCluster]:
        llm_chain = self._build_llm_chain(runtime)
        clusters: list[ClaimCluster] = []
        llm_cluster_count = 0
        fallback_cluster_count = 0

        for group in grouped_claims:
            canonical_candidate = select_canonical_claim(group)
            source_id_to_url = {claim.source_id: claim.source_url for claim in group}

            llm_judgment: ClusterJudgment | None = None
            if llm_chain is not None:
                llm_judgment = await self._judge_cluster_with_llm(llm_chain, canonical_candidate, group)

            if llm_judgment is not None:
                claim_to_stance = {item.claim_id: item.stance for item in llm_judgment.claim_judgments}
                updated_claims: list[ClaimRecord] = []
                supporting_sources: set[str] = set()
                contradicting_sources: set[str] = set()

                for claim in group:
                    stance = claim_to_stance.get(claim.claim_id, "neutral")
                    updated_claim = claim.model_copy(update={"stance": stance})
                    updated_claims.append(updated_claim)
                    if stance == "supporting":
                        supporting_sources.add(claim.source_url)
                    elif stance == "contradicting":
                        contradicting_sources.add(claim.source_url)

                for source_id in llm_judgment.supporting_source_ids:
                    if source_id in source_id_to_url:
                        supporting_sources.add(source_id_to_url[source_id])
                for source_id in llm_judgment.contradicting_source_ids:
                    if source_id in source_id_to_url:
                        contradicting_sources.add(source_id_to_url[source_id])

                confidence = max(0.0, min(1.0, float(llm_judgment.confidence_score)))
                canonical_claim = llm_judgment.canonical_claim.strip() or canonical_candidate
                llm_cluster_count += 1

                self.logger.info(
                    "stance.cluster_mode",
                    mode="llm",
                    canonical_claim=canonical_claim,
                    claims=len(group),
                    confidence=confidence,
                )
            else:
                updated_claims = []
                supporting_sources: set[str] = set()
                contradicting_sources: set[str] = set()

                for claim in group:
                    stance = await self._classify_stance_fallback(canonical_candidate, claim.claim_text)
                    updated_claim = claim.model_copy(update={"stance": stance})
                    updated_claims.append(updated_claim)
                    if stance == "supporting":
                        supporting_sources.add(claim.source_url)
                    elif stance == "contradicting":
                        contradicting_sources.add(claim.source_url)

                denominator = len(supporting_sources) + len(contradicting_sources)
                confidence = float(len(supporting_sources) / denominator) if denominator else 0.0
                canonical_claim = canonical_candidate
                fallback_cluster_count += 1

                self.logger.info(
                    "stance.cluster_mode",
                    mode="fallback",
                    canonical_claim=canonical_claim,
                    claims=len(group),
                    confidence=confidence,
                )

            clusters.append(
                ClaimCluster(
                    canonical_claim=canonical_claim,
                    supporting_sources=sorted(supporting_sources),
                    contradicting_sources=sorted(contradicting_sources),
                    confidence_score=confidence,
                    claims=updated_claims,
                )
            )

        self.logger.info(
            "stance.mode_summary",
            llm_clusters=llm_cluster_count,
            fallback_clusters=fallback_cluster_count,
            total_clusters=len(clusters),
        )
        self.logger.info("stance.completed", clusters=len(clusters))
        return clusters
