from __future__ import annotations

import asyncio

import structlog

from research_agent.models import ClaimCluster

try:
    import chromadb
except Exception:  # pragma: no cover - optional dependency
    chromadb = None


class OptionalChromaStore:
    def __init__(self, collection_name: str, logger: structlog.stdlib.BoundLogger):
        self.collection_name = collection_name
        self.logger = logger
        self._collection = None

    async def _ensure_collection(self):
        if self._collection is not None:
            return self._collection

        if chromadb is None:
            self.logger.info("vector_store.disabled", reason="chromadb_not_installed")
            return None

        def _create_collection():
            client = chromadb.Client()
            return client.get_or_create_collection(name=self.collection_name)

        try:
            self._collection = await asyncio.to_thread(_create_collection)
            return self._collection
        except Exception as exc:  # noqa: BLE001
            self.logger.warning("vector_store.init_failed", error=str(exc))
            return None

    async def upsert_clusters(self, clusters: list[ClaimCluster]) -> bool:
        collection = await self._ensure_collection()
        if collection is None:
            return False

        ids: list[str] = []
        documents: list[str] = []
        metadatas: list[dict] = []

        for index, cluster in enumerate(clusters, start=1):
            ids.append(f"cluster-{index}")
            documents.append(cluster.canonical_claim)
            metadatas.append(
                {
                    "confidence_score": cluster.confidence_score,
                    "supporting_count": len(cluster.supporting_sources),
                    "contradicting_count": len(cluster.contradicting_sources),
                }
            )

        try:
            await asyncio.to_thread(collection.upsert, ids=ids, documents=documents, metadatas=metadatas)
            self.logger.info("vector_store.upserted", clusters=len(clusters))
            return True
        except Exception as exc:  # noqa: BLE001
            self.logger.warning("vector_store.upsert_failed", error=str(exc))
            return False
