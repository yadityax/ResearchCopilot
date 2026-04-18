"""
Adaptive Memory Service
Dual-write: full history in DynamoDB, semantic search in ChromaDB.
Interest score decay: score(t) = base_score * exp(-λ * days_since_access)
"""
import asyncio
import math
from datetime import datetime, timezone
from typing import List

from loguru import logger

from backend.config import Settings
from backend.models.schemas import (
    MemoryStoreRequest, MemoryRetrieveRequest,
    MemoryRetrieveResponse, MemoryEntry,
)
from backend.services.dynamo_db import DynamoDBService
from backend.services.vector_db import VectorDBService
from backend.services.embedding_service import EmbeddingService


class MemoryService:
    # Interest score decay: λ controls how fast scores decay over time
    # score(t) = base * exp(-λ * days_since_access)
    DECAY_LAMBDA = 0.05   # half-life ≈ 14 days
    BASE_SCORE = 1.0

    def __init__(self, settings: Settings):
        self.settings = settings
        self._dynamo = DynamoDBService(settings)
        self._vector_db = VectorDBService(settings)
        self._embedding_svc = EmbeddingService(settings)

    @staticmethod
    def _decay_score(stored_at_iso: str, base: float = BASE_SCORE, lam: float = DECAY_LAMBDA) -> float:
        """Compute interest score with exponential time decay."""
        try:
            stored = datetime.fromisoformat(stored_at_iso)
            if stored.tzinfo is None:
                stored = stored.replace(tzinfo=timezone.utc)
            days = (datetime.now(timezone.utc) - stored).total_seconds() / 86400
            return base * math.exp(-lam * days)
        except Exception:
            return base

    async def store(self, request: MemoryStoreRequest) -> str:
        now_iso = datetime.now(timezone.utc).isoformat()

        # 1. Write to DynamoDB (persistent history)
        entry_id = await self._dynamo.put_memory_entry(
            session_id=request.session_id,
            role=request.role,
            content=request.content,
            metadata=request.metadata or {},
        )

        # 2. Embed and write to ChromaDB (semantic retrieval) with stored_at for decay
        try:
            embedding = await self._embedding_svc.embed_single(request.content)
            await self._vector_db.store_memory_entry(
                entry_id=entry_id,
                text=request.content,
                embedding=embedding,
                metadata={
                    "session_id": request.session_id,
                    "role": request.role,
                    "paper_title": request.role,
                    "entry_id": entry_id,
                    "stored_at": now_iso,
                    **(request.metadata or {}),
                },
            )
        except Exception as e:
            logger.warning(f"[MemoryService] ChromaDB write failed (non-fatal): {e}")

        logger.info(f"[MemoryService] stored entry {entry_id} for session {request.session_id}")
        return entry_id

    async def retrieve(self, request: MemoryRetrieveRequest) -> MemoryRetrieveResponse:
        if request.query:
            # Semantic retrieval via ChromaDB
            entries = await self._semantic_retrieve(request)
        else:
            # Chronological retrieval via DynamoDB
            entries = await self._chronological_retrieve(request)

        return MemoryRetrieveResponse(
            session_id=request.session_id,
            entries=entries,
            total=len(entries),
        )

    async def _chronological_retrieve(self, request: MemoryRetrieveRequest) -> List[MemoryEntry]:
        raw = await self._dynamo.get_session_entries(request.session_id, request.limit)
        return [
            MemoryEntry(
                entry_id=r.get("entry_id", ""),
                session_id=r.get("session_id", ""),
                role=r.get("role", ""),
                content=r.get("content", ""),
                timestamp=r.get("timestamp", ""),
                metadata=r.get("metadata", {}),
            )
            for r in raw
        ]

    async def forget_topic(self, session_id: str, topic: str, top_k: int = 20) -> dict:
        """
        Explicit forgetting: embed the topic, find matching memory entries,
        delete them from ChromaDB. Returns count of deleted entries.
        """
        try:
            query_emb = await self._embedding_svc.embed_single(topic)
            chunks = await self._vector_db.query_similar(
                collection_name=self.settings.chroma_collection_memory,
                query_embedding=query_emb,
                top_k=top_k,
            )
            # Only delete entries for this session with high relevance (score > 0.5)
            ids_to_delete = [
                c.chunk_id for c in chunks
                if c.score > 0.5 and
                   (not hasattr(c, "metadata") or
                    not c.metadata or
                    c.metadata.get("session_id", session_id) == session_id)
            ]
            if ids_to_delete:
                col = self._vector_db._get_collection(self.settings.chroma_collection_memory)
                col.delete(ids=ids_to_delete)
                logger.info(f"[MemoryService] forgot {len(ids_to_delete)} entries about '{topic}'")
            return {"deleted": len(ids_to_delete), "topic": topic}
        except Exception as e:
            logger.error(f"[MemoryService] forget_topic error: {e}")
            return {"deleted": 0, "topic": topic, "error": str(e)}

    async def _semantic_retrieve(self, request: MemoryRetrieveRequest) -> List[MemoryEntry]:
        query_emb = await self._embedding_svc.embed_single(request.query)
        chunks = await self._vector_db.query_similar(
            collection_name=self.settings.chroma_collection_memory,
            query_embedding=query_emb,
            top_k=request.limit * 3,  # fetch extra, re-rank after decay
        )
        scored = []
        for chunk in chunks:
            stored_at = (chunk.metadata or {}).get("stored_at", "") if hasattr(chunk, "metadata") else ""
            decayed = self._decay_score(stored_at) * chunk.score if stored_at else chunk.score
            scored.append((decayed, chunk))

        # Sort by decayed score descending, keep top-k
        scored.sort(key=lambda x: x[0], reverse=True)
        entries = []
        for decayed_score, chunk in scored[: request.limit]:
            entries.append(
                MemoryEntry(
                    entry_id=chunk.chunk_id,
                    session_id=request.session_id,
                    role=chunk.paper_title if chunk.paper_title in ("user", "assistant") else "assistant",
                    content=chunk.text,
                    timestamp="",
                    metadata={"score": chunk.score, "interest_score": round(decayed_score, 4)},
                )
            )
        return entries
