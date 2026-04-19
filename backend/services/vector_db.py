"""
Detroit Vector Database Service — backed by ChromaDB.
Handles storing and querying 384-dim paper chunk embeddings.
"""
import asyncio
from typing import List, Optional

import chromadb
from chromadb.config import Settings as ChromaSettings
from loguru import logger

from backend.config import Settings
from backend.models.schemas import RetrievedChunk


class VectorDBService:

    def __init__(self, settings: Settings):
        self.settings = settings
        self._client: chromadb.ClientAPI | None = None

    # ── Lazy connection ────────────────────────────────────────────────────

    def _get_client(self) -> chromadb.ClientAPI:
        if self._client is None:
            try:
                # Try to connect to ChromaDB server (Docker)
                self._client = chromadb.HttpClient(
                    host=self.settings.chroma_host,
                    port=self.settings.chroma_port,
                )
                self._client.heartbeat()
                logger.info(f"[VectorDB] Connected to ChromaDB at {self.settings.chroma_host}:{self.settings.chroma_port}")
            except Exception:
                # Fall back to local persistent client (dev without Docker)
                logger.warning("[VectorDB] ChromaDB server unavailable, using local persistent client")
                self._client = chromadb.PersistentClient(path="data/embeddings/chroma")
        return self._client

    def _get_collection(self, name: str):
        client = self._get_client()
        return client.get_or_create_collection(
            name=name,
            metadata={"hnsw:space": "cosine"},
        )

    # ── Store ──────────────────────────────────────────────────────────────

    async def store_chunks(
        self,
        collection_name: str,
        chunks: List[dict],
        embeddings: List[List[float]],
    ) -> int:
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            None, self._store_sync, collection_name, chunks, embeddings
        )

    def _store_sync(
        self,
        collection_name: str,
        chunks: List[dict],
        embeddings: List[List[float]],
    ) -> int:
        collection = self._get_collection(collection_name)

        ids = [c["chunk_id"] for c in chunks]
        documents = [c["text"] for c in chunks]
        metadatas = [c["metadata"] for c in chunks]

        # ChromaDB upsert (idempotent re-ingestion)
        collection.upsert(
            ids=ids,
            embeddings=embeddings,
            documents=documents,
            metadatas=metadatas,
        )
        logger.debug(f"[VectorDB] upserted {len(ids)} chunks into '{collection_name}'")
        return len(ids)

    # ── Query ──────────────────────────────────────────────────────────────

    async def query_similar(
        self,
        collection_name: str,
        query_embedding: List[float],
        top_k: int = 5,
        filter_paper_ids: Optional[List[str]] = None,
    ) -> List[RetrievedChunk]:
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            None, self._query_sync, collection_name, query_embedding, top_k, filter_paper_ids
        )

    def _query_sync(
        self,
        collection_name: str,
        query_embedding: List[float],
        top_k: int,
        filter_paper_ids: Optional[List[str]],
    ) -> List[RetrievedChunk]:
        collection = self._get_collection(collection_name)

        where = None
        if filter_paper_ids:
            where = {"paper_id": {"$in": filter_paper_ids}}

        count = collection.count()
        if count == 0:
            return []

        # Note: in chromadb >=1.x "ids" are always returned; only specify extras
        results = collection.query(
            query_embeddings=[query_embedding],
            n_results=min(top_k, count),
            where=where,
            include=["documents", "metadatas", "distances"],
        )

        chunks = []
        ids   = (results.get("ids")   or [[]])[0]   # always present in chromadb >=1.x
        docs  = (results.get("documents") or [[]])[0]
        metas = (results.get("metadatas") or [[]])[0]
        dists = (results.get("distances") or [[]])[0]

        for chunk_id, doc, meta, dist in zip(ids, docs, metas, dists):
            # ChromaDB cosine distance: score = 1 - distance
            score = round(1.0 - dist, 4)
            chunks.append(
                RetrievedChunk(
                    chunk_id=chunk_id,
                    paper_id=meta.get("paper_id", ""),
                    paper_title=meta.get("paper_title", ""),
                    text=doc,
                    score=score,
                    chunk_index=meta.get("chunk_index", 0),
                    metadata=meta,
                )
            )

        return chunks

    # ── Store memory entry ─────────────────────────────────────────────────

    async def store_memory_entry(
        self,
        entry_id: str,
        text: str,
        embedding: List[float],
        metadata: dict,
    ):
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(
            None, self._store_memory_sync, entry_id, text, embedding, metadata
        )

    def _store_memory_sync(self, entry_id, text, embedding, metadata):
        collection = self._get_collection(self.settings.chroma_collection_memory)
        collection.upsert(
            ids=[entry_id],
            embeddings=[embedding],
            documents=[text],
            metadatas=[metadata],
        )
