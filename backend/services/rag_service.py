"""
RAG Service — full pipeline from query to LLM-grounded answer.
Pipeline:
  1. Rewrite user query (LLM) → clean search terms + well-formed question
  2. Embed the rewritten search query → ChromaDB retrieval
  3. Load session memory
  4. Multi-pass LLM generation (continues if answer is truncated)
  5. Store Q&A turn in memory
"""
import time
import uuid
from typing import List, Tuple, Optional

from loguru import logger

from backend.config import Settings
from backend.models.schemas import (
    RAGQueryRequest, RAGQueryResponse,
    ReportRequest, RetrievedChunk,
)
from backend.services.embedding_service import EmbeddingService
from backend.services.vector_db import VectorDBService
from backend.services.memory_service import MemoryService
from backend.services.llm_service import LLMService
from backend.services.mlflow_service import MLflowService
from backend.services.dynamo_db import DynamoDBService


class RAGService:

    def __init__(self, settings: Settings):
        self.settings = settings
        self._embedding_svc = EmbeddingService(settings)
        self._vector_db = VectorDBService(settings)
        self._memory_svc = MemoryService(settings)
        self._llm = LLMService(settings)
        self._dynamo = DynamoDBService(settings)
        self._mlflow = MLflowService(settings.mlflow_tracking_uri, settings.mlflow_experiment_name)

    # ── RAG Query ──────────────────────────────────────────────────────────

    async def query(self, request: RAGQueryRequest) -> RAGQueryResponse:
        t_total = time.perf_counter()
        session_id = request.session_id or str(uuid.uuid4())

        # 1. Load memory FIRST so rewrite_query can contextualise follow-up questions
        memory_context = ""
        if request.use_memory and request.session_id:
            from backend.models.schemas import MemoryRetrieveRequest
            mem_response = await self._memory_svc.retrieve(
                MemoryRetrieveRequest(session_id=request.session_id, limit=6)
            )
            if mem_response.entries:
                memory_context = "\n".join(
                    f"{e.role.capitalize()}: {e.content[:300]}"
                    for e in mem_response.entries
                )

        # 2. Rewrite query with memory context so short follow-ups ("give equation",
        #    "explain more") are expanded using the conversation history.
        rewritten = await self._llm.rewrite_query(request.query, memory_context=memory_context)
        retrieval_query = rewritten["retrieval_query"]
        answer_question = rewritten["answer_question"]
        paper_name = rewritten.get("paper_name")

        # 2b. Auto-ingest: if a specific paper is detected but not in DB, fetch & ingest it
        if paper_name:
            existing = await self._dynamo.scan_papers_by_keyword(paper_name)
            if not existing:
                logger.info(f"[RAGService] '{paper_name}' not in DB — auto-ingesting...")
                try:
                    from backend.services.ingest_pipeline import IngestPipeline
                    pipeline = IngestPipeline(self.settings)
                    await pipeline.search_and_ingest(
                        query=rewritten["search_query"],
                        max_papers=3,
                    )
                    logger.info(f"[RAGService] auto-ingest complete for '{paper_name}'")
                except Exception as e:
                    logger.warning(f"[RAGService] auto-ingest failed (non-fatal): {e}")

        # 3. Embed the content-specific retrieval query
        query_embedding = await self._embedding_svc.embed_single(retrieval_query)
        retrieval_start = time.perf_counter()

        # 4. Hybrid retrieval: general + paper-specific if a paper was detected
        chunks = await self._hybrid_retrieve(
            query_embedding=query_embedding,
            paper_name=paper_name,
            top_k=max(request.top_k, 8),
            filter_paper_ids=request.paper_ids,
        )
        retrieval_ms = (time.perf_counter() - retrieval_start) * 1000
        logger.debug(f"[RAGService] retrieved {len(chunks)} chunks in {retrieval_ms:.1f}ms")

        # 5. Multi-pass generation with the well-formed question
        t_gen = time.perf_counter()
        answer = await self._llm.generate_long_answer(
            query=answer_question,
            context_chunks=chunks,
            memory_context=memory_context,
            max_passes=3,
        )
        generation_ms = (time.perf_counter() - t_gen) * 1000
        total_ms = (time.perf_counter() - t_total) * 1000
        self._mlflow.log_rag_query(request.query, len(chunks), retrieval_ms, generation_ms, total_ms)

        # Prometheus metrics
        try:
            from backend.services.prom_metrics import (
                RAG_REQUESTS, RAG_TOTAL_DURATION, RAG_RETRIEVAL_DURATION,
                RAG_GENERATION_DURATION, RAG_SOURCES,
            )
            RAG_REQUESTS.labels(status="success").inc()
            RAG_TOTAL_DURATION.observe(total_ms / 1000)
            RAG_RETRIEVAL_DURATION.observe(retrieval_ms / 1000)
            RAG_GENERATION_DURATION.observe(generation_ms / 1000)
            RAG_SOURCES.observe(len(chunks))
        except Exception:
            pass

        # 6. Store Q&A turn in memory (original query for natural history)
        try:
            from backend.models.schemas import MemoryStoreRequest
            await self._memory_svc.store(MemoryStoreRequest(
                session_id=session_id, role="user", content=request.query
            ))
            await self._memory_svc.store(MemoryStoreRequest(
                session_id=session_id, role="assistant", content=answer
            ))
            # Update session activity counter (for users who registered the session)
            if request.user_id:
                await self._dynamo.update_session_activity(request.user_id, session_id)
        except Exception as e:
            logger.warning(f"[RAGService] memory store failed (non-fatal): {e}")

        return RAGQueryResponse(
            query=request.query,
            answer=answer,
            session_id=session_id,
            sources=chunks,
            model_used=self.settings.ollama_model,
        )

    # ── Hybrid Retrieval ───────────────────────────────────────────────────

    async def _hybrid_retrieve(
        self,
        query_embedding: List[float],
        paper_name: Optional[str],
        top_k: int,
        filter_paper_ids: Optional[List[str]],
    ) -> List[RetrievedChunk]:
        """
        If a specific paper is detected in the query, do TWO searches:
          1. Paper-specific: filtered ChromaDB search within that paper's chunks (top 6)
          2. General: unfiltered semantic search (top 5)
        Paper-specific chunks come first so the LLM has guaranteed access to the right content.
        Falls back to general-only if no paper detected or no match found in DynamoDB.
        """
        # General retrieval always runs
        general = await self._vector_db.query_similar(
            collection_name=self.settings.chroma_collection_papers,
            query_embedding=query_embedding,
            top_k=top_k,
            filter_paper_ids=filter_paper_ids,
        )

        if not paper_name:
            return general

        # Look up the paper in DynamoDB by keyword match on title
        matched = await self._dynamo.scan_papers_by_keyword(paper_name)
        if not matched:
            logger.debug(f"[RAGService] no paper found for name='{paper_name}', using general only")
            return general

        matched_ids = [m["paper_id"] for m in matched]
        logger.info(f"[RAGService] paper '{paper_name}' matched IDs: {matched_ids}")

        # Paper-specific retrieval — guaranteed chunks from the target paper
        specific = await self._vector_db.query_similar(
            collection_name=self.settings.chroma_collection_papers,
            query_embedding=query_embedding,
            top_k=8,
            filter_paper_ids=matched_ids,
        )

        if not specific:
            return general

        # Merge: specific first, then general chunks not already included
        seen = {c.chunk_id for c in specific}
        merged = specific + [c for c in general if c.chunk_id not in seen]
        logger.info(f"[RAGService] hybrid: {len(specific)} specific + {len(general)} general = {len(merged)} total")
        return merged[:12]

    # ── Report Generation ──────────────────────────────────────────────────

    async def generate_report(self, request: ReportRequest) -> Tuple[str, List[str]]:
        rewritten = await self._llm.rewrite_query(request.topic)
        query_embedding = await self._embedding_svc.embed_single(rewritten["retrieval_query"])

        chunks = await self._vector_db.query_similar(
            collection_name=self.settings.chroma_collection_papers,
            query_embedding=query_embedding,
            top_k=15,
            filter_paper_ids=request.paper_ids,
        )

        # Multi-pass generation — each pass ~700 words, up to 50 passes = ~35k words
        target_words = min(request.max_length, 50000)
        passes_needed = max(3, min(50, target_words // 700 + 2))

        report = await self._llm.generate_long_report(
            topic=request.topic,
            context_chunks=chunks,
            max_length=target_words,
            max_passes=passes_needed,
        )

        sources = list({c.paper_id for c in chunks if c.paper_id})
        return report, sources
