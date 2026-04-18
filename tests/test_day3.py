"""
Day 3 tests — RAG pipeline, LLM service, memory service, report generation.
All external services (Ollama, DynamoDB, ChromaDB) are stubbed so tests
run fully offline.

Run with: pytest tests/test_day3.py -v
"""
import asyncio
import uuid
from unittest.mock import AsyncMock, MagicMock, patch
import chromadb
import pytest

from backend.config import get_settings
from backend.models.schemas import (
    RAGQueryRequest, ReportRequest, ReportFormat,
    MemoryStoreRequest, MemoryRetrieveRequest,
    EmbeddingQueryRequest,
)
from backend.services.llm_service import LLMService
from backend.services.rag_service import RAGService
from backend.services.memory_service import MemoryService
from backend.services.embedding_service import EmbeddingService
from backend.services.vector_db import VectorDBService

settings = get_settings()
loop = asyncio.new_event_loop()


# ── Shared stubs ───────────────────────────────────────────────────────────

class _MockDynamo:
    async def put_paper(self, *a, **kw): pass
    async def get_paper(self, *a, **kw): return None
    async def put_memory_entry(self, *a, **kw): return str(uuid.uuid4())
    async def get_session_entries(self, *a, **kw): return []


def _ephemeral_vdb():
    vdb = VectorDBService(settings)
    vdb._client = chromadb.EphemeralClient()
    return vdb


class _MockLLM:
    """Offline LLM stub — returns a canned answer instantly."""
    async def generate(self, prompt: str, system=None, **kwargs) -> str:
        return "This is a test answer based on the provided research context."

    async def check_health(self) -> bool:
        return False

    async def rewrite_query(self, query: str, memory_context: str = "") -> dict:
        return {"search_query": query, "retrieval_query": query, "answer_question": query}

    async def generate_long_answer(self, query: str, context_chunks: list, memory_context: str = "", max_passes: int = 3) -> str:
        return "This is a test answer based on the provided research context."

    async def generate_long_report(self, topic: str, context_chunks: list, max_length: int = 800, **kwargs) -> str:
        return "This is a test report."

    def build_rag_prompt(self, query, chunks, memory_context="") -> str:
        return LLMService(settings).build_rag_prompt(query, chunks, memory_context)

    def build_report_prompt(self, topic, chunks, max_length=800) -> str:
        return LLMService(settings).build_report_prompt(topic, chunks, max_length)


# ── LLM Service ────────────────────────────────────────────────────────────

def test_llm_fallback_when_offline():
    """LLMService.generate should return fallback text when Ollama is unreachable."""
    svc = LLMService(settings)
    # Point to a definitely-closed port
    svc.base_url = "http://localhost:19999"
    answer = loop.run_until_complete(svc.generate("What is BERT?"))
    assert "unavailable" in answer.lower() or "ollama" in answer.lower()


def test_llm_rag_prompt_contains_query():
    svc = LLMService(settings)
    from backend.models.schemas import RetrievedChunk
    chunks = [
        RetrievedChunk(
            chunk_id="c1", paper_id="p1",
            paper_title="BERT Paper", text="BERT uses bidirectional encoders.",
            score=0.9, chunk_index=0,
        )
    ]
    prompt = svc.build_rag_prompt("What is BERT?", chunks)
    assert "What is BERT?" in prompt
    assert "BERT uses bidirectional encoders." in prompt
    assert "[Source 1" in prompt


def test_llm_report_prompt_contains_topic():
    svc = LLMService(settings)
    prompt = svc.build_report_prompt("Transformer attention", [], max_length=500)
    assert "Transformer attention" in prompt
    assert "500" in prompt
    assert "## Summary" in prompt


# ── RAG Service ────────────────────────────────────────────────────────────

def _make_rag_svc_with_data():
    """Build a RAGService fully wired to in-memory stubs, pre-populated with data."""
    emb_svc = EmbeddingService(settings)

    # Pre-populate vector DB with 3 chunks
    vdb = _ephemeral_vdb()
    texts = [
        "BERT is a bidirectional transformer pretrained on masked language modeling.",
        "GPT uses autoregressive language modeling for text generation.",
        "Attention mechanisms allow models to weigh token importance dynamically.",
    ]
    embeddings = loop.run_until_complete(emb_svc.embed_batch(texts))
    chunks = [
        {
            "chunk_id": f"rc_{i}",
            "paper_id": f"paper_{i}",
            "text": t,
            "chunk_index": i,
            "metadata": {
                "paper_id": f"paper_{i}",
                "paper_title": f"Test Paper {i}",
                "chunk_index": i,
                "char_start": 0,
                "char_end": len(t),
            },
        }
        for i, t in enumerate(texts)
    ]
    loop.run_until_complete(
        vdb.store_chunks(settings.chroma_collection_papers, chunks, embeddings)
    )

    # Build service with stubs
    svc = RAGService(settings)
    svc._embedding_svc = emb_svc
    svc._vector_db = vdb
    svc._llm = _MockLLM()
    svc._memory_svc.store = AsyncMock(return_value=str(uuid.uuid4()))
    svc._memory_svc._dynamo = _MockDynamo()
    svc._memory_svc._vector_db = _ephemeral_vdb()

    return svc


def test_rag_query_returns_answer():
    svc = _make_rag_svc_with_data()
    req = RAGQueryRequest(query="What is BERT?", top_k=2, use_memory=False)
    result = loop.run_until_complete(svc.query(req))

    assert result.answer == "This is a test answer based on the provided research context."
    assert result.query == "What is BERT?"
    assert result.session_id is not None
    assert result.model_used == settings.ollama_model


def test_rag_query_retrieves_relevant_chunks():
    svc = _make_rag_svc_with_data()
    req = RAGQueryRequest(query="How does attention work?", top_k=3, use_memory=False)
    result = loop.run_until_complete(svc.query(req))

    assert len(result.sources) >= 1
    # At least one source should be about attention
    texts = [s.text.lower() for s in result.sources]
    assert any("attention" in t for t in texts)


def test_rag_query_session_id_preserved():
    svc = _make_rag_svc_with_data()
    sid = str(uuid.uuid4())
    req = RAGQueryRequest(query="Tell me about GPT", session_id=sid, use_memory=False)
    result = loop.run_until_complete(svc.query(req))
    assert result.session_id == sid


def test_rag_query_paper_filter():
    svc = _make_rag_svc_with_data()
    req = RAGQueryRequest(
        query="What is BERT?",
        top_k=3,
        paper_ids=["paper_0"],   # only BERT paper
        use_memory=False,
    )
    result = loop.run_until_complete(svc.query(req))
    # All returned sources must belong to the filtered paper
    for src in result.sources:
        assert src.paper_id == "paper_0"


# ── Report Generation ──────────────────────────────────────────────────────

def test_report_generation():
    svc = _make_rag_svc_with_data()
    req = ReportRequest(
        topic="Transformer attention mechanisms",
        max_length=400,
        format=ReportFormat.MARKDOWN,
    )
    report, sources = loop.run_until_complete(svc.generate_report(req))
    assert len(report) > 10
    assert isinstance(sources, list)


# ── Memory Service ─────────────────────────────────────────────────────────

def _make_memory_svc():
    svc = MemoryService(settings)
    svc._dynamo = _MockDynamo()
    svc._vector_db = _ephemeral_vdb()
    return svc


def test_memory_store():
    svc = _make_memory_svc()
    req = MemoryStoreRequest(
        session_id="test_session_123",
        role="user",
        content="What are transformers?",
    )
    entry_id = loop.run_until_complete(svc.store(req))
    assert entry_id is not None
    assert len(entry_id) > 0


def test_memory_store_and_semantic_retrieve():
    svc = _make_memory_svc()
    sid = "test_session_sem"

    # Store two turns
    loop.run_until_complete(svc.store(MemoryStoreRequest(
        session_id=sid, role="user", content="Explain BERT's pretraining objective."
    )))
    loop.run_until_complete(svc.store(MemoryStoreRequest(
        session_id=sid, role="assistant",
        content="BERT is pretrained using masked language modeling and next sentence prediction."
    )))

    # Semantic retrieval
    result = loop.run_until_complete(svc.retrieve(
        MemoryRetrieveRequest(session_id=sid, limit=5, query="BERT pretraining")
    ))
    assert result.session_id == sid
    # Should find at least one relevant entry
    assert len(result.entries) >= 1


def test_memory_store_and_chronological_retrieve():
    """Chronological retrieve falls back gracefully when DynamoDB returns empty."""
    svc = _make_memory_svc()
    sid = "test_session_chron"
    result = loop.run_until_complete(svc.retrieve(
        MemoryRetrieveRequest(session_id=sid, limit=10)
    ))
    assert result.session_id == sid
    assert isinstance(result.entries, list)


def test_memory_role_stored_correctly():
    """Entries stored in ChromaDB should use 'user'/'assistant' in paper_title field."""
    svc = _make_memory_svc()
    sid = "test_role_session"

    loop.run_until_complete(svc.store(MemoryStoreRequest(
        session_id=sid, role="user", content="Tell me about attention mechanisms."
    )))

    result = loop.run_until_complete(svc.retrieve(
        MemoryRetrieveRequest(session_id=sid, limit=5, query="attention mechanisms")
    ))
    if result.entries:
        assert result.entries[0].role in ("user", "assistant")


# ── Embedding query endpoint ───────────────────────────────────────────────

def test_embedding_query_via_service():
    """
    EmbeddingService.query embeds the question then calls VectorDBService.query_similar.
    We test the full flow by injecting the real VDB (ephemeral) into the service.
    """
    emb_svc = EmbeddingService(settings)
    vdb = _ephemeral_vdb()

    # Pre-populate
    text = "Self-attention is the core of transformers."
    embeddings = loop.run_until_complete(emb_svc.embed_batch([text]))
    chunk = [{
        "chunk_id": "eq_test_0",
        "paper_id": "eq_paper",
        "text": text,
        "chunk_index": 0,
        "metadata": {
            "paper_id": "eq_paper", "paper_title": "Attention Test",
            "chunk_index": 0, "char_start": 0, "char_end": len(text),
        },
    }]
    loop.run_until_complete(vdb.store_chunks(settings.chroma_collection_papers, chunk, embeddings))

    # VectorDBService is a local import inside EmbeddingService.query():
    #   from backend.services.vector_db import VectorDBService
    # Patching at the definition module makes the local import resolve to our stub.
    with patch("backend.services.vector_db.VectorDBService", return_value=vdb):
        req = EmbeddingQueryRequest(query="attention transformer", top_k=1)
        result = loop.run_until_complete(emb_svc.query(req))

    assert result.query == "attention transformer"
    assert len(result.results) == 1
    assert result.results[0].paper_id == "eq_paper"
    assert result.results[0].score > 0.0
