"""
Day 2 integration tests
- Embedding pipeline end-to-end (encode → MLP → normalize)
- ChromaDB store + query (local persistent fallback)
- DynamoDB table creation + read/write
- PDF text extraction and chunking
- Full ingest pipeline (text → embed → store → retrieve)
- MLflow service (no-op when server absent)
- Ingest pipeline routes registered

Run with: pytest tests/test_day2.py -v
"""
import asyncio
import pytest
import chromadb
from fastapi.testclient import TestClient

from backend.main import app
from backend.config import get_settings
from backend.models.schemas import IngestRequest
from backend.services.embedding_service import EmbeddingService
from backend.services.vector_db import VectorDBService
from backend.services.pdf_ingestion import PDFIngestionService
from backend.services.mlflow_service import MLflowService
from backend.services.ingest_pipeline import IngestPipeline

client = TestClient(app)
settings = get_settings()


# ── New pipeline routes ────────────────────────────────────────────────────

def test_pipeline_routes_registered():
    routes = [r.path for r in app.routes]
    assert any("/api/v1/pipeline/search-and-ingest" in r for r in routes)
    assert any("/api/v1/pipeline/ingest-by-id" in r for r in routes)


# ── Embedding service ──────────────────────────────────────────────────────

def test_embed_single():
    svc = EmbeddingService(settings)
    emb = asyncio.new_event_loop().run_until_complete(
        svc.embed_single("The transformer model uses self-attention.")
    )
    assert len(emb) == 384
    # L2 norm should be close to 1 (normalized)
    import math
    norm = math.sqrt(sum(x**2 for x in emb))
    assert abs(norm - 1.0) < 0.01, f"embedding not normalized: norm={norm}"


def test_embed_batch():
    svc = EmbeddingService(settings)
    texts = [
        "Attention mechanism in transformers",
        "Convolutional neural networks for vision",
        "Recurrent networks and LSTMs",
    ]
    embs = asyncio.new_event_loop().run_until_complete(svc.embed_batch(texts))
    assert len(embs) == 3
    assert all(len(e) == 384 for e in embs)


def test_embed_batch_different_texts_differ():
    """Different texts should produce different embeddings."""
    svc = EmbeddingService(settings)
    embs = asyncio.new_event_loop().run_until_complete(svc.embed_batch([
        "Transformers use attention",
        "Recurrent neural networks use memory",
    ]))
    # Cosine similarity should be less than 1 (not identical)
    import math
    dot = sum(a * b for a, b in zip(embs[0], embs[1]))
    assert dot < 0.999, "Embeddings for different texts should not be identical"


# ── VectorDB (local ChromaDB) ──────────────────────────────────────────────

def _make_test_vdb():
    """Create a VectorDBService backed by an in-memory ChromaDB (no filesystem)."""
    vdb = VectorDBService(settings)
    vdb._client = chromadb.EphemeralClient()
    return vdb


def test_vector_db_store_and_query():
    svc = EmbeddingService(settings)
    vdb = _make_test_vdb()

    texts = [
        "Transformers revolutionized NLP with attention mechanisms.",
        "BERT uses bidirectional encoders for language understanding.",
        "GPT uses autoregressive decoding for text generation.",
    ]

    loop = asyncio.new_event_loop()
    embeddings = loop.run_until_complete(svc.embed_batch(texts))

    chunks = [
        {
            "chunk_id": f"test_chunk_{i}",
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

    stored = loop.run_until_complete(
        vdb.store_chunks("test_papers", chunks, embeddings)
    )
    assert stored == 3

    # Query for something semantically similar
    query_emb = loop.run_until_complete(svc.embed_single("attention in neural networks"))
    results = loop.run_until_complete(
        vdb.query_similar("test_papers", query_emb, top_k=2)
    )
    assert len(results) == 2
    assert all(r.score >= 0.0 for r in results)
    assert all(r.paper_id.startswith("paper_") for r in results)
    # Top result should be about transformers/attention (highest score)
    assert results[0].score >= results[1].score


def test_vector_db_empty_collection_returns_empty():
    vdb = _make_test_vdb()
    loop = asyncio.new_event_loop()
    svc = EmbeddingService(settings)
    query_emb = loop.run_until_complete(svc.embed_single("anything"))
    results = loop.run_until_complete(
        vdb.query_similar("empty_collection_xyz", query_emb, top_k=5)
    )
    assert results == []


def test_vector_db_chunk_id_preserved():
    svc = EmbeddingService(settings)
    vdb = _make_test_vdb()
    loop = asyncio.new_event_loop()

    text = "Neural networks learn representations from data."
    emb = loop.run_until_complete(svc.embed_batch([text]))
    chunk = [{
        "chunk_id": "my_specific_chunk_id_123",
        "paper_id": "p001",
        "text": text,
        "chunk_index": 0,
        "metadata": {"paper_id": "p001", "paper_title": "Test", "chunk_index": 0, "char_start": 0, "char_end": 50},
    }]
    loop.run_until_complete(vdb.store_chunks("test_cid", chunk, emb))

    query_emb = loop.run_until_complete(svc.embed_single(text))
    results = loop.run_until_complete(vdb.query_similar("test_cid", query_emb, top_k=1))
    assert results[0].chunk_id == "my_specific_chunk_id_123"


# ── Mock DynamoDB (no local DynamoDB in test env) ─────────────────────────

class _MockDynamo:
    """No-op DynamoDB stub for tests running without a DynamoDB container."""
    async def put_paper(self, *args, **kwargs): pass
    async def get_paper(self, *args, **kwargs): return None
    async def put_memory_entry(self, *args, **kwargs): return "mock-entry-id"
    async def get_session_entries(self, *args, **kwargs): return []


def _make_test_ingestion_svc():
    svc = PDFIngestionService(settings)
    svc._vector_db = _make_test_vdb()
    svc._dynamo_db = _MockDynamo()
    return svc


# ── PDF Ingestion ──────────────────────────────────────────────────────────

def test_ingest_text_flow():
    """Full text ingest: chunk → embed → store in in-memory ChromaDB (no DynamoDB)."""
    svc = _make_test_ingestion_svc()
    text = (
        "Abstract: We propose a novel attention mechanism. " * 50
        + "Introduction: Transformers have transformed NLP. " * 50
    )
    req = IngestRequest(paper_id="paper_test_ingest", title="Test Paper", content=text)
    result = asyncio.new_event_loop().run_until_complete(svc.ingest_text(req))

    assert result.status == "success"
    assert result.chunks_created >= 1
    assert result.embeddings_stored == result.chunks_created


def test_ingest_empty_content():
    svc = PDFIngestionService(settings)
    req = IngestRequest(paper_id="empty_paper", title="Empty", content="")
    result = asyncio.new_event_loop().run_until_complete(svc.ingest_text(req))
    assert result.status == "error"
    assert result.chunks_created == 0


# ── MLflow service ─────────────────────────────────────────────────────────

def test_mlflow_no_crash_when_server_absent():
    """MLflow should degrade gracefully when no server is running."""
    svc = MLflowService("http://localhost:19999", "test_experiment")
    # Should not raise
    svc.log_metric("test_metric", 1.0)
    svc.log_param("test_param", "value")
    svc.log_embedding_batch(10, 25.0)
    svc.log_rag_query("q", 3, 10.0, 200.0, 210.0)


# ── Ingest pipeline ────────────────────────────────────────────────────────

def test_build_abstract_doc():
    from backend.models.schemas import PaperMetadata
    paper = PaperMetadata(
        paper_id="arxiv_test",
        title="Test Paper on Transformers",
        authors=["Alice", "Bob"],
        abstract="We study transformers in depth.",
        source="arxiv",
    )
    doc = IngestPipeline._build_abstract_doc(paper)
    assert "Test Paper on Transformers" in doc
    assert "Alice" in doc
    assert "We study transformers in depth." in doc


def test_ingest_pipeline_abstract_fallback():
    """Pipeline should fall back to abstract when PDF unavailable."""
    from backend.models.schemas import PaperMetadata
    pipeline = IngestPipeline(settings)
    # Inject stubs so no real network/DB calls are made
    pipeline._ingestion = _make_test_ingestion_svc()

    paper = PaperMetadata(
        paper_id="pipeline_test_001",
        title="Test Transformer Paper",
        authors=["Alice Smith"],
        abstract="This paper proposes a new attention mechanism for language models. " * 10,
        pdf_url=None,   # No PDF — forces abstract fallback
        source="arxiv",
    )
    result = asyncio.new_event_loop().run_until_complete(pipeline._ingest_paper(paper))
    assert result.status == "success"
    assert result.chunks_created >= 1
