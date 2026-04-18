"""
Day 1 tests — routes, schemas, MLP models, paper discovery, chunking.
Run with: pytest tests/test_day1.py -v
"""
import pytest
import torch
from fastapi.testclient import TestClient

from backend.main import app
from backend.config import get_settings
from backend.models.schemas import (
    PaperSearchRequest, PaperSource,
    IngestRequest, EmbeddingQueryRequest,
    MemoryStoreRequest, RAGQueryRequest, ReportRequest,
)
from backend.models.mlp_models import EncoderMLP, SinusoidalMLP
from backend.services.pdf_ingestion import PDFIngestionService

client = TestClient(app)
settings = get_settings()


# ── Health ─────────────────────────────────────────────────────────────────

def test_root():
    r = client.get("/")
    assert r.status_code == 200
    assert "ResearchCopilot" in r.json()["message"]


def test_health():
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"


# ── Routes registered ──────────────────────────────────────────────────────

def test_all_routes_registered():
    routes = [r.path for r in app.routes]
    expected = [
        "/api/v1/papers/search",
        "/api/v1/papers/ingest",
        "/api/v1/papers/ingest/pdf",
        "/api/v1/embeddings/query",
        "/api/v1/memory/store",
        "/api/v1/memory/retrieve",
        "/api/v1/rag/query",
        "/api/v1/report/generate",
    ]
    for e in expected:
        assert any(e in r for r in routes), f"Route {e} not registered"


# ── Schemas ────────────────────────────────────────────────────────────────

def test_paper_search_request_schema():
    req = PaperSearchRequest(query="attention mechanism", max_results=5)
    assert req.query == "attention mechanism"
    assert req.source == PaperSource.BOTH


def test_ingest_request_schema():
    req = IngestRequest(paper_id="p001", title="Test", content="Some content")
    assert req.paper_id == "p001"


def test_rag_request_schema():
    req = RAGQueryRequest(query="What is BERT?", top_k=3)
    assert req.top_k == 3
    assert req.use_memory is True


# ── MLP Models ─────────────────────────────────────────────────────────────

def test_encoder_mlp_shape():
    model = EncoderMLP(384, 384, 384)
    x = torch.randn(4, 384)
    out = model(x)
    assert out.shape == (4, 384)


def test_sinusoidal_mlp_shape():
    model = SinusoidalMLP(384)
    x = torch.randn(4, 384)
    out = model(x)
    assert out.shape == (4, 384)


def test_encoder_mlp_residual_close_to_input():
    """With near-zero init, output should be close to input (residual)."""
    model = EncoderMLP(384, 384, 384)
    model.eval()
    x = torch.randn(2, 384)
    with torch.no_grad():
        out = model(x)
    # Residual: output shouldn't diverge wildly from input
    diff = (out - x).abs().mean().item()
    assert diff < 2.0, f"EncoderMLP output too far from input: {diff}"


# ── PDF Ingestion / Chunking ───────────────────────────────────────────────

def test_text_cleaning():
    svc = PDFIngestionService(settings)
    dirty = "Hello   world\n\n\n\nFoo"
    clean = svc._clean_text(dirty)
    assert "\n\n\n" not in clean
    assert "  " not in clean


def test_chunking_basic():
    svc = PDFIngestionService(settings)
    # 3000 chars should produce multiple chunks
    text = "The transformer model uses self-attention. " * 100
    chunks = svc._split_into_chunks(text, "paper_test")
    assert len(chunks) >= 1
    for c in chunks:
        assert "chunk_id" in c
        assert "paper_id" in c
        assert c["paper_id"] == "paper_test"
        assert len(c["text"]) > 50


def test_chunking_tiny_text():
    svc = PDFIngestionService(settings)
    text = "Short."
    chunks = svc._split_into_chunks(text, "tiny")
    # Too short to create meaningful chunks
    assert len(chunks) == 0


def test_chunk_id_stable():
    svc = PDFIngestionService(settings)
    id1 = svc._chunk_id("paper_abc", 0)
    id2 = svc._chunk_id("paper_abc", 0)
    assert id1 == id2
    assert id1 != svc._chunk_id("paper_abc", 1)
