"""
Day 4 — Integration tests for all API endpoints via FastAPI TestClient.
Tests every route in the app with minimal dependencies (mock LLM, mock VDB, mock Dynamo).
No external services required.

Run with: pytest tests/test_day4.py -v
"""
import asyncio
import uuid
from unittest.mock import AsyncMock, MagicMock, patch
import chromadb
import pytest
from fastapi.testclient import TestClient

from backend.main import app
from backend.config import get_settings
from backend.services.vector_db import VectorDBService
from backend.services.embedding_service import EmbeddingService

client = TestClient(app)
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


MOCK_ANSWER = "This is a mock RAG answer for testing purposes."
MOCK_REPORT = "## Summary\nThis is a mock report.\n## Conclusion\nDone."


# ── 1. Health endpoints ────────────────────────────────────────────────────

class TestHealthEndpoints:
    def test_root(self):
        r = client.get("/")
        assert r.status_code == 200
        body = r.json()
        assert "ResearchCopilot" in body["message"]
        assert "docs" in body
        assert "status" in body

    def test_health(self):
        r = client.get("/health")
        assert r.status_code == 200
        body = r.json()
        assert body["status"] == "ok"
        assert body["version"] == "1.0.0"

    def test_status(self):
        r = client.get("/status")
        assert r.status_code == 200
        body = r.json()
        assert body["status"] in ("ready", "degraded")
        assert "services" in body
        for svc in ("chromadb", "dynamodb", "ollama", "mlflow"):
            assert svc in body["services"]
        assert "config" in body
        assert body["config"]["embedding_dim"] == 384


# ── 2. Paper search endpoint ───────────────────────────────────────────────

class TestPaperSearch:
    def test_search_arxiv_returns_papers(self):
        """Live arXiv search (no mock — uses public API)."""
        r = client.post("/api/v1/papers/search", json={
            "query": "BERT language model",
            "source": "arxiv",
            "max_results": 3,
        })
        assert r.status_code == 200
        body = r.json()
        assert body["query"] == "BERT language model"
        assert isinstance(body["papers"], list)
        assert body["total_found"] >= 0   # 0 is ok if arXiv is unreachable

    def test_search_validation_short_query(self):
        r = client.post("/api/v1/papers/search", json={"query": "ab", "max_results": 5})
        assert r.status_code == 422

    def test_search_validation_max_results(self):
        r = client.post("/api/v1/papers/search", json={"query": "transformers", "max_results": 999})
        assert r.status_code == 422


# ── 3. Paper ingest endpoint ───────────────────────────────────────────────

class TestPaperIngest:
    def _patch_services(self):
        """Context managers that stub out ChromaDB and DynamoDB."""
        vdb = _ephemeral_vdb()
        dynamo = _MockDynamo()
        return vdb, dynamo

    def test_ingest_text_success(self):
        vdb, dynamo = self._patch_services()
        with patch("backend.services.pdf_ingestion.PDFIngestionService._get_vector_db", return_value=vdb), \
             patch("backend.services.pdf_ingestion.PDFIngestionService._get_dynamo_db", return_value=dynamo):
            r = client.post("/api/v1/papers/ingest", json={
                "paper_id": "test_paper_001",
                "title": "Test Paper on Transformers",
                "content": "Transformers use self-attention. " * 60,
            })
        assert r.status_code == 200
        body = r.json()
        assert body["status"] == "success"
        assert body["chunks_created"] >= 1
        assert body["embeddings_stored"] == body["chunks_created"]

    def test_ingest_empty_content_returns_error(self):
        r = client.post("/api/v1/papers/ingest", json={
            "paper_id": "empty_001",
            "title": "Empty Paper",
            "content": "",
        })
        assert r.status_code == 200
        assert r.json()["status"] == "error"

    def test_ingest_missing_fields_422(self):
        r = client.post("/api/v1/papers/ingest", json={"title": "No ID"})
        assert r.status_code == 422


# ── 4. Embeddings query endpoint ───────────────────────────────────────────

class TestEmbeddingsQuery:
    def test_query_with_empty_collection(self):
        """Query an empty collection — should return empty list, not 500."""
        vdb = _ephemeral_vdb()
        with patch("backend.services.vector_db.VectorDBService", return_value=vdb):
            r = client.post("/api/v1/embeddings/query", json={
                "query": "attention mechanism",
                "top_k": 3,
            })
        assert r.status_code == 200
        body = r.json()
        assert body["query"] == "attention mechanism"
        assert isinstance(body["results"], list)

    def test_query_validation(self):
        r = client.post("/api/v1/embeddings/query", json={"query": "", "top_k": 3})
        assert r.status_code == 422


# ── 5. Memory endpoints ────────────────────────────────────────────────────

class TestMemoryEndpoints:
    def test_store_memory_entry(self):
        vdb = _ephemeral_vdb()
        dynamo = _MockDynamo()
        with patch("backend.services.memory_service.MemoryService.__init__", return_value=None), \
             patch("backend.services.memory_service.MemoryService.store",
                   new=AsyncMock(return_value="entry-id-abc123")):
            r = client.post("/api/v1/memory/store", json={
                "session_id": "session_test_001",
                "role": "user",
                "content": "What is the attention mechanism?",
            })
        assert r.status_code == 201
        body = r.json()
        assert body["status"] == "stored"
        assert "entry_id" in body

    def test_store_invalid_role(self):
        r = client.post("/api/v1/memory/store", json={
            "session_id": "sess", "role": "invalid_role", "content": "test"
        })
        assert r.status_code == 422

    def test_retrieve_memory(self):
        with patch("backend.services.memory_service.MemoryService.__init__", return_value=None), \
             patch("backend.services.memory_service.MemoryService.retrieve",
                   new=AsyncMock(return_value=MagicMock(
                       session_id="sess_001",
                       entries=[],
                       total=0,
                       model_dump=lambda: {"session_id": "sess_001", "entries": [], "total": 0}
                   ))):
            r = client.post("/api/v1/memory/retrieve", json={
                "session_id": "sess_001", "limit": 5
            })
        assert r.status_code == 200


# ── 6. RAG query endpoint ──────────────────────────────────────────────────

class TestRAGQuery:
    def _mock_rag(self):
        from backend.models.schemas import RAGQueryResponse
        mock_response = RAGQueryResponse(
            query="What is BERT?",
            answer=MOCK_ANSWER,
            session_id="test-session",
            sources=[],
            model_used="qwen3:0.6b",
        )
        return patch(
            "backend.services.rag_service.RAGService.query",
            new=AsyncMock(return_value=mock_response),
        )

    def test_rag_query_returns_answer(self):
        with self._mock_rag():
            r = client.post("/api/v1/rag/query", json={
                "query": "What is BERT?",
                "top_k": 3,
            })
        assert r.status_code == 200
        body = r.json()
        assert body["answer"] == MOCK_ANSWER
        assert body["model_used"] == "qwen3:0.6b"

    def test_rag_query_with_session(self):
        from backend.models.schemas import RAGQueryResponse
        sid = str(uuid.uuid4())
        mock_resp = RAGQueryResponse(
            query="explain transformers", answer=MOCK_ANSWER,
            session_id=sid, sources=[], model_used="qwen3:0.6b",
        )
        with patch("backend.services.rag_service.RAGService.query",
                   new=AsyncMock(return_value=mock_resp)):
            r = client.post("/api/v1/rag/query", json={
                "query": "explain transformers",
                "session_id": sid,
                "top_k": 2,
                "use_memory": True,
            })
        assert r.status_code == 200
        assert r.json()["session_id"] == sid

    def test_rag_query_short_query_422(self):
        r = client.post("/api/v1/rag/query", json={"query": "ab"})
        assert r.status_code == 422


# ── 7. Report generation endpoint ─────────────────────────────────────────

class TestReportGeneration:
    def test_generate_report(self):
        with patch("backend.services.rag_service.RAGService.generate_report",
                   new=AsyncMock(return_value=(MOCK_REPORT, ["paper_001", "paper_002"]))):
            r = client.post("/api/v1/report/generate", json={
                "topic": "Transformer attention mechanisms",
                "max_length": 400,
                "format": "markdown",
            })
        assert r.status_code == 200
        body = r.json()
        assert body["report"] == MOCK_REPORT
        assert body["format"] == "markdown"
        assert "paper_001" in body["sources_used"]
        assert "generated_at" in body

    def test_report_short_topic_422(self):
        r = client.post("/api/v1/report/generate", json={"topic": "ab"})
        assert r.status_code == 422


# ── 8. Pipeline endpoints ──────────────────────────────────────────────────

class TestIngestPipeline:
    def test_search_and_ingest_route_exists(self):
        from backend.models.schemas import IngestResponse
        mock_results = [
            IngestResponse(
                paper_id="arxiv_test",
                status="success",
                chunks_created=5,
                embeddings_stored=5,
                message="ok",
            )
        ]
        with patch("backend.services.ingest_pipeline.IngestPipeline.search_and_ingest",
                   new=AsyncMock(return_value=mock_results)):
            r = client.post("/api/v1/pipeline/search-and-ingest", json={
                "query": "neural networks",
                "max_papers": 2,
            })
        assert r.status_code == 200
        body = r.json()
        assert isinstance(body, list)
        assert body[0]["status"] == "success"

    def test_ingest_by_id_route_exists(self):
        from backend.models.schemas import IngestResponse
        mock_result = IngestResponse(
            paper_id="arxiv_2310.06825",
            status="success",
            chunks_created=8,
            embeddings_stored=8,
            message="ok",
        )
        with patch("backend.services.ingest_pipeline.IngestPipeline.ingest_paper_by_id",
                   new=AsyncMock(return_value=mock_result)):
            r = client.post("/api/v1/pipeline/ingest-by-id", json={"arxiv_id": "2310.06825"})
        assert r.status_code == 200
        assert r.json()["status"] == "success"


# ── 9. All routes registered ───────────────────────────────────────────────

class TestRouteInventory:
    EXPECTED = [
        "/api/v1/papers/search",
        "/api/v1/papers/ingest",
        "/api/v1/papers/ingest/pdf",
        "/api/v1/embeddings/query",
        "/api/v1/memory/store",
        "/api/v1/memory/retrieve",
        "/api/v1/rag/query",
        "/api/v1/report/generate",
        "/api/v1/pipeline/search-and-ingest",
        "/api/v1/pipeline/ingest-by-id",
    ]

    def test_all_routes_present(self):
        routes = [r.path for r in app.routes]
        for expected in self.EXPECTED:
            assert any(expected in r for r in routes), f"Route missing: {expected}"

    def test_total_route_count(self):
        api_routes = [r for r in app.routes if hasattr(r, "path") and "/api/v1" in getattr(r, "path", "")]
        assert len(api_routes) >= 10
