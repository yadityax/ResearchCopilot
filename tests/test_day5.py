"""
Day 5 tests — regression tests for the memory ordering and session isolation fixes.
All external services are stubbed (ephemeral ChromaDB, mock DynamoDB).

Run with: pytest tests/test_day5.py -v
"""
import asyncio
import time
import uuid

import chromadb

from backend.config import get_settings
from backend.models.schemas import MemoryStoreRequest, MemoryRetrieveRequest
from backend.services.dynamo_db import DynamoDBService
from backend.services.memory_service import MemoryService
from backend.services.pdf_ingestion import PDFIngestionService
from backend.services.vector_db import VectorDBService

settings = get_settings()
loop = asyncio.new_event_loop()


class _MockDynamo:
    async def put_memory_entry(self, *a, **kw):
        from datetime import datetime, timezone
        return DynamoDBService.make_entry_id(datetime.now(timezone.utc).isoformat())

    async def get_session_entries(self, *a, **kw):
        return []


def _memory_svc():
    svc = MemoryService(settings)
    svc._dynamo = _MockDynamo()
    svc._vector_db = VectorDBService(settings)
    svc._vector_db._client = chromadb.EphemeralClient()
    return svc


def _store(svc, session_id, content, role="user"):
    loop.run_until_complete(svc.store(MemoryStoreRequest(session_id=session_id, role=role, content=content)))


# ── Sort key ordering ──────────────────────────────────────────────────────

def test_entry_ids_sort_in_time_order():
    """entry_id is the DynamoDB sort key, so lexicographic order must equal write order."""
    from datetime import datetime, timezone
    ids = []
    for _ in range(5):
        ids.append(DynamoDBService.make_entry_id(datetime.now(timezone.utc).isoformat()))
        time.sleep(0.002)
    assert ids == sorted(ids)


def test_entry_ids_unique_for_same_timestamp():
    ts = "2026-10-30T10:00:00+00:00"
    assert DynamoDBService.make_entry_id(ts) != DynamoDBService.make_entry_id(ts)


# ── Paper lookup ───────────────────────────────────────────────────────────

_PAPERS = [
    {"paper_id": "attn", "title": "Attention Is All You Need"},
    {"paper_id": "bert", "title": "BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding"},
    {"paper_id": "sbert", "title": "Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks"},
    {"paper_id": "ddpm", "title": "Denoising Diffusion Probabilistic Models"},
]


def _ids(matched):
    return [m["paper_id"] for m in matched]


def test_match_papers_by_title_when_acronym_not_in_title():
    matched = DynamoDBService.match_papers(_PAPERS, "DDPM", "Denoising Diffusion Probabilistic Models")
    assert _ids(matched) == ["ddpm"]


def test_match_papers_title_beats_ambiguous_name():
    """'Transformer' as a substring hits the BERT title; the official title must win."""
    matched = DynamoDBService.match_papers(_PAPERS, "Transformer", "Attention Is All You Need")
    assert _ids(matched) == ["attn"]


def test_match_papers_title_tolerates_case_and_punctuation():
    title = "bert - pre-training of deep bidirectional transformers for language understanding"
    matched = DynamoDBService.match_papers(_PAPERS, "BERT", title)
    assert _ids(matched) == ["bert"]


def test_match_papers_name_fallback_is_whole_word():
    assert _ids(DynamoDBService.match_papers(_PAPERS, "Transformer", None)) == []
    assert set(_ids(DynamoDBService.match_papers(_PAPERS, "BERT", None))) == {"bert", "sbert"}


# ── Session isolation ──────────────────────────────────────────────────────

def test_semantic_retrieve_only_returns_own_session():
    svc = _memory_svc()
    mine, other = f"mine_{uuid.uuid4().hex}", f"other_{uuid.uuid4().hex}"
    _store(svc, other, "LoRA freezes pretrained weights and trains low-rank matrices A and B.")
    _store(svc, mine, "Diffusion models learn to reverse a gradual noising process.")

    result = loop.run_until_complete(svc.retrieve(
        MemoryRetrieveRequest(session_id=mine, limit=5, query="LoRA low-rank matrices")
    ))
    assert all("LoRA" not in e.content for e in result.entries)


def test_forget_is_not_crowded_out_by_other_sessions():
    """Many similar entries in another session must not stop this session forgetting its own."""
    svc = _memory_svc()
    mine, other = f"mine_{uuid.uuid4().hex}", f"other_{uuid.uuid4().hex}"
    for i in range(10):
        _store(svc, other, f"BERT uses masked language modeling, example {i}.")
    _store(svc, mine, "BERT uses masked language modeling during pretraining.")

    result = loop.run_until_complete(svc.forget_topic(mine, "BERT masked language modeling", top_k=3))
    assert result["deleted"] == 1

    # The other session's memories are untouched
    remaining = loop.run_until_complete(svc.retrieve(
        MemoryRetrieveRequest(session_id=other, limit=20, query="BERT masked language modeling")
    ))
    assert len(remaining.entries) == 10


# ── Embedding determinism ──────────────────────────────────────────────────

def test_embeddings_identical_across_service_instances():
    """
    Each pod / restart builds its own EmbeddingService. Vectors stored by one instance must match
    queries embedded by another, so no randomly initialised layer may sit in the embedding path.
    """
    from backend.services.embedding_service import EmbeddingService
    text = "LoRA freezes the pretrained weights and trains low-rank matrices."
    a = loop.run_until_complete(EmbeddingService(settings).embed_single(text))
    b = loop.run_until_complete(EmbeddingService(settings).embed_single(text))
    assert max(abs(x - y) for x, y in zip(a, b)) < 1e-5


# ── Chunk size vs embedder input limit ─────────────────────────────────────

def test_chunks_fit_embedder_input_limit():
    """
    all-MiniLM-L6-v2 truncates input at 256 word pieces; anything past that is never embedded.
    Default chunks must fit, so the whole chunk contributes to its vector.
    """
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(settings.embedding_model)

    paragraph = (
        "The attention function maps a query and a set of key-value pairs to an output. "
        "The output is computed as a weighted sum of the values, where the weight assigned to each "
        "value is computed by a compatibility function of the query with the corresponding key. "
    )
    ingest = PDFIngestionService(settings)
    chunks = ingest._split_into_chunks(paragraph * 40, "p_limit")
    assert len(chunks) > 1
    for c in chunks:
        assert len(tok.encode(c["text"])) <= 256
