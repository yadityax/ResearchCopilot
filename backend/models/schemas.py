from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any
from datetime import datetime
from enum import Enum


# ── Enums ──────────────────────────────────────────────────────────────────

class PaperSource(str, Enum):
    ARXIV = "arxiv"
    SEMANTIC_SCHOLAR = "semantic_scholar"
    BOTH = "both"


class ReportFormat(str, Enum):
    MARKDOWN = "markdown"
    PLAIN = "plain"


# ── Paper Schemas ──────────────────────────────────────────────────────────

class PaperSearchRequest(BaseModel):
    query: str = Field(..., min_length=3, description="Search query for papers")
    source: PaperSource = PaperSource.BOTH
    max_results: int = Field(default=10, ge=1, le=50)
    year_from: Optional[int] = None
    year_to: Optional[int] = None


class PaperMetadata(BaseModel):
    paper_id: str
    title: str
    authors: List[str]
    abstract: str
    published_date: Optional[str] = None
    url: Optional[str] = None
    pdf_url: Optional[str] = None
    source: str
    categories: List[str] = []
    citation_count: Optional[int] = None


class PaperSearchResponse(BaseModel):
    query: str
    total_found: int
    papers: List[PaperMetadata]


# ── Ingest Schemas ─────────────────────────────────────────────────────────

class IngestRequest(BaseModel):
    paper_id: str = Field(..., description="Unique identifier for the paper")
    title: str
    content: Optional[str] = Field(None, description="Raw text if already extracted")


class IngestResponse(BaseModel):
    paper_id: str
    status: str
    chunks_created: int
    embeddings_stored: int
    message: str


# ── Embedding / Query Schemas ──────────────────────────────────────────────

class EmbeddingQueryRequest(BaseModel):
    query: str = Field(..., min_length=1)
    top_k: int = Field(default=5, ge=1, le=20)
    collection: str = "research_papers"
    filter_paper_ids: Optional[List[str]] = None


class RetrievedChunk(BaseModel):
    chunk_id: str
    paper_id: str
    paper_title: str
    text: str
    score: float
    chunk_index: int
    metadata: Dict[str, Any] = Field(default_factory=dict)


class EmbeddingQueryResponse(BaseModel):
    query: str
    results: List[RetrievedChunk]
    total_results: int


# ── Memory Schemas ─────────────────────────────────────────────────────────

class MemoryStoreRequest(BaseModel):
    session_id: str = Field(..., description="Unique session identifier")
    role: str = Field(..., pattern="^(user|assistant)$")
    content: str
    metadata: Optional[Dict[str, Any]] = {}


class MemoryRetrieveRequest(BaseModel):
    session_id: str
    limit: int = Field(default=10, ge=1, le=50)
    query: Optional[str] = None  # if set, returns semantically relevant turns


class MemoryEntry(BaseModel):
    entry_id: str
    session_id: str
    role: str
    content: str
    timestamp: str
    metadata: Dict[str, Any] = {}


class MemoryRetrieveResponse(BaseModel):
    session_id: str
    entries: List[MemoryEntry]
    total: int


class ForgetRequest(BaseModel):
    session_id: str = Field(..., description="Session whose memory to search")
    topic: str = Field(..., min_length=2, description="Topic to forget, e.g. 'GANs'")
    top_k: int = Field(default=20, ge=1, le=100)


# ── Chat Session Schemas ────────────────────────────────────────────────────

class ChatSessionCreate(BaseModel):
    user_id: str
    session_id: str
    session_name: Optional[str] = "New Chat"


class ChatSessionRename(BaseModel):
    session_name: str


class ChatSessionInfo(BaseModel):
    session_id: str
    session_name: str
    created_at: str
    last_message_at: Optional[str] = None
    message_count: int = 0


class UserSessionsResponse(BaseModel):
    user_id: str
    sessions: List[ChatSessionInfo]


# ── RAG Schemas ────────────────────────────────────────────────────────────

class RAGQueryRequest(BaseModel):
    query: str = Field(..., min_length=3)
    session_id: Optional[str] = None
    user_id: Optional[str] = None
    top_k: int = Field(default=5, ge=1, le=20)
    paper_ids: Optional[List[str]] = None  # restrict context to specific papers
    use_memory: bool = True


class RAGQueryResponse(BaseModel):
    query: str
    answer: str
    session_id: Optional[str]
    sources: List[RetrievedChunk]
    model_used: str
    tokens_used: Optional[int] = None


# ── Report Schemas ─────────────────────────────────────────────────────────

class ReportRequest(BaseModel):
    topic: str = Field(..., min_length=3, description="Research topic for the report")
    session_id: Optional[str] = None
    paper_ids: Optional[List[str]] = None
    max_length: int = Field(default=2000, ge=100, le=50000)
    format: ReportFormat = ReportFormat.MARKDOWN


class ReportResponse(BaseModel):
    topic: str
    report: str
    format: str
    sources_used: List[str]
    generated_at: str


# ── Health ─────────────────────────────────────────────────────────────────

class HealthResponse(BaseModel):
    status: str
    services: Dict[str, str]
    version: str = "1.0.0"
