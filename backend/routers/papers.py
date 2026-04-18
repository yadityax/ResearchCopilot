from fastapi import APIRouter, UploadFile, File, Form, Depends, HTTPException, status
from loguru import logger
import tempfile, os

from backend.models.schemas import (
    PaperSearchRequest, PaperSearchResponse,
    IngestRequest, IngestResponse,
)
from backend.services.paper_discovery import PaperDiscoveryService
from backend.services.pdf_ingestion import PDFIngestionService
from backend.services.llm_service import LLMService
from backend.config import get_settings, Settings

router = APIRouter(prefix="/papers", tags=["Papers"])


def get_discovery_service(settings: Settings = Depends(get_settings)) -> PaperDiscoveryService:
    return PaperDiscoveryService(settings)


def get_ingestion_service(settings: Settings = Depends(get_settings)) -> PDFIngestionService:
    return PDFIngestionService(settings)


# ── Endpoint 1: /papers/search ─────────────────────────────────────────────
@router.post("/search", response_model=PaperSearchResponse)
async def search_papers(
    request: PaperSearchRequest,
    svc: PaperDiscoveryService = Depends(get_discovery_service),
    settings: Settings = Depends(get_settings),
):
    """
    Search for research papers across arXiv and/or Semantic Scholar.
    Rewrites the user's raw query with LLM before searching for better results.
    """
    logger.info(f"[search_papers] raw_query='{request.query}' source={request.source}")
    try:
        # Rewrite query for better arXiv keyword matching
        llm = LLMService(settings)
        rewritten = await llm.rewrite_query(request.query)
        clean_query = rewritten["search_query"]  # broad arXiv keywords
        logger.info(f"[search_papers] rewritten_query='{clean_query}'")

        rewritten_request = PaperSearchRequest(
            query=clean_query,
            source=request.source,
            max_results=request.max_results,
            year_from=request.year_from,
            year_to=request.year_to,
        )
        results = await svc.search(rewritten_request)
        return results
    except Exception as e:
        logger.error(f"[search_papers] error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ── Endpoint 2: /papers/ingest (text content) ──────────────────────────────
@router.post("/ingest", response_model=IngestResponse)
async def ingest_paper_text(
    request: IngestRequest,
    svc: PDFIngestionService = Depends(get_ingestion_service),
):
    """
    Ingest a paper from raw text content. Chunks the text and stores
    embeddings in ChromaDB with metadata in DynamoDB.
    """
    logger.info(f"[ingest_paper_text] paper_id={request.paper_id}")
    try:
        result = await svc.ingest_text(request)
        return result
    except Exception as e:
        logger.error(f"[ingest_paper_text] error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ── Endpoint 2b: /papers/ingest/pdf (file upload) ─────────────────────────
@router.post("/ingest/pdf", response_model=IngestResponse)
async def ingest_paper_pdf(
    paper_id: str = Form(...),
    title: str = Form(...),
    file: UploadFile = File(...),
    svc: PDFIngestionService = Depends(get_ingestion_service),
):
    """
    Upload a PDF file, extract text, chunk it, and store embeddings.
    """
    if not file.filename.endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are accepted.")

    logger.info(f"[ingest_paper_pdf] paper_id={paper_id} filename={file.filename}")

    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        tmp.write(await file.read())
        tmp_path = tmp.name

    try:
        result = await svc.ingest_pdf(paper_id=paper_id, title=title, pdf_path=tmp_path)
        return result
    except Exception as e:
        logger.error(f"[ingest_paper_pdf] error: {e}")
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        os.unlink(tmp_path)


# ── Endpoint: /papers/author ───────────────────────────────────────────────
@router.get("/author")
async def search_by_author(
    name: str,
    max_results: int = 10,
    svc: PaperDiscoveryService = Depends(get_discovery_service),
):
    """
    Search Semantic Scholar for an author by name.
    Returns author profile (h-index, citation count) + their top papers.
    """
    logger.info(f"[search_by_author] name='{name}'")
    try:
        result = await svc.search_by_author(name, max_results)
        if not result["author"]:
            raise HTTPException(status_code=404, detail=f"Author '{name}' not found on Semantic Scholar.")
        return result
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"[search_by_author] error: {e}")
        raise HTTPException(status_code=500, detail=str(e))
