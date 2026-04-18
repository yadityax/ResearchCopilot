from fastapi import APIRouter, Depends, HTTPException
from loguru import logger
from datetime import datetime, timezone

from backend.models.schemas import ReportRequest, ReportResponse
from backend.services.rag_service import RAGService
from backend.config import get_settings, Settings

router = APIRouter(prefix="/report", tags=["Research Reports"])


def get_rag_service(settings: Settings = Depends(get_settings)) -> RAGService:
    return RAGService(settings)


# ── Endpoint 7: /report/generate ──────────────────────────────────────────
@router.post("/generate", response_model=ReportResponse)
async def generate_report(
    request: ReportRequest,
    svc: RAGService = Depends(get_rag_service),
):
    """
    Generate a structured research report on a topic.
    Uses RAG to ground the report in ingested papers.
    Returns markdown-formatted report with citations.
    """
    logger.info(f"[generate_report] topic='{request.topic}'")
    try:
        report_text, sources = await svc.generate_report(request)
        return ReportResponse(
            topic=request.topic,
            report=report_text,
            format=request.format.value,
            sources_used=sources,
            generated_at=datetime.now(timezone.utc).isoformat(),
        )
    except Exception as e:
        logger.error(f"[generate_report] error: {e}")
        raise HTTPException(status_code=500, detail=str(e))
