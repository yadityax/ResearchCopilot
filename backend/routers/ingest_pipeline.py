from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field
from typing import Optional, List
from loguru import logger

from backend.services.ingest_pipeline import IngestPipeline
from backend.models.schemas import IngestResponse, PaperSource
from backend.config import get_settings, Settings

router = APIRouter(prefix="/pipeline", tags=["Ingest Pipeline"])


class PipelineIngestRequest(BaseModel):
    query: str = Field(..., min_length=3)
    max_papers: int = Field(default=5, ge=1, le=20)
    source: PaperSource = PaperSource.ARXIV
    year_from: Optional[int] = None


class PipelineIngestByIdRequest(BaseModel):
    arxiv_id: str = Field(..., description="arXiv paper ID e.g. '2310.06825'")


def get_pipeline(settings: Settings = Depends(get_settings)) -> IngestPipeline:
    return IngestPipeline(settings)


@router.post("/search-and-ingest", response_model=List[IngestResponse])
async def search_and_ingest(
    request: PipelineIngestRequest,
    pipeline: IngestPipeline = Depends(get_pipeline),
):
    """Search arXiv, fetch paper text, embed and store — in one call."""
    logger.info(f"[pipeline] search_and_ingest query='{request.query}'")
    try:
        return await pipeline.search_and_ingest(
            query=request.query,
            max_papers=request.max_papers,
            source=request.source,
            year_from=request.year_from,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/ingest-by-id", response_model=IngestResponse)
async def ingest_by_arxiv_id(
    request: PipelineIngestByIdRequest,
    pipeline: IngestPipeline = Depends(get_pipeline),
):
    """Ingest a specific arXiv paper by its ID."""
    try:
        return await pipeline.ingest_paper_by_id(request.arxiv_id)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
