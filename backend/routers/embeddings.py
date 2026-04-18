from fastapi import APIRouter, Depends, HTTPException
from loguru import logger

from backend.models.schemas import EmbeddingQueryRequest, EmbeddingQueryResponse
from backend.services.embedding_service import EmbeddingService
from backend.config import get_settings, Settings

router = APIRouter(prefix="/embeddings", tags=["Embeddings"])


def get_embedding_service(settings: Settings = Depends(get_settings)) -> EmbeddingService:
    return EmbeddingService(settings)


# ── Endpoint 3: /embeddings/query ─────────────────────────────────────────
@router.post("/query", response_model=EmbeddingQueryResponse)
async def query_embeddings(
    request: EmbeddingQueryRequest,
    svc: EmbeddingService = Depends(get_embedding_service),
):
    """
    Embed the query using sentence-transformers → MLP projection →
    retrieve top-k similar chunks from ChromaDB.
    """
    logger.info(f"[query_embeddings] query='{request.query}' top_k={request.top_k}")
    try:
        return await svc.query(request)
    except Exception as e:
        logger.error(f"[query_embeddings] error: {e}")
        raise HTTPException(status_code=500, detail=str(e))
