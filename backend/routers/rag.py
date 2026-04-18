from fastapi import APIRouter, Depends, HTTPException
from loguru import logger

from backend.models.schemas import RAGQueryRequest, RAGQueryResponse
from backend.services.rag_service import RAGService
from backend.config import get_settings, Settings

router = APIRouter(prefix="/rag", tags=["RAG Pipeline"])


def get_rag_service(settings: Settings = Depends(get_settings)) -> RAGService:
    return RAGService(settings)


# ── Endpoint 6: /rag/query ─────────────────────────────────────────────────
@router.post("/query", response_model=RAGQueryResponse)
async def rag_query(
    request: RAGQueryRequest,
    svc: RAGService = Depends(get_rag_service),
):
    """
    Full RAG pipeline:
      1. Embed query (sentence-transformers → MLP projection)
      2. Retrieve top-k relevant chunks from ChromaDB
      3. Optionally load session memory for context
      4. Build prompt and call Llama 3 via Ollama
      5. Return answer + source citations
    """
    logger.info(f"[rag_query] query='{request.query}' session_id={request.session_id}")
    try:
        response = await svc.query(request)
        return response
    except Exception as e:
        logger.error(f"[rag_query] error: {e}")
        raise HTTPException(status_code=500, detail=str(e))
