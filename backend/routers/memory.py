from fastapi import APIRouter, Depends, HTTPException
from loguru import logger

from backend.models.schemas import (
    MemoryStoreRequest, MemoryRetrieveRequest, MemoryRetrieveResponse,
    ChatSessionCreate, ChatSessionRename, ChatSessionInfo, UserSessionsResponse, ForgetRequest,
)
from backend.services.memory_service import MemoryService
from backend.services.dynamo_db import DynamoDBService
from backend.config import get_settings, Settings

router = APIRouter(prefix="/memory", tags=["Adaptive Memory"])


def get_memory_service(settings: Settings = Depends(get_settings)) -> MemoryService:
    return MemoryService(settings)

def get_dynamo(settings: Settings = Depends(get_settings)) -> DynamoDBService:
    return DynamoDBService(settings)


# ── Endpoint 4: /memory/store ──────────────────────────────────────────────
@router.post("/store", status_code=201)
async def store_memory(
    request: MemoryStoreRequest,
    svc: MemoryService = Depends(get_memory_service),
):
    """
    Persist a conversation turn (user or assistant) to the adaptive memory store.
    Stores both in DynamoDB (full history) and ChromaDB (semantic search).
    """
    logger.info(f"[store_memory] session_id={request.session_id} role={request.role}")
    try:
        entry_id = await svc.store(request)
        return {"status": "stored", "entry_id": entry_id}
    except Exception as e:
        logger.error(f"[store_memory] error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ── Endpoint 5: /memory/retrieve ──────────────────────────────────────────
@router.post("/retrieve", response_model=MemoryRetrieveResponse)
async def retrieve_memory(
    request: MemoryRetrieveRequest,
    svc: MemoryService = Depends(get_memory_service),
):
    """
    Retrieve conversation history for a session.
    If `query` is provided, returns semantically relevant turns from ChromaDB.
    Otherwise returns the last N turns from DynamoDB.
    """
    logger.info(f"[retrieve_memory] session_id={request.session_id}")
    try:
        return await svc.retrieve(request)
    except Exception as e:
        logger.error(f"[retrieve_memory] error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ── Chat Session Endpoints ─────────────────────────────────────────────────

@router.post("/sessions", status_code=201)
async def create_session(
    request: ChatSessionCreate,
    dynamo: DynamoDBService = Depends(get_dynamo),
):
    """Create a new named chat session for a user."""
    try:
        await dynamo.create_user_session(request.user_id, request.session_id, request.session_name)
        return {"status": "created", "session_id": request.session_id}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/sessions/{user_id}", response_model=UserSessionsResponse)
async def list_sessions(
    user_id: str,
    dynamo: DynamoDBService = Depends(get_dynamo),
):
    """List all chat sessions for a user, newest first."""
    try:
        raw = await dynamo.get_user_sessions(user_id)
        sessions = sorted(
            [ChatSessionInfo(
                session_id=s["session_id"],
                session_name=s.get("session_name", "New Chat"),
                created_at=s.get("created_at", ""),
                last_message_at=s.get("last_message_at"),
                message_count=int(s.get("message_count", 0)),
            ) for s in raw],
            key=lambda x: x.last_message_at or x.created_at,
            reverse=True,
        )
        return UserSessionsResponse(user_id=user_id, sessions=sessions)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.put("/sessions/{user_id}/{session_id}/rename")
async def rename_session(
    user_id: str,
    session_id: str,
    request: ChatSessionRename,
    dynamo: DynamoDBService = Depends(get_dynamo),
):
    """Rename a chat session."""
    try:
        await dynamo.rename_user_session(user_id, session_id, request.session_name)
        return {"status": "renamed"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/sessions/{user_id}/{session_id}")
async def delete_session(
    user_id: str,
    session_id: str,
    dynamo: DynamoDBService = Depends(get_dynamo),
):
    """Delete a chat session record (does not delete memory entries)."""
    try:
        await dynamo.delete_user_session(user_id, session_id)
        return {"status": "deleted"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ── Explicit Forgetting ────────────────────────────────────────────────────
@router.post("/forget")
async def forget_topic(
    request: ForgetRequest,
    svc: MemoryService = Depends(get_memory_service),
):
    """
    Explicit forgetting: delete memory entries semantically related to a topic.
    e.g. POST /memory/forget {"session_id": "...", "topic": "GANs"}
    Removes matching vectors from ChromaDB so they no longer influence retrieval.
    """
    logger.info(f"[forget_topic] session={request.session_id} topic='{request.topic}'")
    try:
        result = await svc.forget_topic(request.session_id, request.topic, request.top_k)
        return {"status": "ok", **result}
    except Exception as e:
        logger.error(f"[forget_topic] error: {e}")
        raise HTTPException(status_code=500, detail=str(e))
