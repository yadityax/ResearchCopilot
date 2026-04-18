from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response
from contextlib import asynccontextmanager
from loguru import logger
import sys
import time
import httpx
import asyncio

from prometheus_client import generate_latest, CONTENT_TYPE_LATEST

from backend.config import get_settings
from backend.routers import papers, embeddings, memory, rag, reports, ingest_pipeline
from backend.services.monitoring_service import (
    log_system_snapshot, log_to_mlflow, request_tracker, get_gpu_stats, get_system_stats
)
from backend.services.mlflow_service import MLflowService
from backend.services.prom_metrics import (
    HTTP_REQUESTS, HTTP_LATENCY, update_system_gauges
)

# ── Logging setup ──────────────────────────────────────────────────────────
settings = get_settings()
logger.remove()
logger.add(
    sys.stdout,
    level=settings.log_level,
    format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan> - {message}",
)
logger.add("data/logs/backend.log", rotation="10 MB", retention="7 days", level="DEBUG")


# ── Service health probes ──────────────────────────────────────────────────
async def _probe(url: str, timeout: float = 3.0) -> bool:
    try:
        async with httpx.AsyncClient(timeout=timeout) as c:
            r = await c.get(url)
            return r.status_code < 500
    except Exception:
        return False


async def _check_services() -> dict:
    checks = await asyncio.gather(
        _probe(f"http://{settings.chroma_host}:{settings.chroma_port}/api/v2/heartbeat"),
        _probe(f"{settings.dynamodb_endpoint_url}/"),
        _probe(f"{settings.ollama_base_url}/api/tags"),
        _probe(f"{settings.mlflow_tracking_uri}/health"),
        return_exceptions=True,
    )
    names = ["chromadb", "dynamodb", "ollama", "mlflow"]
    return {
        name: ("online" if result is True else "offline")
        for name, result in zip(names, checks)
    }


# ── Periodic system monitor ────────────────────────────────────────────────
_mlflow_svc = MLflowService(settings.mlflow_tracking_uri, settings.mlflow_experiment_name)

async def _periodic_monitor():
    """Log system + GPU stats every 5 min to loguru, MLflow, and Prometheus gauges."""
    while True:
        await asyncio.sleep(300)
        try:
            snapshot = log_system_snapshot("periodic")
            log_to_mlflow(_mlflow_svc, "periodic", snapshot["system"], snapshot["gpus"])
            update_system_gauges()
        except Exception as e:
            logger.warning(f"[Monitor] periodic snapshot failed: {e}")


async def _prometheus_gauge_updater():
    """Update Prometheus system/GPU gauges every 15 seconds."""
    while True:
        await asyncio.sleep(15)
        try:
            update_system_gauges()
        except Exception:
            pass


# ── Lifespan ───────────────────────────────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info(f"Starting {settings.app_name} (env={settings.app_env})")
    service_status = await _check_services()
    for name, status in service_status.items():
        icon = "✓" if status == "online" else "✗"
        log_fn = logger.info if status == "online" else logger.warning
        log_fn(f"  {icon} {name}: {status}")

    # Log initial system snapshot
    snapshot = log_system_snapshot("startup")
    log_to_mlflow(_mlflow_svc, "startup", snapshot["system"], snapshot["gpus"])

    # Start background monitors
    monitor_task = asyncio.create_task(_periodic_monitor())
    gauge_task = asyncio.create_task(_prometheus_gauge_updater())
    update_system_gauges()  # immediate first reading
    logger.info("[Monitor] periodic system monitor started (interval=5min)")
    logger.info("[Prometheus] gauge updater started (interval=15s)")

    yield

    monitor_task.cancel()
    gauge_task.cancel()
    logger.info(f"{settings.app_name} shutting down.")
    log_system_snapshot("shutdown")


# ── App ────────────────────────────────────────────────────────────────────
app = FastAPI(
    title="ResearchCopilot API",
    description=(
        "Production-grade AI Research Assistant with Adaptive Memory & RAG Pipeline.\n\n"
        "**Architecture:** arXiv/Semantic Scholar → PDF ingestion → sentence-transformers "
        "(all-MiniLM-L6-v2, 384-dim) → EncoderMLP + SinusoidalMLP → ChromaDB → Llama 3 (Ollama)\n\n"
        "**MLOps:** MLflow experiment tracking · DynamoDB metadata store · Docker Compose orchestration"
    ),
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.middleware("http")
async def request_logging_middleware(request: Request, call_next):
    """Log every request: method, path, status, latency. Track errors."""
    start = time.perf_counter()
    path = request.url.path
    method = request.method
    client_ip = request.client.host if request.client else "unknown"
    logger.info(f"[REQ] {method} {path} from {client_ip}")
    try:
        response = await call_next(request)
        latency_s = time.perf_counter() - start
        latency_ms = latency_s * 1000
        is_error = response.status_code >= 400
        request_tracker.record(latency_ms, is_error=is_error)
        # Prometheus
        HTTP_REQUESTS.labels(method=method, endpoint=path, status=str(response.status_code)).inc()
        HTTP_LATENCY.labels(method=method, endpoint=path).observe(latency_s)
        level = "warning" if is_error else "info"
        getattr(logger, level)(
            f"[RES] {method} {path} → {response.status_code} ({latency_ms:.1f}ms)"
        )
        return response
    except Exception as exc:
        latency_s = time.perf_counter() - start
        request_tracker.record(latency_s * 1000, is_error=True)
        HTTP_REQUESTS.labels(method=method, endpoint=path, status="500").inc()
        HTTP_LATENCY.labels(method=method, endpoint=path).observe(latency_s)
        logger.error(
            f"[ERR] {method} {path} → EXCEPTION after {latency_s*1000:.1f}ms: {exc}"
        )
        raise

# ── Routers ────────────────────────────────────────────────────────────────
PREFIX = settings.api_prefix

app.include_router(papers.router,          prefix=PREFIX)  # /papers/search, /papers/ingest, /papers/ingest/pdf
app.include_router(embeddings.router,      prefix=PREFIX)  # /embeddings/query
app.include_router(memory.router,          prefix=PREFIX)  # /memory/store, /memory/retrieve
app.include_router(rag.router,             prefix=PREFIX)  # /rag/query
app.include_router(reports.router,         prefix=PREFIX)  # /report/generate
app.include_router(ingest_pipeline.router, prefix=PREFIX)  # /pipeline/search-and-ingest, /pipeline/ingest-by-id


# ── Health & Status ────────────────────────────────────────────────────────
@app.get("/health", tags=["Health"])
async def health():
    """Quick liveness probe — always returns 200 if the server is up."""
    return {"status": "ok", "app": settings.app_name, "env": settings.app_env, "version": "1.0.0"}


@app.get("/status", tags=["Health"])
async def status():
    """Detailed readiness probe — checks all downstream services."""
    service_status = await _check_services()
    all_ok = all(v == "online" for v in service_status.values())
    return {
        "status": "ready" if all_ok else "degraded",
        "app": settings.app_name,
        "services": service_status,
        "config": {
            "embedding_model": settings.embedding_model,
            "embedding_dim": settings.embedding_dim,
            "llm_model": settings.ollama_model,
            "chunk_size": settings.chunk_size,
            "chunk_overlap": settings.chunk_overlap,
        },
    }


@app.get("/metrics", tags=["Health"])
async def metrics():
    """Real-time system, GPU, and request metrics."""
    sys_stats = get_system_stats()
    gpu_stats = get_gpu_stats()
    req_stats = request_tracker.summary()
    return {
        "requests": req_stats,
        "system": sys_stats,
        "gpus": gpu_stats,
    }


@app.get("/prometheus", tags=["Health"], include_in_schema=False)
async def prometheus_metrics():
    """Prometheus scrape endpoint — returns metrics in text exposition format."""
    return Response(content=generate_latest(), media_type=CONTENT_TYPE_LATEST)


@app.get("/stats", tags=["Health"])
async def stats():
    """Live counts: total papers ingested and total users."""
    import boto3, requests as req_lib
    from backend.config import get_settings
    s = get_settings()

    # Papers — ChromaDB research_papers collection
    paper_chunks, unique_papers, users = 0, 0, 0
    try:
        chroma_base = f"http://{s.chroma_host}:{s.chroma_port}/api/v2/tenants/default_tenant/databases/default_database"
        cols = req_lib.get(f"{chroma_base}/collections", timeout=3).json()
        col_map = {c["name"]: c["id"] for c in cols}
        if "research_papers" in col_map:
            paper_chunks = req_lib.get(f"{chroma_base}/collections/{col_map['research_papers']}/count", timeout=3).json()
        # Unique papers by paper_id metadata via DynamoDB ResearchPapers table
    except Exception:
        pass

    try:
        ddb = boto3.resource("dynamodb", region_name=s.aws_region,
                             aws_access_key_id=s.aws_access_key_id,
                             aws_secret_access_key=s.aws_secret_access_key,
                             endpoint_url=s.dynamodb_endpoint_url)
        papers_tbl = ddb.Table("ResearchPapers")
        unique_papers = papers_tbl.scan(Select="COUNT")["Count"]

        sessions_tbl = ddb.Table("user_sessions")
        session_rows = sessions_tbl.scan().get("Items", [])
        users = len({str(item.get("user_id", "unknown")) for item in session_rows if item.get("user_id")})
    except Exception:
        pass

    return {
        "papers": unique_papers,
        "paper_chunks": paper_chunks,
        "users": users,
    }


@app.get("/papers-list", tags=["Health"])
async def papers_list():
    """All ingested papers with title, arxiv link, and chunk count."""
    import boto3
    from backend.config import get_settings
    s = get_settings()
    papers = []
    try:
        ddb = boto3.resource("dynamodb", region_name=s.aws_region,
                             aws_access_key_id=s.aws_access_key_id,
                             aws_secret_access_key=s.aws_secret_access_key,
                             endpoint_url=s.dynamodb_endpoint_url)
        tbl = ddb.Table("ResearchPapers")
        resp = tbl.scan()
        for item in resp.get("Items", []):
            pid = item.get("paper_id", "")
            arxiv_id = pid.replace("arxiv_", "").rsplit("v", 1)[0] if pid.startswith("arxiv_") else ""
            papers.append({
                "title": item.get("title", "Unknown"),
                "paper_id": pid,
                "arxiv_url": f"https://arxiv.org/abs/{arxiv_id}" if arxiv_id else "",
                "chunks": int(item.get("chunk_count", 0)),
                "ingested_at": str(item.get("ingested_at", "")),
            })
        papers.sort(key=lambda x: x["ingested_at"], reverse=True)
    except Exception:
        pass
    return {"papers": papers, "total": len(papers)}


@app.get("/users", tags=["Health"])
async def users_list():
    """List all users with their session activity."""
    import boto3
    from backend.config import get_settings
    s = get_settings()
    rows = []
    try:
        ddb = boto3.resource("dynamodb", region_name=s.aws_region,
                             aws_access_key_id=s.aws_access_key_id,
                             aws_secret_access_key=s.aws_secret_access_key,
                             endpoint_url=s.dynamodb_endpoint_url)
        tbl = ddb.Table("user_sessions")
        resp = tbl.scan()
        seen = {}
        for item in resp.get("Items", []):
            uid = item.get("user_id", "unknown")
            if uid not in seen:
                seen[uid] = {"user_id": uid, "sessions": 0, "messages": 0, "last_active": ""}
            seen[uid]["sessions"] += 1
            seen[uid]["messages"] += int(item.get("message_count", 0))
            ts = str(item.get("last_message_at", ""))
            if ts > seen[uid]["last_active"]:
                seen[uid]["last_active"] = ts
        rows = sorted(seen.values(), key=lambda x: x["last_active"], reverse=True)
    except Exception:
        pass
    return {"users": rows, "total": len(rows)}


@app.get("/users/details", tags=["Health"])
async def users_details():
    """Return all users with their complete chat session details."""
    import boto3
    from backend.config import get_settings

    s = get_settings()
    detailed = []

    try:
        ddb = boto3.resource(
            "dynamodb",
            region_name=s.aws_region,
            aws_access_key_id=s.aws_access_key_id,
            aws_secret_access_key=s.aws_secret_access_key,
            endpoint_url=s.dynamodb_endpoint_url,
        )
        tbl = ddb.Table("user_sessions")
        resp = tbl.scan()
        user_map = {}

        for item in resp.get("Items", []):
            uid = str(item.get("user_id", "unknown"))
            session_info = {
                "session_id": str(item.get("session_id", "")),
                "session_name": str(item.get("session_name", "New Chat")),
                "message_count": int(item.get("message_count", 0)),
                "last_message_at": str(item.get("last_message_at", "")),
                "updated_at": str(item.get("updated_at", "")),
            }

            if uid not in user_map:
                user_map[uid] = {
                    "user_id": uid,
                    "sessions": [],
                    "total_sessions": 0,
                    "total_messages": 0,
                    "last_active": "",
                }

            user_map[uid]["sessions"].append(session_info)
            user_map[uid]["total_sessions"] += 1
            user_map[uid]["total_messages"] += session_info["message_count"]

            ts = session_info["last_message_at"]
            if ts > user_map[uid]["last_active"]:
                user_map[uid]["last_active"] = ts

        for user_row in user_map.values():
            user_row["sessions"] = sorted(
                user_row["sessions"],
                key=lambda x: x.get("last_message_at", ""),
                reverse=True,
            )
            detailed.append(user_row)

        detailed = sorted(detailed, key=lambda x: x.get("last_active", ""), reverse=True)
    except Exception:
        pass

    return {
        "users": detailed,
        "total_users": len(detailed),
        "total_sessions": sum(u.get("total_sessions", 0) for u in detailed),
    }


@app.get("/", tags=["Health"])
async def root():
    return {
        "message": f"Welcome to {settings.app_name} API",
        "docs": "/docs",
        "health": "/health",
        "status": "/status",
        "metrics": "/metrics",
    }
