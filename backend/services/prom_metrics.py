"""
Prometheus metrics definitions for ResearchCopilot.
All metrics are module-level singletons — import and use anywhere.
"""
from prometheus_client import Counter, Histogram, Gauge

# ── HTTP ───────────────────────────────────────────────────────────────────────
HTTP_REQUESTS = Counter(
    "rc_http_requests_total",
    "Total HTTP requests",
    ["method", "endpoint", "status"],
)
HTTP_LATENCY = Histogram(
    "rc_http_request_duration_seconds",
    "HTTP request duration in seconds",
    ["method", "endpoint"],
    buckets=[0.01, 0.05, 0.1, 0.25, 0.5, 1, 2.5, 5, 10, 30],
)

# ── RAG ───────────────────────────────────────────────────────────────────────
RAG_REQUESTS = Counter("rc_rag_queries_total", "Total RAG queries", ["status"])
RAG_TOTAL_DURATION = Histogram(
    "rc_rag_query_duration_seconds",
    "End-to-end RAG query duration",
    buckets=[0.5, 1, 2, 5, 10, 30, 60, 120, 300],
)
RAG_RETRIEVAL_DURATION = Histogram(
    "rc_rag_retrieval_duration_seconds",
    "ChromaDB retrieval phase duration",
    buckets=[0.01, 0.05, 0.1, 0.5, 1, 2, 5],
)
RAG_GENERATION_DURATION = Histogram(
    "rc_rag_generation_duration_seconds",
    "LLM generation phase duration",
    buckets=[0.5, 1, 2, 5, 10, 30, 60, 120],
)
RAG_SOURCES = Histogram(
    "rc_rag_sources_retrieved",
    "Number of sources retrieved per query",
    buckets=[0, 1, 2, 3, 5, 8, 10, 15],
)

# ── Ingest ────────────────────────────────────────────────────────────────────
INGEST_TOTAL = Counter(
    "rc_paper_ingests_total", "Paper ingest attempts", ["status"]
)
INGEST_CHUNKS = Counter("rc_paper_chunks_total", "Total chunks created from papers")
INGEST_DURATION = Histogram(
    "rc_paper_ingest_duration_seconds",
    "Paper ingest duration",
    buckets=[0.1, 0.5, 1, 2, 5, 10, 30, 60],
)

# ── Embeddings ────────────────────────────────────────────────────────────────
EMBED_DURATION = Histogram(
    "rc_embedding_duration_seconds",
    "Embedding batch duration",
    buckets=[0.01, 0.05, 0.1, 0.5, 1, 2, 5],
)
EMBED_BATCH_SIZE = Histogram(
    "rc_embedding_batch_size",
    "Embedding batch size",
    buckets=[1, 5, 10, 20, 50, 100, 200, 512],
)

# ── GPU (gauges updated every 15 s) ───────────────────────────────────────────
GPU_MEM_USED = Gauge(
    "rc_gpu_memory_used_mb", "GPU memory used (MB)", ["gpu_id", "gpu_name"]
)
GPU_MEM_PCT = Gauge(
    "rc_gpu_memory_pct", "GPU memory used (%)", ["gpu_id", "gpu_name"]
)
GPU_UTIL = Gauge(
    "rc_gpu_utilization_pct", "GPU utilization (%)", ["gpu_id", "gpu_name"]
)
GPU_TEMP = Gauge(
    "rc_gpu_temperature_c", "GPU temperature (°C)", ["gpu_id", "gpu_name"]
)
GPU_POWER = Gauge(
    "rc_gpu_power_w", "GPU power draw (W)", ["gpu_id", "gpu_name"]
)

# ── System ────────────────────────────────────────────────────────────────────
SYS_CPU = Gauge("rc_system_cpu_pct", "System CPU usage (%)")
SYS_RAM = Gauge("rc_system_ram_pct", "System RAM usage (%)")
SYS_DISK = Gauge("rc_system_disk_pct", "System disk usage (%)")


def update_system_gauges():
    """Push current GPU + system stats into Prometheus gauges."""
    try:
        from backend.services.monitoring_service import get_gpu_stats, get_system_stats
        sys = get_system_stats()
        if sys:
            SYS_CPU.set(sys.get("cpu_pct", 0))
            SYS_RAM.set(sys.get("ram_pct", 0))
            SYS_DISK.set(sys.get("disk_pct", 0))
        for gpu in get_gpu_stats():
            lbl = {"gpu_id": str(gpu["index"]), "gpu_name": gpu["name"]}
            GPU_MEM_USED.labels(**lbl).set(gpu["memory_used_mb"])
            GPU_MEM_PCT.labels(**lbl).set(gpu["memory_pct"])
            GPU_UTIL.labels(**lbl).set(gpu["gpu_util_pct"])
            GPU_TEMP.labels(**lbl).set(gpu["temp_c"])
            GPU_POWER.labels(**lbl).set(gpu["power_w"])
    except Exception:
        pass
