"""
MLflow Logging Service
Tracks embedding quality, retrieval latency, RAG query metrics, and ingestion stats.
All logging is non-blocking — failures are silently swallowed so the main pipeline
is never interrupted by a missing MLflow server.
"""
import time
import functools
from contextlib import contextmanager
from typing import Optional, Dict, Any, Callable
from loguru import logger

try:
    import mlflow
    import mlflow.pytorch
    _MLFLOW_AVAILABLE = True
except ImportError:
    _MLFLOW_AVAILABLE = False


class MLflowService:

    def __init__(self, tracking_uri: str, experiment_name: str):
        self.tracking_uri = tracking_uri
        self.experiment_name = experiment_name
        self._initialized = False

    def _ensure_init(self):
        if self._initialized or not _MLFLOW_AVAILABLE:
            return
        try:
            mlflow.set_tracking_uri(self.tracking_uri)
            mlflow.set_experiment(self.experiment_name)
            self._initialized = True
            logger.info(f"[MLflow] initialized — tracking at {self.tracking_uri}")
        except Exception as e:
            logger.warning(f"[MLflow] init failed (non-fatal): {e}")

    # ── Context manager for a run ──────────────────────────────────────────

    @contextmanager
    def run(self, run_name: str, tags: Optional[Dict[str, str]] = None):
        """Use as: with mlflow_svc.run('embed_query'): ..."""
        self._ensure_init()
        if not _MLFLOW_AVAILABLE or not self._initialized:
            yield
            return
        try:
            with mlflow.start_run(run_name=run_name, tags=tags or {}):
                yield
        except Exception as e:
            logger.warning(f"[MLflow] run context error (non-fatal): {e}")
            yield

    # ── Metric helpers ─────────────────────────────────────────────────────

    def log_metric(self, key: str, value: float, step: Optional[int] = None):
        if not _MLFLOW_AVAILABLE or not self._initialized:
            return
        try:
            mlflow.log_metric(key, value, step=step)
        except Exception:
            pass

    def log_metrics(self, metrics: Dict[str, float], step: Optional[int] = None):
        if not _MLFLOW_AVAILABLE or not self._initialized:
            return
        try:
            mlflow.log_metrics(metrics, step=step)
        except Exception:
            pass

    def log_param(self, key: str, value: Any):
        if not _MLFLOW_AVAILABLE or not self._initialized:
            return
        try:
            mlflow.log_param(key, value)
        except Exception:
            pass

    def log_params(self, params: Dict[str, Any]):
        if not _MLFLOW_AVAILABLE or not self._initialized:
            return
        try:
            mlflow.log_params(params)
        except Exception:
            pass

    # ── Timing decorator ───────────────────────────────────────────────────

    def timed(self, metric_name: str):
        """Decorator: logs wall-clock latency of an async function as a metric."""
        def decorator(func: Callable):
            @functools.wraps(func)
            async def wrapper(*args, **kwargs):
                start = time.perf_counter()
                result = await func(*args, **kwargs)
                elapsed_ms = (time.perf_counter() - start) * 1000
                self.log_metric(metric_name, round(elapsed_ms, 2))
                logger.debug(f"[MLflow] {metric_name}={elapsed_ms:.1f}ms")
                return result
            return wrapper
        return decorator

    # ── Domain-specific loggers ────────────────────────────────────────────

    def log_ingest(self, paper_id: str, chunk_count: int, embed_count: int, duration_ms: float):
        self._ensure_init()
        if not _MLFLOW_AVAILABLE or not self._initialized:
            return
        try:
            with mlflow.start_run(run_name=f"ingest_{paper_id[:20]}"):
                mlflow.log_params({"ingest/paper_id": paper_id})
                mlflow.log_metrics({
                    "ingest/chunk_count": chunk_count,
                    "ingest/embed_count": embed_count,
                    "ingest/duration_ms": duration_ms,
                })
            logger.debug(f"[MLflow] logged ingest for {paper_id}: {chunk_count} chunks, {duration_ms:.1f}ms")
        except Exception as e:
            logger.warning(f"[MLflow] log_ingest failed (non-fatal): {e}")

    def log_rag_query(
        self,
        query: str,
        num_sources: int,
        retrieval_ms: float,
        generation_ms: float,
        total_ms: float,
    ):
        self._ensure_init()
        if not _MLFLOW_AVAILABLE or not self._initialized:
            return
        try:
            from backend.services.monitoring_service import get_gpu_stats, get_system_stats
            sys_stats = get_system_stats()
            gpu_stats = get_gpu_stats()
            with mlflow.start_run(run_name="rag_query"):
                metrics = {
                    "rag/num_sources": num_sources,
                    "rag/retrieval_ms": retrieval_ms,
                    "rag/generation_ms": generation_ms,
                    "rag/total_ms": total_ms,
                }
                if sys_stats:
                    metrics["system/cpu_pct"] = sys_stats.get("cpu_pct", 0)
                    metrics["system/ram_pct"] = sys_stats.get("ram_pct", 0)
                for gpu in gpu_stats:
                    idx = gpu["index"]
                    metrics[f"gpu{idx}/memory_pct"] = gpu["memory_pct"]
                    metrics[f"gpu{idx}/util_pct"] = gpu["gpu_util_pct"]
                    metrics[f"gpu{idx}/temp_c"] = gpu["temp_c"]
                mlflow.log_metrics(metrics)
            logger.debug(f"[MLflow] logged RAG query: {num_sources} sources, total={total_ms:.1f}ms")
        except Exception as e:
            logger.warning(f"[MLflow] log_rag_query failed (non-fatal): {e}")

    def log_embedding_batch(self, batch_size: int, duration_ms: float):
        self._ensure_init()
        if not _MLFLOW_AVAILABLE or not self._initialized:
            return
        try:
            with mlflow.start_run(run_name="embedding_batch"):
                mlflow.log_metrics({
                    "embedding/batch_size": batch_size,
                    "embedding/duration_ms": duration_ms,
                    "embedding/ms_per_sample": round(duration_ms / max(batch_size, 1), 2),
                })
        except Exception as e:
            logger.warning(f"[MLflow] log_embedding_batch failed (non-fatal): {e}")
