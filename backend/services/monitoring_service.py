"""
Monitoring Service — system, GPU, and service health metrics.
Logs to loguru and MLflow. Called periodically and on each request.
"""
import subprocess
import time
from typing import Dict, Any

import psutil
from loguru import logger


# ── GPU ─────────────────────────────────────────────────────────────────────

def get_gpu_stats() -> list[dict]:
    """Query nvidia-smi for per-GPU stats."""
    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=index,name,memory.used,memory.total,utilization.gpu,temperature.gpu,power.draw",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True, text=True, timeout=5,
        )
        gpus = []
        for line in result.stdout.strip().splitlines():
            parts = [p.strip() for p in line.split(",")]
            if len(parts) >= 7:
                gpus.append({
                    "index": int(parts[0]),
                    "name": parts[1],
                    "memory_used_mb": float(parts[2]),
                    "memory_total_mb": float(parts[3]),
                    "memory_pct": round(float(parts[2]) / max(float(parts[3]), 1) * 100, 1),
                    "gpu_util_pct": float(parts[4]),
                    "temp_c": float(parts[5]),
                    "power_w": float(parts[6]) if parts[6] not in ("N/A", "[N/A]") else 0.0,
                })
        return gpus
    except Exception as e:
        logger.warning(f"[Monitor] nvidia-smi failed: {e}")
        return []


# ── CPU / RAM ────────────────────────────────────────────────────────────────

def get_system_stats() -> Dict[str, Any]:
    """Return CPU, RAM, and disk stats."""
    try:
        vm = psutil.virtual_memory()
        disk = psutil.disk_usage("/")
        return {
            "cpu_pct": psutil.cpu_percent(interval=None),
            "cpu_count": psutil.cpu_count(),
            "ram_used_gb": round(vm.used / 1e9, 2),
            "ram_total_gb": round(vm.total / 1e9, 2),
            "ram_pct": vm.percent,
            "disk_used_gb": round(disk.used / 1e9, 2),
            "disk_total_gb": round(disk.total / 1e9, 2),
            "disk_pct": disk.percent,
        }
    except Exception as e:
        logger.warning(f"[Monitor] system stats failed: {e}")
        return {}


# ── Combined snapshot ────────────────────────────────────────────────────────

def log_system_snapshot(tag: str = "periodic") -> Dict[str, Any]:
    """Log a full system + GPU snapshot to loguru."""
    sys_stats = get_system_stats()
    gpu_stats = get_gpu_stats()

    logger.info(
        f"[Monitor:{tag}] CPU={sys_stats.get('cpu_pct', '?')}% | "
        f"RAM={sys_stats.get('ram_used_gb', '?')}/{sys_stats.get('ram_total_gb', '?')}GB "
        f"({sys_stats.get('ram_pct', '?')}%)"
    )
    for gpu in gpu_stats:
        logger.info(
            f"[Monitor:{tag}] GPU[{gpu['index']}] {gpu['name']} | "
            f"VRAM={gpu['memory_used_mb']:.0f}/{gpu['memory_total_mb']:.0f}MB ({gpu['memory_pct']}%) | "
            f"Util={gpu['gpu_util_pct']}% | Temp={gpu['temp_c']}°C | Power={gpu['power_w']}W"
        )
    return {"system": sys_stats, "gpus": gpu_stats}


def log_to_mlflow(mlflow_svc, tag: str, sys_stats: dict, gpu_stats: list):
    """Push system metrics to MLflow as a run."""
    try:
        import mlflow
        mlflow.set_tracking_uri(mlflow_svc.tracking_uri)
        mlflow.set_experiment(mlflow_svc.experiment_name)
        with mlflow.start_run(run_name=f"system_{tag}"):
            metrics = {}
            if sys_stats:
                metrics.update({
                    "system/cpu_pct": sys_stats.get("cpu_pct", 0),
                    "system/ram_pct": sys_stats.get("ram_pct", 0),
                    "system/ram_used_gb": sys_stats.get("ram_used_gb", 0),
                    "system/disk_pct": sys_stats.get("disk_pct", 0),
                })
            for gpu in gpu_stats:
                idx = gpu["index"]
                metrics.update({
                    f"gpu{idx}/memory_pct": gpu["memory_pct"],
                    f"gpu{idx}/memory_used_mb": gpu["memory_used_mb"],
                    f"gpu{idx}/util_pct": gpu["gpu_util_pct"],
                    f"gpu{idx}/temp_c": gpu["temp_c"],
                    f"gpu{idx}/power_w": gpu["power_w"],
                })
            mlflow.log_metrics(metrics)
    except Exception as e:
        logger.warning(f"[Monitor] MLflow system log failed: {e}")


# ── Request stats tracker (in-memory) ───────────────────────────────────────

class RequestTracker:
    """Thread-safe in-memory counters for request stats."""

    def __init__(self):
        self.total = 0
        self.errors = 0
        self.latencies: list[float] = []
        self._start = time.time()

    def record(self, latency_ms: float, is_error: bool = False):
        self.total += 1
        if is_error:
            self.errors += 1
        self.latencies.append(latency_ms)
        if len(self.latencies) > 1000:
            self.latencies = self.latencies[-500:]

    def summary(self) -> dict:
        lats = self.latencies
        uptime_s = time.time() - self._start
        return {
            "total_requests": self.total,
            "error_requests": self.errors,
            "error_rate_pct": round(self.errors / max(self.total, 1) * 100, 2),
            "avg_latency_ms": round(sum(lats) / max(len(lats), 1), 1),
            "p95_latency_ms": round(sorted(lats)[int(len(lats) * 0.95)] if lats else 0, 1),
            "uptime_seconds": round(uptime_s, 0),
            "req_per_minute": round(self.total / max(uptime_s / 60, 0.01), 2),
        }


# Global tracker instance
request_tracker = RequestTracker()
