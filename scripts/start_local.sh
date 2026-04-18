#!/usr/bin/env bash
# ──────────────────────────────────────────────────────────────────────────
# ResearchCopilot — Local startup (no Docker)
# Starts: ChromaDB · MLflow · FastAPI backend · Streamlit
# Safe to re-run: kills any existing processes on the same ports first.
# ──────────────────────────────────────────────────────────────────────────
set -e
cd "$(dirname "$0")/.."

echo ""
echo "╔══════════════════════════════════════════════════════════════╗"
echo "║         ResearchCopilot — Local Startup                     ║"
echo "╚══════════════════════════════════════════════════════════════╝"
echo ""

# ── Prerequisites ──────────────────────────────────────────────────────────
command -v python3 >/dev/null 2>&1 || { echo "python3 required"; exit 1; }
command -v uvicorn >/dev/null 2>&1 || { echo "Run: pip install -r requirements.txt"; exit 1; }

# ── Copy .env if missing ───────────────────────────────────────────────────
[ -f .env ] || { cp .env.example .env; echo "  ✓ .env created from .env.example"; }

# ── Create data dirs ───────────────────────────────────────────────────────
mkdir -p data/logs data/papers data/embeddings/chroma

# ── Kill any existing ResearchCopilot processes on our ports ──────────────
echo "  → Releasing ports 8000 8501 8001 5000..."
for PORT in 8000 8501 8001 5000; do
    PIDS=$(lsof -ti tcp:$PORT 2>/dev/null || true)
    if [ -n "$PIDS" ]; then
        kill $PIDS 2>/dev/null || true
        echo "    killed existing process(es) on :$PORT"
    fi
done
sleep 1

# ── ChromaDB (background) ─────────────────────────────────────────────────
echo "  → Starting ChromaDB on :8001..."
python3 -m chromadb.cli.cli run --path data/embeddings/chroma --port 8001 &>/dev/null &
CHROMA_PID=$!

# ── MLflow (background) ───────────────────────────────────────────────────
echo "  → Starting MLflow on :5000..."
mlflow server --host 0.0.0.0 --port 5000 \
    --backend-store-uri sqlite:///data/mlflow.db \
    --default-artifact-root data/mlflow_artifacts &>/dev/null &
MLFLOW_PID=$!

sleep 2

# ── Initialize DBs ─────────────────────────────────────────────────────────
echo "  → Initializing databases..."
python3 scripts/init_db.py 2>/dev/null || echo "  ⚠  DB init skipped (services may not be ready)"

# ── Start personal Ollama instance on GPU 0 (CUDA device 1) ──────────────
# Runs on port 11435 to avoid conflicts with the system Ollama on 11434.
# CUDA_VISIBLE_DEVICES=1 pins to RTX PRO 5000 Blackwell (the free GPU).
if command -v ollama >/dev/null 2>&1; then
    OLLAMA_MODEL="${OLLAMA_MODEL:-qwen3:0.6b}"
    OLLAMA_PORT="${OLLAMA_PORT:-11435}"
    if ! curl -s "http://localhost:${OLLAMA_PORT}/api/tags" &>/dev/null; then
        echo "  → Starting personal Ollama on :${OLLAMA_PORT} (GPU 0)..."
        CUDA_VISIBLE_DEVICES=1 OLLAMA_HOST="127.0.0.1:${OLLAMA_PORT}" \
            ollama serve &>/tmp/ollama_personal.log &
        sleep 3
    else
        echo "  ✓ Ollama already running on :${OLLAMA_PORT}"
    fi
    echo "  → Ensuring model '${OLLAMA_MODEL}' is available..."
    OLLAMA_HOST="127.0.0.1:${OLLAMA_PORT}" ollama pull "${OLLAMA_MODEL}" &>/dev/null || true
else
    echo "  ⚠  Ollama not found — RAG answers will use fallback text."
fi

# ── FastAPI backend ────────────────────────────────────────────────────────
echo "  → Starting FastAPI on :8000..."
uvicorn backend.main:app --host 0.0.0.0 --port 8000 --reload &
BACKEND_PID=$!

sleep 3

# ── Streamlit frontend ─────────────────────────────────────────────────────
echo "  → Starting Streamlit on :8501..."
streamlit run frontend/app.py \
    --server.address 0.0.0.0 \
    --server.port 8501 \
    --server.headless true &
FRONTEND_PID=$!

echo ""
echo "  ✓ All services started!"
echo ""
echo "  Streamlit UI  : http://localhost:8501"
echo "  API Docs      : http://localhost:8000/docs"
echo "  API Status    : http://localhost:8000/status"
echo "  MLflow        : http://localhost:5000"
echo "  ChromaDB      : http://localhost:8001"
echo ""
echo "  Press Ctrl+C to stop all services."
echo ""

# ── Cleanup on exit ───────────────────────────────────────────────────────
trap "echo ''; echo 'Stopping...'; kill $CHROMA_PID $MLFLOW_PID $BACKEND_PID $FRONTEND_PID 2>/dev/null; exit 0" INT TERM
wait
