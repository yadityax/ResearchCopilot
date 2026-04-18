#!/usr/bin/env bash
# ──────────────────────────────────────────────────────────────────────────
# ResearchCopilot — Docker Compose startup
# ──────────────────────────────────────────────────────────────────────────
set -e
cd "$(dirname "$0")/.."

echo ""
echo "╔══════════════════════════════════════════════════════════════╗"
echo "║         ResearchCopilot — Docker Compose Startup            ║"
echo "╚══════════════════════════════════════════════════════════════╝"
echo ""

command -v docker >/dev/null 2>&1 || { echo "Docker required"; exit 1; }

[ -f .env ] || cp .env.example .env && echo "  ✓ .env created"

echo "  → Building and starting containers..."
docker compose up --build -d

echo "  → Waiting for services to be healthy..."
sleep 10

OLLAMA_MODEL="${OLLAMA_MODEL:-qwen3.5:latest}"
echo "  → Pulling '${OLLAMA_MODEL}' inside Ollama container..."
docker exec researchcopilot_ollama ollama pull "${OLLAMA_MODEL}" || \
    echo "  ⚠  Ollama pull failed — RAG will use fallback responses."

echo "  → Initializing databases..."
docker exec researchcopilot_backend python3 scripts/init_db.py || true

echo ""
echo "  ✓ Stack is up!"
echo ""
echo "  Streamlit UI  : http://localhost:8501"
echo "  API Docs      : http://localhost:8000/docs"
echo "  API Status    : http://localhost:8000/status"
echo "  MLflow        : http://localhost:5000"
echo "  ChromaDB      : http://localhost:8001"
echo ""
echo "  Logs  : docker compose logs -f"
echo "  Stop  : docker compose down"
echo ""
