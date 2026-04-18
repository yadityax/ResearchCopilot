# ResearchCopilot
### A Production-Grade AI Research Assistant with Adaptive Memory

MLOps Project — April 2026

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                     Streamlit Frontend (8501)                   │
│   Paper Discovery │ PDF Ingestion │ RAG Q&A │ Memory History    │
└────────────────────────────┬────────────────────────────────────┘
                             │ HTTP
┌────────────────────────────▼────────────────────────────────────┐
│                   FastAPI Backend (8000)                        │
│  /papers/search   /papers/ingest   /embeddings/query           │
│  /memory/store    /memory/retrieve  /rag/query                 │
│  /report/generate  /pipeline/search-and-ingest                 │
└──────┬──────────────────┬──────────────┬────────────────────────┘
       │                  │              │
       ▼                  ▼              ▼
┌──────────────┐  ┌───────────────┐  ┌──────────────────────────┐
│   arXiv API  │  │  Embedding    │  │   Adaptive Memory        │
│ Semantic     │  │  Pipeline     │  │                          │
│ Scholar API  │  │               │  │  DynamoDB (history)      │
└──────────────┘  │ MiniLM-L6-v2  │  │  ChromaDB (semantic)     │
                  │ (pretrained)  │  └──────────────────────────┘
                  │ EncoderMLP    │
                  │ SinusoidalMLP │
                  │ (384-dim)     │
                  └──────┬────────┘
                         │
                  ┌──────▼────────┐
                  │  ChromaDB     │  ← Detroit Vector Database
                  │  Vector Store │
                  └──────┬────────┘
                         │
                  ┌──────▼────────┐
                  │  Qwen 3.5     │  ← via Ollama (local, pretrained)
                  │  (Ollama)     │
                  └───────────────┘

MLOps:  MLflow (5000) · Docker Compose · pytest (61 tests)
```

---

## Tech Stack

| Component | Technology |
|---|---|
| Backend | FastAPI + Uvicorn |
| Frontend | Streamlit |
| LLM | Qwen 3.5 via Ollama (local, pretrained) |
| Embeddings | `sentence-transformers/all-MiniLM-L6-v2` (384-dim, pretrained) |
| MLP Layers | PyTorch — EncoderMLP + SinusoidalMLP (projection on top of pretrained) |
| Vector DB | ChromaDB |
| Metadata DB | AWS DynamoDB (local Docker image in dev) |
| Paper APIs | arXiv API + Semantic Scholar API |
| Experiment Tracking | MLflow |
| Containerisation | Docker Compose |
| Testing | pytest — 61 tests across 4 test files |

---

## Prerequisites

- Python 3.11+
- [Ollama](https://ollama.ai) (for LLM; optional — falls back to offline mode)
- Docker + Docker Compose (for containerised setup)

---

## Quickstart — Local (no Docker)

```bash
# 1. Clone and install
git clone <repo>
cd MLOps_Project
pip install -r requirements.txt

# 2. Configure
cp .env.example .env        # edit as needed

# 3. Pull the LLM (optional but recommended)
ollama pull qwen3.5:latest

# 4. Start everything
bash scripts/start_local.sh
```

**Or start services individually:**

```bash
# Terminal 1 — Backend
uvicorn backend.main:app --reload

# Terminal 2 — Frontend
streamlit run frontend/app.py

# Terminal 3 — ChromaDB
python -m chromadb.cli.cli run --path data/embeddings/chroma --port 8001

# Terminal 4 — MLflow
mlflow server --host 0.0.0.0 --port 5000 --backend-store-uri sqlite:///data/mlflow.db
```

---

## Quickstart — Docker Compose

```bash
cp .env.example .env
bash scripts/start_docker.sh
```

---

## Services

| Service | URL | Description |
|---|---|---|
| Streamlit UI | http://localhost:8501 | Main user interface |
| API Docs | http://localhost:8000/docs | Swagger / OpenAPI |
| API Status | http://localhost:8000/status | Health + service readiness |
| MLflow | http://localhost:5000 | Experiment tracking dashboard |
| ChromaDB | http://localhost:8001 | Vector database |

---

## API Endpoints

| Method | Endpoint | Description |
|---|---|---|
| POST | `/api/v1/papers/search` | Search arXiv + Semantic Scholar |
| POST | `/api/v1/papers/ingest` | Ingest paper from text content |
| POST | `/api/v1/papers/ingest/pdf` | Ingest paper from PDF upload |
| POST | `/api/v1/embeddings/query` | Semantic similarity search |
| POST | `/api/v1/memory/store` | Store a conversation turn |
| POST | `/api/v1/memory/retrieve` | Retrieve session history |
| POST | `/api/v1/rag/query` | Full RAG pipeline Q&A |
| POST | `/api/v1/report/generate` | Generate research report |
| POST | `/api/v1/pipeline/search-and-ingest` | Search + ingest in one call |
| POST | `/api/v1/pipeline/ingest-by-id` | Ingest single arXiv paper by ID |

---

## Demo

```bash
# With backend running:
python scripts/demo.py --topic "transformer attention mechanism" \
                       --query "What is self-attention?"

# Ingest specific papers:
python scripts/ingest_arxiv.py --query "BERT language model" --max 5
python scripts/ingest_arxiv.py --id 1706.03762   # Attention Is All You Need
```

---

## Testing

```bash
# Run all 61 tests
pytest tests/ -v

# Run by day
pytest tests/test_day1.py -v   # schemas, MLP models, chunking
pytest tests/test_day2.py -v   # embeddings, vector DB, ingest pipeline
pytest tests/test_day3.py -v   # RAG, memory, report generation
pytest tests/test_day4.py -v   # API endpoint integration tests
```

All tests run **fully offline** — no Ollama, DynamoDB, or ChromaDB server needed.

---

## RAG Pipeline Detail

```
User Query
    │
    ▼
sentence-transformers/all-MiniLM-L6-v2   (pretrained, 384-dim)
    │
    ▼
EncoderMLP  (2-layer PyTorch, residual connection)
    │
    ▼
SinusoidalMLP  (positional encoding layer)
    │
    ▼
ChromaDB cosine similarity search  (top-k chunks)
    │
    ▼
[Optional] Adaptive Memory context  (last 6 turns)
    │
    ▼
Qwen 3.5 prompt  →  Ollama  →  Answer + Citations
```

---

## Project Structure

```
MLOps_Project/
├── backend/
│   ├── main.py                  # FastAPI app
│   ├── config.py                # Pydantic settings
│   ├── models/
│   │   ├── schemas.py           # Request/response models
│   │   └── mlp_models.py        # EncoderMLP + SinusoidalMLP
│   ├── routers/                 # One file per endpoint group
│   └── services/
│       ├── paper_discovery.py   # arXiv + Semantic Scholar
│       ├── pdf_ingestion.py     # PDF parsing + chunking
│       ├── embedding_service.py # MiniLM + MLP projection
│       ├── vector_db.py         # ChromaDB interface
│       ├── dynamo_db.py         # DynamoDB interface
│       ├── memory_service.py    # Adaptive memory
│       ├── llm_service.py       # Ollama/Llama3
│       ├── rag_service.py       # Full RAG pipeline
│       ├── ingest_pipeline.py   # Search → embed → store
│       └── mlflow_service.py    # Experiment tracking
├── frontend/
│   ├── app.py                   # Streamlit entry point
│   └── pages/
│       ├── paper_discovery.py
│       ├── pdf_ingestion.py
│       ├── rag_query.py
│       └── memory_history.py
├── tests/
│   ├── test_day1.py  (13 tests)
│   ├── test_day2.py  (12 tests)
│   ├── test_day3.py  (13 tests)
│   └── test_day4.py  (varies)
├── scripts/
│   ├── start_local.sh
│   ├── start_docker.sh
│   ├── demo.py
│   ├── ingest_arxiv.py
│   └── init_db.py
├── docker/
│   ├── Dockerfile.backend
│   └── Dockerfile.frontend
├── docker-compose.yml
├── requirements.txt
└── .env.example
```

---

## Notes

- **Neural watermarking** is intentionally excluded from this implementation.
- All ML models are **pretrained** — no training from scratch.
- The MLP layers (EncoderMLP, SinusoidalMLP) are lightweight projection layers
  initialised near-identity so they do not distort pretrained embeddings.
- ChromaDB automatically falls back to a local persistent client if the Docker
  server is not running, so development works without Docker.
