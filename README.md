# ResearchCopilot: a retrieval-augmented research assistant with adaptive memory

Course project (MLOps). A self-hosted assistant that finds research papers on arXiv and Semantic Scholar, ingests them
(including uploaded PDFs), and answers questions grounded in the retrieved text with `[Source N]` citations. It remembers
the conversation, lets interest in old topics fade over time, and lets the user delete a topic on request. Everything is
built from **pretrained** components (no model is trained in this repository) and shipped as a real service: containerised,
tested and deployed by GitHub Actions to Kubernetes, and monitored with Prometheus, Grafana and MLflow.

**Live demo: https://huggingface.co/spaces/m25csa001/ResearchCopilot** | **Code: https://github.com/yadityax/ResearchCopilot**

**In one paragraph.** A FastAPI backend exposes 16 REST endpoints (`/api/v1/...`) for paper search, PDF/text ingestion,
retrieval, RAG question answering, long-form report generation and session memory. Text is embedded with
`all-MiniLM-L6-v2` (384-dim), stored in ChromaDB, and answered by Qwen 3.5 (35B) served locally through Ollama. A query is
first rewritten by the LLM into search terms, a retrieval query, a well-formed question and (if one is named) the paper's
official title, then answered from a paper-scoped + general retrieval. Conversation history is written to DynamoDB and ChromaDB, and semantic search over it is
re-ranked with an exponential recency decay. A GitHub Actions pipeline lints and tests every push (71 pytest tests) and, on
`main`, builds and pushes a Docker image and rolls it out to a Kubernetes cluster that autoscales between 2 and 10 pods.
A small labelled benchmark (32 questions over 12 arXiv papers, see [Evaluation](#evaluation)) measures retrieval and answer
quality; the limitations are listed under [Caveats](#caveats).

## Problem statement

Reading a new research area means finding the right papers, then finding the right paragraph, equation or definition inside
them. A plain LLM answers fluently but cannot point to a source and often gets equations wrong. A plain retrieval demo finds
paragraphs but forgets what the user asked a minute ago, so "give the equation" or "explain more" fails.

ResearchCopilot targets three things:
1. **Grounded answers.** Answers are built from retrieved paper chunks and cite them, and equations are returned in LaTeX.
2. **Memory that behaves like memory.** Follow-up questions are resolved from the conversation, old topics lose weight over
   time, and a topic can be explicitly forgotten.
3. **A system, not a notebook.** It runs as a service with health checks, metrics, autoscaling and an automated build/deploy
   pipeline, using only local models and free APIs (no paid LLM key).

## Architecture

```
                 Gradio Space on Hugging Face  |  Streamlit app (local)
   Paper Discovery | PDF Ingestion | RAG Q&A | Author Discovery | Memory History
                                      |  HTTP  (/api/v1/...)
                                      v
+-------------------------------------------------------------------------------+
|  FastAPI backend  (Kubernetes: 2-10 pods, NodePort 30800)  /health /metrics    |
|                                                                               |
|  routers:  papers  embeddings  rag  memory  pipeline  report                  |
|                                                                               |
|  RAGService --> LLMService.rewrite_query --> EmbeddingService --> VectorDB    |
|      |                                                     |                  |
|      |            hybrid retrieval (paper-specific + general)                  |
|      v                                                                        |
|  LLMService.generate_long_answer (multi-pass)      MemoryService              |
+------+----------------+-----------------+----------------+--------------------+
       |                |                 |                |
       v                v                 v                v
  Ollama (local)     ChromaDB          DynamoDB        arXiv + Semantic Scholar
  Qwen 3.5 35B       research_papers   ResearchPapers  (paper search, authors)
                     adaptive_memory   UserSessions
                                       user_sessions

  Monitoring hub (nginx :9999) -> Grafana  Prometheus  MLflow  API docs
```

### What happens on `POST /rag/query`

1. **Load memory first.** With `use_memory` and a `session_id`, the last 6 turns are read (chronologically, from DynamoDB) so the
   rewrite step can see them.
2. **Rewrite the query.** One few-shot LLM call returns five fields: `SEARCH` (broad keywords for paper discovery), `RETRIEVE`
   (terms that would appear *inside* the relevant section, e.g. "W0 BA low-rank decomposition delta W"), `QUESTION` (a
   well-formed question), `PAPER` (a named paper such as LoRA or DDPM, or `NONE`) and `TITLE` (that paper's official title).
   Conversation history is used to expand short follow-ups. If the call fails, the raw query is used throughout. On the
   benchmark, embedding `RETRIEVE` instead of the raw query raises Recall@5 from 0.31 to 0.75.
3. **Auto-ingest.** If a specific paper is named and it is not in DynamoDB, the pipeline searches arXiv (by its title) and ingests
   up to 3 papers before answering. The lookup matches the official `TITLE` first (fuzzy, case- and punctuation-insensitive) and
   falls back to a whole-word match on `PAPER`, because short names are often absent from titles ("DDPM") or ambiguous
   ("Transformer" appears in the BERT and ViT titles). A failure here is logged and does not fail the request.
4. **Paper-scoped + general retrieval.** A general ChromaDB search always runs. If a paper was named and found, a second search
   filtered to that paper's chunks runs too. Paper-specific chunks come first, duplicates are dropped, and the merged list is
   capped at 12. (This is metadata-filtered + unfiltered dense search, not the BM25 + dense "hybrid search" of the IR literature.)
5. **Multi-pass generation.** The prompt contains the numbered chunks, the conversation history and formatting rules
   (LaTeX equations, each written once). Each Ollama call is capped at 1,024 new tokens (8,192-token context, temperature 0.3).
   If the answer does not end on sentence-final punctuation, up to two continuation passes resume from the last 600 characters.
6. **Write back.** The user turn and the answer are stored in memory. This step is best-effort: a storage failure is logged and the
   answer is still returned.
7. **Instrument.** Retrieval, generation and total latency go to Prometheus histograms and MLflow.

`POST /report/generate` uses the same retrieval (top 15 chunks) and a longer continuation loop (up to 50 passes, budgeted at about
700 words per pass) to produce a Markdown report with Summary, Key Findings, Mathematical Foundations, Methodology, Conclusion and
References sections.

### Embeddings and the MLP layers

Text is encoded with the pretrained `sentence-transformers/all-MiniLM-L6-v2` (384-dim, L2-normalised) and stored as is.

The project's original design also passed these vectors through two small untrained PyTorch modules, `EncoderMLP` and
`SinusoidalMLP` (`backend/models/mlp_models.py`). They were meant to be close to the identity, but the `LayerNorm` inside
`EncoderMLP` rescales its hidden layer, so its residual update had about the same norm as the embedding itself (cosine 0.70
between raw and projected vectors). The weights were also freshly random in every process, so two pods, or one pod before and
after a restart, embedded the same text with cosine similarity of only about 0.5: chunks stored by one process were searched
with queries from a different vector space. On the benchmark they also lowered mean Recall@5 by about 7 points, with an
11-point swing depending on the random draw. They are therefore no longer applied; the modules remain as a hook for contrastive fine-tuning with saved, versioned weights.

### Adaptive memory

Every turn is written twice: the full history goes to DynamoDB (partition key `session_id`, sort key `entry_id`), and an embedded
copy goes to the ChromaDB `adaptive_memory` collection with a `stored_at` timestamp. `entry_id` is `<ISO timestamp>#<random suffix>`,
so the sort key orders a session's turns by time and "the last 6 turns" really are the most recent ones.

- **Recency decay.** Semantic search over memory (`/memory/retrieve` with a `query`) is filtered to the session, fetches 3x the
  requested entries and re-ranks
  them by `score = similarity * exp(-lambda * days)` with `lambda = 0.05`, a half-life of about 14 days. `days` is the time since the
  entry was *stored*; access is not tracked. Chronological retrieval, which the RAG pipeline uses for its 6-turn context, is not
  decayed.
- **Explicit forgetting.** `POST /memory/forget` embeds the topic, takes the session's `top_k` (default 20) most similar memory
  entries and deletes those with similarity above 0.5 from ChromaDB, so they no longer surface in semantic
  search. The DynamoDB history row is not deleted. The 0.5 threshold is a chosen default, not a tuned value.
- **Sessions.** A per-user session registry (`user_sessions` table) supports listing, renaming and deleting sessions.

### Ingestion and paper discovery

- **PDFs.** PyMuPDF extracts text block by block, tags section headers and flags math-heavy blocks (regex heuristics for LaTeX-like
  content). On any PyMuPDF error it falls back to pdfplumber's plain text. There is no OCR, so scanned PDFs yield little or no text.
- **Chunking.** Sliding character windows of 1,024 characters with 128 overlap (the config values 256 and 32 are "tokens" at an
  approximate 4 characters each), snapped to a sentence boundary. MiniLM reads at most 256 word pieces; the earlier 2,048-character
  chunks were a median 455 word pieces, so 45% of the corpus text was never embedded. At 1,024 characters that falls to 5.5%, and
  Recall@5 on the production path rises from 0.69 to 0.78. Chunk IDs are deterministic and written with `upsert`, so re-ingesting a paper is idempotent.
- **Discovery.** arXiv and Semantic Scholar are searched concurrently; if one fails the other's results are still returned.
  Results are de-duplicated by exact lower-cased title. `GET /papers/author` returns a Semantic Scholar author profile (h-index,
  citation count) and top papers.

## Data stores

| Store | Name | Holds |
|---|---|---|
| ChromaDB (cosine, 384-dim) | `research_papers` | paper chunks with `paper_id`, title, chunk index |
| ChromaDB (cosine, 384-dim) | `adaptive_memory` | embedded conversation turns with `session_id`, role, `stored_at` |
| DynamoDB | `ResearchPapers` (key `paper_id`) | paper metadata |
| DynamoDB | `UserSessions` (key `session_id` + time-ordered `entry_id`) | full conversation history |
| DynamoDB | `user_sessions` | per-user session registry and message counts |

ChromaDB connects to the server (`CHROMA_HOST:CHROMA_PORT`) and, if it is unreachable, falls back to a local persistent client in
`data/embeddings/chroma`. DynamoDB runs as `amazon/dynamodb-local` by default; changing `DYNAMODB_ENDPOINT_URL` and the AWS
credentials points the same code at real AWS.

## API

Routers are mounted under `/api/v1`. Operational endpoints are at the root.

| Method | Endpoint | Description |
|---|---|---|
| POST | `/papers/search` | Search arXiv and Semantic Scholar |
| POST | `/papers/ingest` | Ingest a paper from text |
| POST | `/papers/ingest/pdf` | Ingest a paper from a PDF upload |
| GET | `/papers/author` | Author profile and top papers (Semantic Scholar) |
| POST | `/pipeline/search-and-ingest` | Search, embed and store in one call |
| POST | `/pipeline/ingest-by-id` | Ingest a single arXiv paper by ID |
| POST | `/embeddings/query` | Semantic similarity search over chunks |
| POST | `/rag/query` | Full RAG pipeline with citations |
| POST | `/report/generate` | Long-form research report |
| POST | `/memory/store` | Store a conversation turn |
| POST | `/memory/retrieve` | Chronological or semantic (decay re-ranked) history |
| POST | `/memory/forget` | Forget a topic |
| POST | `/memory/sessions` | Register a session |
| GET | `/memory/sessions/{user_id}` | List a user's sessions |
| PUT | `/memory/sessions/{user_id}/{session_id}/rename` | Rename a session |
| DELETE | `/memory/sessions/{user_id}/{session_id}` | Delete a session |

Root endpoints: `/health`, `/status`, `/metrics`, `/prometheus` (Prometheus scrape target), `/stats`, `/papers-list`, `/users`,
`/users/details`. Interactive docs are at `/docs`.

## MLOps

**Containers.** The root `Dockerfile` (python:3.12-slim, dependencies in a cached layer, no `.env` baked in) is what CI/CD builds.
`docker-compose.yml` starts the supporting services: ChromaDB (8001), DynamoDB Local (8002), MLflow (5000), Prometheus (9090),
Grafana (3000) and an nginx monitoring hub (9999). The backend itself runs with `uvicorn` on the host or in Kubernetes.

**Kubernetes** (`k8s/`, namespace `researchcopilot`; CD targets a `kind` cluster, see `k8s/kind-config.yaml`).
- 2 replicas, each requesting 0.5 CPU / 1 GiB and limited to 2 CPU / 4 GiB, with readiness and liveness probes on `/health`.
- Exposed as a NodePort on 30800.
- A `HorizontalPodAutoscaler` scales 2 to 10 pods at 70% CPU or 80% memory, with a 30 s scale-up stabilisation window.
- Configuration comes from a ConfigMap (`.env` is not in the image). Ollama listens only on `127.0.0.1`, which pods cannot reach, so
  `k8s/deploy.sh` starts a small TCP forwarder (`0.0.0.0:11436` to `127.0.0.1:11434`) and the ConfigMap points
  `OLLAMA_BASE_URL` at it through the kind network gateway.

**CI** (`.github/workflows/ci.yml`), on every push and on pull requests to `main`: Python 3.12, `flake8` on `backend/`, then `pytest`.

**CD** (`.github/workflows/cd.yml`), on push to `main`:
1. Build the image and push it to Docker Hub tagged `latest` and with the commit SHA.
2. On a self-hosted runner (the machine that hosts the kind cluster), pull and `kind load` the image, apply the manifests, and run a
   rolling restart.
3. Smoke test: `curl /health` must return `"status":"ok"`, otherwise the job fails.

**Observability.**
- *Prometheus* scrapes `/prometheus` every 15 s: HTTP request counts and latency by method, endpoint and status; RAG query counts;
  retrieval, generation and end-to-end duration histograms; sources retrieved per query; and ingestion metrics.
- *Grafana* ships one provisioned dashboard, "ResearchCopilot - Backend Overview": API rate, errors and P50/P95/P99 latency; RAG
  duration and retrieval-vs-generation time; GPU memory, utilisation, temperature and power (read from `nvidia-smi`); CPU and RAM;
  and paper ingestion. No alert rules are defined.
- *MLflow* logs each ingestion (chunks, embeddings, duration), each RAG query (chunks, retrieval/generation/total ms) and each embedding
  batch.
- The nginx hub (port 9999) puts the API docs, Grafana, Prometheus and MLflow behind one address and serves a small stats page.

**Data versioning.** `params.yaml` holds the chunking, embedding, retrieval and LLM settings, and `dvc.yaml` declares an ingestion
stage over `data/papers/` and `data/embeddings/` keyed on those parameters. Ingestion itself runs through the API, so the stage is a
tracking declaration rather than an executable pipeline.

## Evaluation

`eval/run_eval.py` runs the project's own parsing, chunking, embedding, ChromaDB, query-rewrite, retrieval and generation code
against a hand-labelled benchmark (`eval/questions.json`): **32 questions over 12 arXiv papers** (Transformer, BERT, LoRA, DDPM,
ResNet, RAG, GAN, ViT, plus Adam, BatchNorm, Sentence-BERT and MiniLM as distractors). Questions are written the way users type
("ddpm noise schedule values"). Each one is labelled with its paper and an *evidence phrase*, a short string checked to occur in the
extracted text (e.g. "random Gaussian initialization for A and zero for B"). A retrieved chunk counts as relevant if it comes from
the right paper and contains that phrase, so the labels do not depend on how the text is chunked. Full results are in
[eval/results/report.md](eval/results/report.md); the run is deterministic given the cached query rewrites.

```bash
python eval/run_eval.py --model qwen3.8:latest            # downloads the papers on first run; ~1 h with generation
python eval/run_eval.py --model qwen3.8:latest --skip-generation   # retrieval only, a few minutes
```

**Retrieval** (Recall@5: an evidence chunk is in the top 5; production path = rewritten query + paper-scoped retrieval):

| Change | Recall@5 | Recall@10 |
|---|---:|---:|
| Before: raw query, 2,048-char chunks, untrained MLP (mean of 5 seeds) | 0.375 | 0.512 |
| Rewritten + paper-scoped with the old 2,048-char chunks and untrained MLP (mean of 5 seeds) | 0.625 | 0.706 |
| Raw query, current embedding and chunking | 0.312 | 0.562 |
| + LLM query rewriting (`RETRIEVE` line) | 0.750 | 0.875 |
| + paper-scoped retrieval (current production path) | **0.781** | **0.875** |

- Query rewriting is the largest single gain (+44 points of Recall@5 over the raw query). Short queries rarely use the wording of
  the passage that answers them; the rewrite describes what the answer text looks like.
- Paper lookup now resolves all 25 questions in which the rewrite names a paper to exactly the right paper. In an earlier run
  with the old substring lookup, 8 of 23 resolved correctly: DDPM, ResNet, RAG, GAN and ViT never matched their titles (which
  triggers a needless auto-ingest on every such query), and "Transformer" matched the BERT, ViT and MiniLM papers.
- ChromaDB's default HNSW `search_ef` of 10 returned only about 92% of the true top 10, differently on each index build; it is now 100,
  which matches exact search.

**Generation** (all 32 questions, production path, `qwen3.8` 27B via Ollama, graded against a hand-written reference answer):

| Metric | RAG | Same model, no retrieval |
|---|---:|---:|
| Mean correctness (1-5) | **4.88** | 3.91 |
| Answers scored 4 or 5 | **94%** | 69% |
| Answers scored 1 | 0 | 5 |
| Answers citing `[Source N]`, all citations in range | 100% | n/a |

Every closed-book failure was a paper-specific detail (which matrices LoRA adapts, RAG's generator and its 21M-passage index, the
GAN optimum result), which is the case retrieval is for; well-known facts such as DDPM's FID of 3.17 were answered from memory
either way. Latency per question: retrieval p50 27 ms; generation p50 70 s and p95 129 s on the 27B model with up to three passes,
so generation is effectively all of the end-to-end latency.

*Limitations of this evaluation.* The grader is the same local model, at temperature 0, so absolute scores carry self-preference
bias; the RAG vs closed-book gap uses the same grader for both and is the more reliable signal. Its groundedness score (are the
claims supported by the retrieved chunks) was 5/5 for every answer, so it does not discriminate and is not reported as a result;
for example, one RAG answer quoted a wrong parameter-reduction ratio and was scored 3 for correctness but still 5 for
groundedness. With 32 questions, one question is about 3 points of Recall@5. The generation runs were made before the `search_ef`
fix.

## Testing

71 pytest tests in five files, written day by day as the system was built. Ollama, DynamoDB and the ChromaDB server are replaced with
`unittest.mock`, so no running services are needed.

| File | Tests | Covers |
|---|---:|---|
| `tests/test_day1.py` | 13 | schemas, MLP modules, chunking |
| `tests/test_day2.py` | 12 | embeddings, vector DB, ingest pipeline |
| `tests/test_day3.py` | 13 | RAG, memory, report generation |
| `tests/test_day4.py` | 23 | API endpoints and the route inventory |
| `tests/test_day5.py` | 10 | regressions: time-ordered memory, session isolation, paper lookup, deterministic embeddings, chunk size |

These tests check the service plumbing (contracts, fallbacks, routing). Answer quality is measured separately by the
[evaluation](#evaluation), which needs Ollama and is not part of CI.

```bash
pytest tests/ -v
```

## Caveats

- **Small evaluation.** 32 hand-labelled questions over 12 papers: one question moves Recall@5 by about 3 points, and query rewriting
  is sampled at temperature 0.3, so differences of a few points are within noise. Generation is graded by the same local model that
  produced the answers. There is no load test behind the autoscaling settings.
- **No authentication.** The API has no auth layer, and `session_id` / `user_id` are trusted from the request, so one caller can read
  another session's history. It is meant for a private network. Uploaded PDFs are parsed without sandboxing.
- **One LLM instance.** All pods share a single Ollama server, so adding pods scales the API layer but not generation. If Ollama is
  unreachable, `/rag/query` returns a notice explaining how to start it instead of an answer; there is no retry or queue.
- **Re-ingest after upgrading.** Vectors stored before the embedding and chunking fixes came from a random projection and 2,048-character
  chunks; clear the `research_papers` and `adaptive_memory` collections and re-ingest. History entries written with the old random
  `entry_id` do not sort by time.
- **Memory decay is recency-based.** It uses time since storage, not time since last use, and only affects semantic memory search,
  not the 6-turn context given to the LLM. Forgetting removes the semantic copy, not the DynamoDB history.
- **Discovery is simple.** De-duplication is by exact title; there is no OCR for scanned PDFs, and PyMuPDF garbles many equations
  (subscripts and bars become loose characters); the query-rewrite prompt is tuned with a handful of examples, and a wrong
  `PAPER`/`TITLE` guess can trigger an unnecessary auto-ingest.
- **Multi-pass generation is a heuristic.** Completeness is judged by the final character of the text, so a truncated answer that
  happens to end in a period is treated as finished.
- **Hardware.** The default model is a 35B-parameter Qwen served locally, so the LLM is the heaviest dependency and needs a large-memory
  GPU. Change `OLLAMA_MODEL` to use a smaller model.

## Try it live

**Live demo: https://huggingface.co/spaces/m25csa001/ResearchCopilot**

The Space is a Gradio front end (`huggingface/app.py`) with five tabs: Paper Discovery (chat that finds and ingests papers),
PDF Ingestion (upload or paste text), RAG Q&A (with a top-k slider), Author Discovery, and Memory History (chronological view,
semantic search and Forget Topic). It only renders the UI; every request goes to the backend named by the `BACKEND_URL` Space secret.

The backend and the LLM run on the author's own machine, not on Hugging Face. **If that machine is off or unreachable, the Space still
loads but shows "backend unreachable".** Free Spaces also go to sleep after a period of inactivity, so the first visit can take a minute.

## Run locally

```bash
git clone https://github.com/yadityax/ResearchCopilot && cd ResearchCopilot
pip install -r requirements.txt            # Python 3.12 is what CI and the image use
cp .env.example .env                       # edit as needed

ollama pull qwen3.5:35b                    # or set OLLAMA_MODEL to a smaller model

bash scripts/start_local.sh                # ChromaDB, MLflow, backend, Streamlit
```

Or start the pieces yourself:

```bash
# supporting services (ChromaDB :8001, DynamoDB Local :8002, MLflow :5000, Prometheus, Grafana, hub :9999)
docker compose up -d

uvicorn backend.main:app --reload          # backend on :8000  (docs at /docs)
streamlit run frontend/app.py              # UI on :8501  (set API_BASE_URL if the backend is elsewhere)
```

Seed the database with papers:

```bash
python scripts/ingest_arxiv.py --query "BERT language model" --max 5
python scripts/ingest_arxiv.py --id 1706.03762                 # Attention Is All You Need
python scripts/ingest_foundational_papers.py --dry-run         # bulk list of foundational ML/CV papers
python scripts/demo.py --topic "transformer attention mechanism" --query "What is self-attention?"
```

| Service | URL |
|---|---|
| Streamlit UI | http://localhost:8501 |
| API docs | http://localhost:8000/docs |
| API status | http://localhost:8000/status |
| MLflow | http://localhost:5000 |
| ChromaDB | http://localhost:8001 |
| Monitoring hub | http://localhost:9999 |

Paper text is fetched at run time from the public arXiv and Semantic Scholar APIs (an optional `SEMANTIC_SCHOLAR_API_KEY` raises the
rate limit) and stored locally in `data/`; no paper content is included in this repository.

## Deploying

```bash
# Manual Kubernetes deploy: starts the Ollama forwarder, builds the image, imports it into k3s,
# applies k8s/*.yaml and waits for the pods (CD does the equivalent for a kind cluster)
bash k8s/deploy.sh

# Hugging Face Space: upload the contents of huggingface/ and set the BACKEND_URL secret
# (Space settings -> Secrets) to the public URL of the backend
```

Pushing to `main` runs the CD workflow, which requires the repository secrets `DOCKERHUB_USERNAME` and `DOCKERHUB_TOKEN` and a
registered self-hosted runner on the cluster machine. `k8s/deploy.sh` contains machine-specific paths and IP addresses and will need
editing for another host.

## Repository layout

```
backend/
  main.py                 FastAPI app, root endpoints (/health, /stats, /users, ...), router mounting
  config.py               Pydantic settings (env / .env)
  models/schemas.py       request and response models
  models/mlp_models.py    EncoderMLP, SinusoidalMLP
  routers/                papers, embeddings, rag, memory, reports, ingest_pipeline
  services/
    rag_service.py        query -> rewrite -> hybrid retrieval -> generation -> memory
    llm_service.py        Ollama client, query rewriting, multi-pass generation, prompts
    embedding_service.py  MiniLM + MLP projection
    vector_db.py          ChromaDB client with local fallback
    dynamo_db.py          papers, sessions and user-session tables
    memory_service.py     dual-write memory, recency decay, forgetting
    pdf_ingestion.py      PyMuPDF / pdfplumber parsing, section tagging, chunking
    paper_discovery.py    arXiv + Semantic Scholar search, author lookup
    ingest_pipeline.py    search -> parse -> embed -> store
    mlflow_service.py     experiment logging
    prom_metrics.py       Prometheus metric definitions
    monitoring_service.py CPU / RAM / GPU stats
frontend/                 Streamlit app (app.py, pages/)
huggingface/              Gradio Space (app.py, help.md, README.md)
k8s/                      namespace, configmap, deployment, service, hpa, kind config, deploy.sh
monitoring/               prometheus.yml, Grafana provisioning + dashboard, nginx hub
docker/, Dockerfile       container builds (the root Dockerfile is used by CD)
docker-compose.yml        supporting services
.github/workflows/        ci.yml, cd.yml
tests/                    71 tests (test_day1 ... test_day5)
eval/                     questions.json (labelled benchmark), run_eval.py, results/ (report.md and raw outputs)
scripts/                  start_local.sh, ingest_arxiv.py, ingest_foundational_papers.py, demo.py, init_db.py
params.yaml, dvc.yaml     tracked pipeline parameters
```

## References

- P. Lewis et al., *Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks*, NeurIPS 2020 (arXiv:2005.11401).
- N. Reimers, I. Gurevych, *Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks*, EMNLP 2019 (arXiv:1908.10084).
- W. Wang et al., *MiniLM: Deep Self-Attention Distillation for Task-Agnostic Compression of Pre-Trained Transformers*, NeurIPS 2020
  (arXiv:2002.10957).
- Tools: [Ollama](https://ollama.com), [ChromaDB](https://www.trychroma.com), [FastAPI](https://fastapi.tiangolo.com),
  [arXiv API](https://info.arxiv.org/help/api/), [Semantic Scholar API](https://www.semanticscholar.org/product/api),
  [MLflow](https://mlflow.org), [Gradio](https://www.gradio.app).
