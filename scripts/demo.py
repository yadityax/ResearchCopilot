"""
ResearchCopilot — End-to-End Demo Script
Runs the full pipeline:
  1. Health check
  2. Search arXiv for papers
  3. Ingest papers (abstract fallback)
  4. Query embeddings to verify storage
  5. RAG Q&A
  6. Generate a research report
  7. Store + retrieve memory

Usage:
    # Backend must be running first:
    uvicorn backend.main:app --reload

    python scripts/demo.py
    python scripts/demo.py --topic "BERT language model" --query "What is masked language modeling?"
"""
import sys, os, argparse, asyncio, json, time
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import httpx
from loguru import logger

BASE = "http://localhost:8000/api/v1"
HEADERS = {"Content-Type": "application/json"}


# ── HTTP helpers ───────────────────────────────────────────────────────────

async def post(client: httpx.AsyncClient, endpoint: str, payload: dict, timeout=60) -> dict:
    r = await client.post(f"{BASE}{endpoint}", json=payload, timeout=timeout)
    r.raise_for_status()
    return r.json()


async def get(client: httpx.AsyncClient, path: str) -> dict:
    r = await client.get(f"http://localhost:8000{path}", timeout=10)
    r.raise_for_status()
    return r.json()


def _sep(title=""):
    width = 70
    if title:
        pad = (width - len(title) - 2) // 2
        print(f"\n{'─'*pad} {title} {'─'*pad}")
    else:
        print("─" * width)


def _ok(msg):   print(f"  ✓  {msg}")
def _warn(msg): print(f"  ⚠  {msg}")
def _info(msg): print(f"     {msg}")


# ── Demo steps ─────────────────────────────────────────────────────────────

async def run(topic: str, question: str):
    async with httpx.AsyncClient() as client:

        # ── Step 1: Health ─────────────────────────────────────────────────
        _sep("1. Health Check")
        health = await get(client, "/health")
        _ok(f"Backend online  — {health['app']} v{health['version']}")

        status = await get(client, "/status")
        for svc, st in status["services"].items():
            fn = _ok if st == "online" else _warn
            fn(f"{svc}: {st}")
        _info(f"Embedding model : {status['config']['embedding_model']}")
        _info(f"LLM model       : {status['config']['llm_model']}")

        # ── Step 2: Paper search ───────────────────────────────────────────
        _sep("2. Paper Discovery (arXiv)")
        t0 = time.perf_counter()
        search = await post(client, "/papers/search", {
            "query": topic, "source": "arxiv", "max_results": 4, "year_from": 2020
        }, timeout=30)
        elapsed = (time.perf_counter() - t0) * 1000
        _ok(f"Found {search['total_found']} papers in {elapsed:.0f}ms")
        papers = search["papers"]
        for p in papers[:3]:
            _info(f"  [{p['source'].upper()}] {p['title'][:70]}")

        if not papers:
            _warn("No papers found — skipping ingest. Try a different topic.")
            return

        # ── Step 3: Ingest via pipeline ────────────────────────────────────
        _sep("3. Auto-Ingest Pipeline")
        t0 = time.perf_counter()
        ingest_results = await post(client, "/pipeline/search-and-ingest", {
            "query": topic, "max_papers": 3, "source": "arxiv"
        }, timeout=180)
        elapsed = (time.perf_counter() - t0) * 1000
        ok_count = sum(1 for r in ingest_results if r["status"] == "success")
        _ok(f"Ingested {ok_count}/{len(ingest_results)} papers in {elapsed:.0f}ms")
        for r in ingest_results:
            icon = "✓" if r["status"] == "success" else "✗"
            _info(f"  {icon} {r['paper_id'][:35]:<35}  chunks={r['chunks_created']:>3}  embeds={r['embeddings_stored']:>3}")

        # ── Step 4: Embedding query ────────────────────────────────────────
        _sep("4. Vector Similarity Search")
        t0 = time.perf_counter()
        emb_result = await post(client, "/embeddings/query", {
            "query": question, "top_k": 3
        }, timeout=30)
        elapsed = (time.perf_counter() - t0) * 1000
        _ok(f"Retrieved {emb_result['total_results']} chunks in {elapsed:.0f}ms")
        for chunk in emb_result["results"]:
            _info(f"  score={chunk['score']:.3f}  paper={chunk['paper_id'][:30]}")
            _info(f"    \"{chunk['text'][:80]}...\"")

        # ── Step 5: RAG Q&A ────────────────────────────────────────────────
        _sep("5. RAG Q&A")
        session_id = "demo_session_001"
        t0 = time.perf_counter()
        rag_result = await post(client, "/rag/query", {
            "query": question,
            "session_id": session_id,
            "top_k": 3,
            "use_memory": True,
        }, timeout=120)
        elapsed = (time.perf_counter() - t0) * 1000
        _ok(f"Answer generated in {elapsed:.0f}ms  (model: {rag_result['model_used']})")
        _info(f"  Sources: {len(rag_result['sources'])}")
        print()
        print("  ANSWER:")
        for line in rag_result["answer"].split("\n"):
            print(f"    {line}")

        # ── Step 6: Report generation ──────────────────────────────────────
        _sep("6. Research Report Generation")
        t0 = time.perf_counter()
        report = await post(client, "/report/generate", {
            "topic": topic,
            "session_id": session_id,
            "max_length": 500,
            "format": "markdown",
        }, timeout=180)
        elapsed = (time.perf_counter() - t0) * 1000
        _ok(f"Report generated in {elapsed:.0f}ms  ({len(report['sources_used'])} sources)")
        print()
        # Print first 600 chars of report
        preview = report["report"][:600]
        for line in preview.split("\n"):
            print(f"    {line}")
        if len(report["report"]) > 600:
            print("    ...")

        # ── Step 7: Adaptive memory ────────────────────────────────────────
        _sep("7. Adaptive Memory")
        store = await post(client, "/memory/store", {
            "session_id": session_id,
            "role": "user",
            "content": question,
        }, timeout=30)
        _ok(f"Stored memory entry: {store.get('entry_id', '')[:20]}...")

        retrieve = await post(client, "/memory/retrieve", {
            "session_id": session_id, "limit": 5
        }, timeout=20)
        _ok(f"Retrieved {retrieve['total']} memory entries for session")

        _sep("Demo Complete")
        print()
        _ok("Full pipeline executed successfully!")
        _info("Next steps:")
        _info("  • Open http://localhost:8501 for the Streamlit UI")
        _info("  • Open http://localhost:8000/docs for the API docs")
        _info("  • Open http://localhost:5000 for the MLflow dashboard")
        print()


def main():
    parser = argparse.ArgumentParser(description="ResearchCopilot E2E Demo")
    parser.add_argument("--topic",    "-t", default="transformer attention mechanism",
                        help="Research topic to search and ingest")
    parser.add_argument("--question", "-q", default="What is the attention mechanism in transformers?",
                        help="Question to ask the RAG system")
    args = parser.parse_args()

    print()
    print("╔══════════════════════════════════════════════════════════════╗")
    print("║          ResearchCopilot — End-to-End Demo                  ║")
    print("╚══════════════════════════════════════════════════════════════╝")
    print(f"  Topic   : {args.topic}")
    print(f"  Question: {args.question}")
    print()

    try:
        asyncio.run(run(args.topic, args.question))
    except httpx.ConnectError:
        print("\n✗  Cannot connect to backend.")
        print("   Run:  uvicorn backend.main:app --reload")
        sys.exit(1)
    except KeyboardInterrupt:
        print("\nDemo interrupted.")


if __name__ == "__main__":
    main()
