"""
ResearchCopilot evaluation harness.

Measures retrieval and answer quality on a labelled question set (eval/questions.json) built over
12 arXiv papers (8 with questions, 4 distractors). It runs the project's own code paths:
PDFIngestionService (parsing + chunking), EmbeddingService, VectorDBService (ChromaDB, in-memory),
LLMService.rewrite_query, RAGService._hybrid_retrieve and LLMService.generate_long_answer.

Retrieval (no LLM needed except for query rewriting, which is cached):
  * evidence Recall@5  - a chunk from the gold paper that contains the gold evidence phrase is in the top 5
  * paper Hit@5        - any chunk from the gold paper is in the top 5
  * MRR@10             - reciprocal rank of the first evidence chunk
  ablations: chunk size, MLP projection on/off, raw vs rewritten query, general vs paper-scoped retrieval

Generation (LLM-as-judge against a reference answer, same local model, temperature 0):
  * correctness 1-5 for RAG answers and for a closed-book baseline (same model, no retrieval)
  * groundedness 1-5 (are the RAG answer's claims supported by the retrieved chunks)
  * citation validity  (answer cites [Source N] and every N exists)

Usage (from the repo root):
  python eval/run_eval.py                      # retrieval + generation
  python eval/run_eval.py --skip-generation    # retrieval only
  python eval/run_eval.py --model qwen3.8:latest
"""
import argparse
import asyncio
import json
import os
import re
import statistics
import sys
import time
import unicodedata
from pathlib import Path

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")  # embeddings on CPU, same as CI

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import chromadb  # noqa: E402
import httpx  # noqa: E402
import torch  # noqa: E402
from loguru import logger  # noqa: E402

from backend.config import Settings  # noqa: E402
from backend.models.mlp_models import EncoderMLP, SinusoidalMLP  # noqa: E402
from backend.services.dynamo_db import DynamoDBService  # noqa: E402
from backend.services.embedding_service import EmbeddingService  # noqa: E402
from backend.services.llm_service import LLMService  # noqa: E402
from backend.services.mlflow_service import MLflowService  # noqa: E402
from backend.services.pdf_ingestion import PDFIngestionService  # noqa: E402
from backend.services.rag_service import RAGService  # noqa: E402
from backend.services.vector_db import VectorDBService  # noqa: E402

EVAL_DIR = ROOT / "eval"
PDF_DIR = EVAL_DIR / "data" / "pdfs"
RESULTS_DIR = EVAL_DIR / "results"

# (chunk_size, chunk_overlap) in the config's "tokens"; the chunker multiplies by 4 to get characters
CHUNK_CONFIGS = [(512, 64), (256, 32)]
EMBEDDER_MAX_TOKENS = 256  # all-MiniLM-L6-v2 max_seq_length
MLP_SEEDS = range(5)       # random initialisations of the old untrained projection
QUERY_MODES = ("raw", "rewritten", "paper_scoped")
METRICS = ("recall@5", "recall@10", "paper_hit@5", "mrr@10")

# Keep the eval from writing thousands of runs to a real MLflow server
MLflowService.log_embedding_batch = lambda *a, **k: None
MLflowService.log_rag_query = lambda *a, **k: None
logger.remove()
logger.add(sys.stderr, level="WARNING")


def norm(s: str) -> str:
    """Case-, whitespace- and ligature-insensitive form used for evidence matching."""
    return re.sub(r"\s+", "", unicodedata.normalize("NFKC", s).lower())


def is_evidence(chunk, q) -> bool:
    if chunk.paper_id != q["paper_id"]:
        return False
    text = norm(chunk.text)
    return any(norm(e) in text for e in q["evidence"])


# ── Corpus ─────────────────────────────────────────────────────────────────

def build_chunks(papers: dict, chunk_size: int, overlap: int) -> list:
    ingest = PDFIngestionService(Settings(chunk_size=chunk_size, chunk_overlap=overlap))
    chunks = []
    for paper_id, title in papers.items():
        pdf = PDF_DIR / f"{paper_id.removeprefix('arxiv_')}.pdf"
        if not pdf.exists():
            print(f"  downloading {pdf.stem} from arXiv")
            PDF_DIR.mkdir(parents=True, exist_ok=True)
            r = httpx.get(f"https://arxiv.org/pdf/{pdf.stem}", follow_redirects=True, timeout=120)
            r.raise_for_status()
            pdf.write_bytes(r.content)
            time.sleep(3)  # arXiv asks for a delay between automated requests
        text = ingest._clean_text(ingest._extract_text_from_pdf(str(pdf)))
        ingest._current_title = title
        chunks.extend(ingest._split_into_chunks(text, paper_id))
    return chunks


def project(vectors: list, seed: int | None) -> list:
    """
    seed=None: production embedding (MiniLM only). seed=k: the pre-fix behaviour, where an untrained
    EncoderMLP + SinusoidalMLP with fresh random weights sat on top of MiniLM in every process.
    Documents and queries of one index always share a seed, i.e. the best case of a single process.
    """
    if seed is None:
        return vectors
    torch.manual_seed(seed)
    enc, sin = EncoderMLP().eval(), SinusoidalMLP().eval()
    with torch.no_grad():
        return torch.nn.functional.normalize(sin(enc(torch.tensor(vectors))), dim=1).tolist()


async def index(chunks: list, doc_vectors: list, name: str) -> VectorDBService:
    vdb = VectorDBService(Settings(chroma_collection_papers=name))
    vdb._client = chromadb.EphemeralClient()
    try:
        vdb._client.delete_collection(name)  # recreate so the current collection metadata applies
    except Exception:
        pass
    await vdb.store_chunks(name, chunks, doc_vectors)
    return vdb


def truncation_stats(chunks: list, tokenizer) -> dict:
    lengths = [len(tokenizer.encode(c["text"])) for c in chunks]
    over = [n for n in lengths if n > EMBEDDER_MAX_TOKENS]
    lost = sum(n - EMBEDDER_MAX_TOKENS for n in over) / sum(lengths)
    return {
        "chunks": len(chunks),
        "median_tokens": int(statistics.median(lengths)),
        "pct_chunks_truncated": round(100 * len(over) / len(lengths), 1),
        "pct_tokens_never_embedded": round(100 * lost, 1),
    }


# ── Query rewriting (cached: it is the slow, non-deterministic part) ──────

async def get_rewrites(questions: list, llm: LLMService, model: str) -> dict:
    cache_path = RESULTS_DIR / f"rewrites_{model.replace(':', '_')}.json"
    cache = json.loads(cache_path.read_text()) if cache_path.exists() else {}
    for q in questions:
        if q["query"] not in cache:
            print(f"  rewriting: {q['query']}")
            cache[q["query"]] = await llm.rewrite_query(q["query"])
            cache_path.write_text(json.dumps(cache, indent=2, ensure_ascii=False))
    return cache


# ── Retrieval eval ─────────────────────────────────────────────────────────

class _TitleIndex:
    """Stands in for the DynamoDB papers table; matching uses the production DynamoDBService.match_papers."""

    def __init__(self, papers: dict):
        self.items = [{"paper_id": pid, "title": t} for pid, t in papers.items()]

    async def find_papers(self, paper_name, paper_title=None) -> list:
        return DynamoDBService.match_papers(self.items, paper_name, paper_title)


def score(ranked: list, q: dict) -> dict:
    ev_rank = next((i + 1 for i, c in enumerate(ranked[:10]) if is_evidence(c, q)), None)
    return {
        "recall@5": ev_rank is not None and ev_rank <= 5,
        "recall@10": ev_rank is not None,
        "paper_hit@5": any(c.paper_id == q["paper_id"] for c in ranked[:5]),
        "mrr@10": 1 / ev_rank if ev_rank else 0.0,
    }


async def retrieve(mode: str, q: dict, rw: dict, emb_svc, vdb, rag: RAGService, name: str, seed=None) -> list:
    query = q["query"] if mode == "raw" else rw["retrieval_query"]
    emb = project([await emb_svc.embed_single(query)], seed)[0]
    if mode == "paper_scoped":
        rag._vector_db = vdb
        rag.settings = Settings(chroma_collection_papers=name)
        return await rag._hybrid_retrieve(emb, rw.get("paper_name"), top_k=10, filter_paper_ids=None,
                                          paper_title=rw.get("paper_title"))
    return await vdb.query_similar(name, emb, top_k=10)


async def run_retrieval(data: dict, rewrites: dict, emb_svc: EmbeddingService) -> tuple:
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(Settings().embedding_model)

    rag = RAGService(Settings())
    rag._dynamo = _TitleIndex(data["papers"])

    rows, per_question, chunk_stats, indexes = [], {}, {}, {}
    for cs, ov in CHUNK_CONFIGS:
        chunks = build_chunks(data["papers"], cs, ov)
        label = f"{cs * 4} chars"
        chunk_stats[label] = truncation_stats(chunks, tokenizer)
        print(f"  {label}: {chunk_stats[label]}")
        raw_vectors = await emb_svc.embed_batch([c["text"] for c in chunks])

        # seed=None is the production embedding; the seeds reproduce the old random MLP
        scores = {}  # (seed, mode) -> list of per-question scores
        for seed in [None, *MLP_SEEDS]:
            name = f"eval_{cs}_{'prod' if seed is None else f'mlp{seed}'}"
            vdb = await index(chunks, project(raw_vectors, seed), name)
            if seed is None:
                indexes[cs] = (vdb, name)
            for mode in QUERY_MODES:
                scores[(seed, mode)] = []
                for q in data["questions"]:
                    ranked = await retrieve(mode, q, rewrites[q["query"]], emb_svc, vdb, rag, name, seed)
                    s = score(ranked, q)
                    scores[(seed, mode)].append(s)
                    if seed is None:
                        per_question.setdefault(q["id"], {})[f"{label}|{mode}"] = s

        for mode in QUERY_MODES:
            prod = {m: round(statistics.mean(float(s[m]) for s in scores[(None, mode)]), 3) for m in METRICS}
            rows.append({"chunks": label, "embedding": "MiniLM", "query": mode, **prod})
            per_seed = [{m: statistics.mean(float(s[m]) for s in scores[(seed, mode)]) for m in METRICS}
                        for seed in MLP_SEEDS]
            mlp = {m: round(statistics.mean(r[m] for r in per_seed), 3) for m in METRICS}
            mlp["recall@5_range"] = (round(min(r["recall@5"] for r in per_seed), 3),
                                     round(max(r["recall@5"] for r in per_seed), 3))
            rows.append({"chunks": label, "embedding": f"MiniLM + untrained MLP ({len(MLP_SEEDS)} seeds)",
                         "query": mode, **mlp})
            print(f"    {mode:13s} MiniLM {prod}\n    {'':13s} +MLP   {mlp}")
    return rows, per_question, chunk_stats, indexes, rag


# ── Generation eval ────────────────────────────────────────────────────────

JUDGE_PROMPT = """You are a strict grader for answers about research papers.

Question: {question}

Reference answer (ground truth): {reference}

Candidate answer:
{answer}
{sources_block}
Grade the candidate. Return JSON only:
{{"correctness": <1-5: 5 = all key facts of the reference are present and correct, 3 = partly correct or missing key facts, 1 = wrong or missing>,
  "groundedness": <1-5 or null: only if sources are given. 5 = every factual claim is supported by the sources (claims the answer explicitly labels as outside knowledge do not count against it), 1 = mostly unsupported>,
  "reason": "<one sentence>"}}"""


async def judge(model: str, base_url: str, q: dict, answer: str, chunks: list | None) -> dict:
    sources_block = ""
    if chunks is not None:
        joined = "\n\n".join(f"[Source {i + 1}]\n{c.text}" for i, c in enumerate(chunks))
        sources_block = f"\nRetrieved sources the answer was supposed to use:\n{joined}\n"
    prompt = JUDGE_PROMPT.format(question=q["query"], reference=q["reference"], answer=answer,
                                 sources_block=sources_block)
    payload = {"model": model, "prompt": prompt, "stream": False, "think": False, "format": "json",
               "options": {"temperature": 0, "num_ctx": 16384, "num_predict": 300}}
    async with httpx.AsyncClient(timeout=300) as client:
        r = await client.post(f"{base_url}/api/generate", json=payload)
        r.raise_for_status()
    try:
        return json.loads(r.json()["response"])
    except json.JSONDecodeError:
        return {"correctness": None, "groundedness": None, "reason": "unparseable judge output"}


def citation_check(answer: str, n_sources: int) -> dict:
    cited = {int(n) for n in re.findall(r"Source\s+(\d+)", answer)}
    return {"cites_sources": bool(cited), "citations_valid": all(1 <= n <= n_sources for n in cited)}


async def run_generation(data, rewrites, emb_svc, vdb, name, rag, llm: LLMService, settings: Settings) -> list:
    cache_path = RESULTS_DIR / f"generation_{settings.ollama_model.replace(':', '_')}.json"
    done = {r["id"]: r for r in json.loads(cache_path.read_text())} if cache_path.exists() else {}
    for q in data["questions"]:
        if q["id"] in done:
            continue
        print(f"  generating {q['id']}: {q['query']}")
        rw = rewrites[q["query"]]
        t0 = time.perf_counter()
        chunks = await retrieve("paper_scoped", q, rw, emb_svc, vdb, rag, name)
        t1 = time.perf_counter()
        rag_answer = await llm.generate_long_answer(rw["answer_question"], chunks, max_passes=3)
        t2 = time.perf_counter()
        closed_book = await llm.generate(
            f"Answer this question about machine learning research concisely and accurately.\n\n"
            f"Question: {rw['answer_question']}\n\nAnswer:"
        )
        rag_grade = await judge(settings.ollama_model, settings.ollama_base_url, q, rag_answer, chunks)
        cb_grade = await judge(settings.ollama_model, settings.ollama_base_url, q, closed_book, None)
        done[q["id"]] = {
            "id": q["id"], "query": q["query"],
            "retrieval_ms": round((t1 - t0) * 1000), "generation_ms": round((t2 - t1) * 1000),
            "rag": {"answer": rag_answer, **rag_grade, **citation_check(rag_answer, len(chunks))},
            "closed_book": {"answer": closed_book, "correctness": cb_grade.get("correctness"),
                            "reason": cb_grade.get("reason")},
        }
        cache_path.write_text(json.dumps(list(done.values()), indent=2, ensure_ascii=False))
    return [done[q["id"]] for q in data["questions"]]


def summarize_generation(gen: list) -> dict:
    def mean(xs):
        xs = [x for x in xs if isinstance(x, (int, float))]
        return round(statistics.mean(xs), 2) if xs else None

    def pctl(xs, p):
        xs = sorted(xs)
        return xs[min(len(xs) - 1, int(p * len(xs)))]

    gen_ms = [g["generation_ms"] for g in gen]
    return {
        "n": len(gen),
        "rag_correctness": mean(g["rag"].get("correctness") for g in gen),
        "closed_book_correctness": mean(g["closed_book"].get("correctness") for g in gen),
        "rag_correct_pct(>=4)": round(100 * mean(float((g["rag"].get("correctness") or 0) >= 4) for g in gen)),
        "closed_book_correct_pct(>=4)": round(
            100 * mean(float((g["closed_book"].get("correctness") or 0) >= 4) for g in gen)),
        "rag_groundedness": mean(g["rag"].get("groundedness") for g in gen),
        "pct_answers_citing_sources": round(100 * mean(float(g["rag"]["cites_sources"]) for g in gen)),
        "pct_citations_valid": round(100 * mean(float(g["rag"]["citations_valid"]) for g in gen)),
        "retrieval_ms_p50": pctl([g["retrieval_ms"] for g in gen], 0.5),
        "generation_ms_p50": pctl(gen_ms, 0.5),
        "generation_ms_p95": pctl(gen_ms, 0.95),
    }


# ── Report ─────────────────────────────────────────────────────────────────

def write_report(rows, chunk_stats, per_question, data, rewrites, gen_summary, model):
    lines = [
        "# ResearchCopilot evaluation results",
        "",
        f"Generated by `eval/run_eval.py`. {len(data['questions'])} questions over {len(data['papers'])} arXiv papers "
        f"(8 with questions, 4 distractors). Query rewriting and generation use `{model}` via Ollama.",
        "",
        "## Chunking vs the embedder's 256-token input limit",
        "",
        "| Chunk size | Chunks | Median tokens | Chunks truncated | Tokens never embedded |",
        "|---|---:|---:|---:|---:|",
    ]
    for label, s in chunk_stats.items():
        lines.append(f"| {label} | {s['chunks']} | {s['median_tokens']} | {s['pct_chunks_truncated']}% | "
                     f"{s['pct_tokens_never_embedded']}% |")
    lines += [
        "",
        "## Retrieval",
        "",
        "`recall@k`: a chunk containing the labelled evidence is in the top k. `paper_hit@5`: any chunk from the right "
        "paper is in the top 5. `mrr@10`: mean reciprocal rank of the first evidence chunk. "
        "`raw` embeds the user's text as typed, `rewritten` embeds the LLM's RETRIEVE line, `paper_scoped` is the "
        "production path (rewritten query + paper-filtered search merged with general search). "
        "The untrained-MLP rows reproduce the pre-fix embedding (averaged over random initialisations, "
        "with the Recall@5 range across seeds).",
        "",
        "| Chunks | Embedding | Query | Recall@5 | Recall@10 | Paper hit@5 | MRR@10 |",
        "|---|---|---|---:|---:|---:|---:|",
    ]
    for r in rows:
        rng = f" ({r['recall@5_range'][0]:.2f}–{r['recall@5_range'][1]:.2f})" if "recall@5_range" in r else ""
        lines.append(f"| {r['chunks']} | {r['embedding']} | {r['query']} | {r['recall@5']:.3f}{rng} | "
                     f"{r['recall@10']:.3f} | {r['paper_hit@5']:.3f} | {r['mrr@10']:.3f} |")

    items = [{"paper_id": pid, "title": t} for pid, t in data["papers"].items()]
    lines += ["", "## Paper detection (PAPER / TITLE fields of the rewrite)", "",
              "`Resolved to` is what the production lookup (`DynamoDBService.match_papers`) returns; `ok` means it is "
              "exactly the gold paper.", "",
              "| Question | PAPER | Resolved to | ok |", "|---|---|---|---|"]
    for q in data["questions"]:
        rw = rewrites[q["query"]]
        matched = DynamoDBService.match_papers(items, rw.get("paper_name"), rw.get("paper_title")) \
            if rw.get("paper_name") else []
        ids = [m["paper_id"] for m in matched]
        ok = "-" if not rw.get("paper_name") else ("yes" if ids == [q["paper_id"]] else "no")
        lines.append(f"| {q['query']} | {rw.get('paper_name') or '-'} | {', '.join(ids) or '-'} | {ok} |")

    if gen_summary:
        lines += ["", "## Generation (LLM-as-judge, production retrieval path)", "",
                  "| Metric | Value |", "|---|---:|"]
        lines += [f"| {k} | {v} |" for k, v in gen_summary.items()]
        lines += ["", "The judge is the same local model that generated the answers, grading against a hand-written "
                  "reference at temperature 0, so absolute scores carry self-preference bias; the RAG vs closed-book "
                  "comparison uses the same judge for both and is the more reliable signal."]

    (RESULTS_DIR / "report.md").write_text("\n".join(lines) + "\n")
    (RESULTS_DIR / "retrieval.json").write_text(json.dumps(
        {"summary": rows, "chunk_stats": chunk_stats, "per_question": per_question}, indent=2))


# ── Main ───────────────────────────────────────────────────────────────────

async def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=None, help="Ollama model (default: OLLAMA_MODEL from settings)")
    parser.add_argument("--skip-generation", action="store_true")
    parser.add_argument("--gen-chunk-size", type=int, default=256,
                        help="chunk_size config (tokens) used for the generation run")
    args = parser.parse_args()

    settings = Settings(**({"ollama_model": args.model} if args.model else {}))
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    data = json.loads((EVAL_DIR / "questions.json").read_text())
    llm = LLMService(settings)
    if not await llm.check_health():
        sys.exit(f"Ollama is not reachable at {settings.ollama_base_url}")

    emb_svc = EmbeddingService(settings)

    print("1/3 query rewriting")
    rewrites = await get_rewrites(data["questions"], llm, settings.ollama_model)

    print("2/3 retrieval")
    rows, per_question, chunk_stats, indexes, rag = await run_retrieval(data, rewrites, emb_svc)

    gen_summary = None
    if not args.skip_generation:
        print("3/3 generation")
        vdb, name = indexes[args.gen_chunk_size]
        gen = await run_generation(data, rewrites, emb_svc, vdb, name, rag, llm, settings)
        gen_summary = summarize_generation(gen)
        print(json.dumps(gen_summary, indent=2))

    write_report(rows, chunk_stats, per_question, data, rewrites, gen_summary, settings.ollama_model)
    print(f"report written to {RESULTS_DIR / 'report.md'}")


if __name__ == "__main__":
    asyncio.run(main())
