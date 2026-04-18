"""
ResearchCopilot — Gradio Frontend (Hugging Face Spaces)
Connects to a remote FastAPI backend running on your GPU machine.
Set BACKEND_URL as a Space Secret pointing to your backend's public URL.
"""
import os
import re
import uuid
import hashlib

import gradio as gr
import requests

# ── Config ─────────────────────────────────────────────────────────────────
BACKEND_URL = os.getenv("BACKEND_URL", "").rstrip("/")
API_BASE = BACKEND_URL + "/api/v1" if BACKEND_URL else ""

# ── API helpers ─────────────────────────────────────────────────────────────

def api_post(endpoint: str, payload: dict, timeout: int = 60):
    if not API_BASE:
        return None, "BACKEND_URL not configured. Add it in Space Settings → Secrets."
    try:
        r = requests.post(f"{API_BASE}{endpoint}", json=payload, timeout=timeout)
        r.raise_for_status()
        return r.json(), None
    except requests.exceptions.ConnectionError:
        return None, f"Cannot reach backend at {BACKEND_URL}. Make sure it is running and publicly accessible."
    except requests.exceptions.Timeout:
        return None, f"Request to {endpoint} timed out after {timeout}s."
    except requests.exceptions.HTTPError as e:
        return None, f"API error {e.response.status_code}: {e.response.text[:300]}"
    except Exception as e:
        return None, f"Unexpected error: {e}"


def api_post_file(endpoint: str, file_bytes: bytes, filename: str, data: dict, timeout: int = 120):
    if not API_BASE:
        return None, "BACKEND_URL not configured."
    try:
        r = requests.post(
            f"{API_BASE}{endpoint}",
            files={"file": (filename, file_bytes, "application/pdf")},
            data=data,
            timeout=timeout,
        )
        r.raise_for_status()
        return r.json(), None
    except Exception as e:
        return None, f"Error: {e}"


def api_get(endpoint: str):
    if not BACKEND_URL:
        return None
    try:
        r = requests.get(f"{BACKEND_URL}{endpoint}", timeout=10)
        r.raise_for_status()
        return r.json()
    except Exception:
        return None


def check_backend_status():
    if not BACKEND_URL:
        return "⚠️ **BACKEND_URL not set** — Go to Space Settings → Secrets and add `BACKEND_URL`"
    health = api_get("/health")
    if health and health.get("status") == "ok":
        return f"✅ **Backend Online** — `{BACKEND_URL}`"
    return f"❌ **Backend Offline** — `{BACKEND_URL}` (is the server running and ngrok active?)"


def get_kb_stats() -> str:
    """Return a markdown string showing papers in the knowledge base."""
    if not BACKEND_URL:
        return "📚 Knowledge Base: _backend not connected_"
    try:
        r = requests.get(f"{BACKEND_URL}/stats", timeout=5)
        if r.status_code == 200:
            d = r.json()
            papers = d.get("papers", 0)
            chunks = d.get("paper_chunks", 0)
            return (
                f"📚 **{papers} papers** in knowledge base &nbsp;·&nbsp; "
                f"🧩 **{chunks} chunks** indexed"
            )
    except Exception:
        pass
    return "📚 Knowledge Base: _unavailable_"


def forget_topic_fn(session_id: str, topic: str, top_k: int = 20) -> str:
    if not BACKEND_URL:
        return "_Backend not connected._"
    if not session_id:
        return "⚠️ Please log in first."
    if not topic.strip():
        return "⚠️ Enter a topic to forget."
    try:
        r = requests.post(f"{BACKEND_URL}/api/v1/memory/forget",
                          json={"session_id": session_id, "topic": topic, "top_k": int(top_k)},
                          timeout=15)
        if r.status_code == 200:
            d = r.json()
            n = d.get("deleted", 0)
            if n == 0:
                return f"ℹ️ No memory entries about **{topic}** found (score threshold not met)."
            return f"✅ Forgot **{n}** memory chunks related to **{topic}**."
        return f"❌ Error {r.status_code}: {r.text}"
    except Exception as e:
        return f"❌ Request failed: {e}"


def search_author(author_name: str, max_results: int = 10):
    """Search author via backend and return (profile_md, papers_md)."""
    if not BACKEND_URL:
        return "_Backend not connected._", ""
    if not author_name.strip():
        return "_Enter an author name._", ""
    try:
        r = requests.get(f"{BACKEND_URL}/api/v1/papers/author",
                         params={"name": author_name, "max_results": int(max_results)}, timeout=15)
        if r.status_code == 404:
            return f"❌ Author **{author_name}** not found on Semantic Scholar.", ""
        if r.status_code != 200:
            return f"❌ Error: {r.status_code}", ""
        d = r.json()
        a = d["author"]
        profile = (
            f"### 👤 {a['name']}\n"
            f"📄 **{a['paper_count']}** papers &nbsp;·&nbsp; "
            f"📣 **{a['citation_count']:,}** citations &nbsp;·&nbsp; "
            f"**h-index: {a['h_index']}**"
        )
        papers = d["papers"]
        if not papers:
            return profile, "_No papers found._"
        lines = ["| # | Title | Year | Citations |",
                 "|---|-------|------|-----------|"]
        for i, p in enumerate(papers, 1):
            title = p["title"][:65] + "…" if len(p["title"]) > 65 else p["title"]
            link = f"[{title}]({p['arxiv_url']})" if p.get("arxiv_url") else title
            lines.append(f"| {i} | {link} | {p.get('year') or '—'} | {p.get('citations', 0):,} |")
        return profile, "\n".join(lines)
    except Exception as e:
        return f"❌ Request failed: {e}", ""


def get_papers_list() -> str:
    """Return markdown table of all ingested papers."""
    if not BACKEND_URL:
        return "_Backend not connected._"
    try:
        r = requests.get(f"{BACKEND_URL}/papers-list", timeout=10)
        if r.status_code == 200:
            d = r.json()
            papers = d.get("papers", [])
            if not papers:
                return "_No papers ingested yet._"
            lines = ["| # | Title | Chunks | Ingested |",
                     "|---|-------|--------|----------|"]
            for i, p in enumerate(papers, 1):
                title = p["title"][:70] + "…" if len(p["title"]) > 70 else p["title"]
                link = f"[{title}]({p['arxiv_url']})" if p.get("arxiv_url") else title
                date = p["ingested_at"][:10] if p.get("ingested_at") else "—"
                lines.append(f"| {i} | {link} | {p['chunks']} | {date} |")
            return "\n".join(lines)
    except Exception:
        pass
    return "_Failed to load paper list._"


# ── Formatters ──────────────────────────────────────────────────────────────

def _fmt_papers(papers: list) -> str:
    if not papers:
        return ""
    lines = [f"\n\n---\n### 📄 Papers Found ({len(papers)})"]
    for p in papers:
        authors = ", ".join(p.get("authors", [])[:3])
        date = p.get("published_date", "")
        links = []
        if p.get("url"):
            links.append(f"[View]({p['url']})")
        if p.get("pdf_url"):
            links.append(f"[PDF]({p['pdf_url']})")
        lines.append(
            f"\n**{p['title']}**  \n"
            f"_{authors}_ · {date}  \n"
            + (" · ".join(links) if links else "")
        )
    return "\n".join(lines)


def _fmt_sources(sources: list) -> str:
    if not sources:
        return ""
    lines = [f"\n\n---\n### 📚 Sources Retrieved ({len(sources)})"]
    for i, s in enumerate(sources[:6]):
        title = s.get("paper_title") or "Untitled"
        score = s.get("score", 0)
        text = s.get("text", "")[:250]
        lines.append(f"\n**[{i+1}] {title}** · Score: `{score:.3f}`  \n{text}…")
    return "\n".join(lines)


# ── Paper Discovery ─────────────────────────────────────────────────────────

# ── Session management helpers ───────────────────────────────────────────────

def _fetch_sessions(user_id: str):
    if not user_id or not API_BASE:
        return []
    try:
        r = requests.get(f"{API_BASE}/memory/sessions/{user_id}", timeout=10)
        r.raise_for_status()
        return r.json().get("sessions", [])
    except Exception:
        return []


def _sessions_to_choices(sessions: list):
    return [(s.get("session_name", "New Chat")[:40], s["session_id"]) for s in sessions]


def login_fn(username: str):
    if not username.strip():
        return "", gr.update(choices=[], value=None), "⚠️ Enter a username."
    uid = username.strip().lower().replace(" ", "_")
    sessions = _fetch_sessions(uid)
    choices = _sessions_to_choices(sessions)
    msg = f"✅ Logged in as **{uid}** — {len(sessions)} saved chat(s)"
    return uid, gr.update(choices=choices, value=choices[0][1] if choices else None), msg


def new_chat_fn(user_id: str):
    if not user_id:
        return "", gr.update(), [], []
    sid = str(uuid.uuid4())
    api_post("/memory/sessions", {"user_id": user_id, "session_id": sid, "session_name": "New Chat"})
    sessions = _fetch_sessions(user_id)
    choices = _sessions_to_choices(sessions)
    return sid, gr.update(choices=choices, value=sid), [], []


def switch_session_fn(session_id: str):
    if not session_id:
        return [], []
    result, _ = api_post("/memory/retrieve", {"session_id": session_id, "limit": 50})
    if not result:
        return [], []
    history = []
    for e in result.get("entries", []):
        role = e.get("role", "assistant")
        content = e.get("content", "")
        if role == "assistant":
            content = _clean_report(content)
        history.append({"role": role, "content": content})
    return history, history


def rename_session_fn(user_id: str, session_id: str, new_name: str):
    if not user_id or not session_id or not new_name.strip():
        return gr.update()
    try:
        requests.put(f"{API_BASE}/memory/sessions/{user_id}/{session_id}/rename",
                     json={"session_name": new_name.strip()}, timeout=10)
    except Exception:
        pass
    sessions = _fetch_sessions(user_id)
    return gr.update(choices=_sessions_to_choices(sessions), value=session_id)


def delete_session_fn(user_id: str, session_id: str):
    if not user_id or not session_id:
        return "", gr.update(), [], []
    try:
        requests.delete(f"{API_BASE}/memory/sessions/{user_id}/{session_id}", timeout=10)
    except Exception:
        pass
    sessions = _fetch_sessions(user_id)
    choices = _sessions_to_choices(sessions)
    new_sid = choices[0][1] if choices else ""
    return new_sid, gr.update(choices=choices, value=new_sid if new_sid else None), [], []


# ── Paper Discovery ─────────────────────────────────────────────────────────

def paper_discovery_fn(message: str, history: list, session_id: str, user_id: str, max_results: int = 10):
    if not message.strip():
        yield history, session_id
        return

    # Auto-name session from first message
    if not history and user_id and session_id:
        try:
            requests.put(f"{API_BASE}/memory/sessions/{user_id}/{session_id}/rename",
                         json={"session_name": message[:40]}, timeout=5)
        except Exception:
            pass

    is_followup = len(history) > 0

    history = history + [
        {"role": "user", "content": message},
        {"role": "assistant", "content": "🔍 Searching arXiv for relevant papers…" if not is_followup else "🤖 Generating answer from knowledge base…"},
    ]
    yield history, session_id

    papers = []
    if not is_followup:
        # Step 1 — search (first message only)
        result, err = api_post(
            "/papers/search",
            {"query": message, "source": "arxiv", "max_results": int(max_results)},
            timeout=60,
        )
        papers = result.get("papers", []) if result else []

        if err or not papers:
            notice = err or "⚠️ No papers found — answering from existing knowledge base."
            history[-1]["content"] = notice + "\n\n🤖 Generating answer…"
            yield history, session_id
            papers = []
        else:
            history[-1]["content"] = f"✅ Found **{len(papers)}** papers. Ingesting…"
            yield history, session_id

            # Step 2 — ingest
            ingested = 0
            skipped = 0
            for i, paper in enumerate(papers):
                authors = ", ".join(paper.get("authors", [])[:5])
                content = (
                    f"Title: {paper['title']}\n\n"
                    f"Authors: {authors}\n\n"
                    f"Published: {paper.get('published_date', 'N/A')}\n\n"
                    f"Abstract:\n{paper.get('abstract', '')}"
                )
                res, _ = api_post(
                    "/papers/ingest",
                    {"paper_id": paper["paper_id"], "title": paper["title"], "content": content},
                    timeout=60,
                )
                if res and res.get("status") == "success":
                    ingested += 1
                elif res and res.get("status") == "already_exists":
                    skipped += 1
                status_msg = f"📥 Ingested **{ingested}** new"
                if skipped:
                    status_msg += f", **{skipped}** already in KB"
                history[-1]["content"] = status_msg + f" / {i+1} papers…\n\n🤖 Generating answer…"
                yield history, session_id

    # Step 3 — RAG
    history[-1]["content"] = "🤖 Generating answer from knowledge base…"
    yield history, session_id

    rag, rag_err = api_post(
        "/rag/query",
        {"query": message, "session_id": session_id, "user_id": user_id, "top_k": 5, "use_memory": True},
        timeout=300,
    )

    if rag_err or not rag:
        history[-1]["content"] = rag_err or "❌ Failed to get a response from the backend."
        yield history, session_id
        return

    answer = rag.get("answer", "").strip() or "The model is warming up — please try again in a moment."
    sources = rag.get("sources", [])
    model = "qwen3.5:0.8b"

    footer = f"\n\n---\n_Model: `{model}` · Sources: {len(sources)}_"
    if papers:
        footer = f"\n\n---\n_Model: `{model}` · Papers: {len(papers)} · Sources: {len(sources)}_"
    full = (
        _clean_report(answer)
        + (_fmt_papers(papers) if papers else "")
        + _fmt_sources(sources)
        + footer
    )
    history[-1]["content"] = full
    yield history, session_id


# ── RAG Q&A ─────────────────────────────────────────────────────────────────

def rag_fn(message: str, history: list, session_id: str, user_id: str, top_k: int, use_memory: bool, paper_filter: str):
    if not message.strip():
        yield history
        return

    paper_ids = [p.strip() for p in paper_filter.split(",") if p.strip()] or None

    # Auto-name session from first message
    if not history and user_id and session_id:
        try:
            requests.put(f"{API_BASE}/memory/sessions/{user_id}/{session_id}/rename",
                         json={"session_name": message[:40]}, timeout=5)
        except Exception:
            pass

    history = history + [
        {"role": "user", "content": message},
        {"role": "assistant", "content": "⏳ Retrieving context and generating answer…"},
    ]
    yield history

    result, err = api_post(
        "/rag/query",
        {"query": message, "session_id": session_id, "user_id": user_id,
         "top_k": top_k, "use_memory": use_memory, "paper_ids": paper_ids},
        timeout=300,
    )

    if err or not result:
        history[-1]["content"] = err or "❌ Failed to get a response from the backend."
        yield history
        return

    answer = result.get("answer", "").strip() or "⚠️ Model did not return an answer — try again."
    sources = result.get("sources", [])

    history[-1]["content"] = (
        _clean_report(answer)
        + _fmt_sources(sources)
        + f"\n\n---\n_Model: `qwen3.5:0.8b` · Sources: {len(sources)}_"
    )
    yield history


# ── PDF Ingestion ────────────────────────────────────────────────────────────

def ingest_pdf_fn(file, title: str, paper_id_input: str) -> str:
    if not title.strip():
        return "❌ Paper title is required."
    if file is None:
        return "❌ Please upload a PDF file."
    pid = paper_id_input.strip() or "paper_" + hashlib.md5(title.lower().strip().encode()).hexdigest()[:8]
    with open(file, "rb") as f:
        file_bytes = f.read()
    filename = os.path.basename(file)
    result, err = api_post_file("/papers/ingest/pdf", file_bytes, filename, {"paper_id": pid, "title": title})
    if err:
        return f"❌ {err}"
    if result and result.get("status") == "success":
        return (
            f"✅ Ingested successfully!\n\n"
            f"- **Paper ID:** `{pid}`\n"
            f"- **Chunks created:** {result.get('chunks_created', 0)}\n"
            f"- **Embeddings stored:** {result.get('embeddings_stored', 0)}"
        )
    return f"❌ Ingestion failed: {result.get('message', 'Unknown error') if result else 'No response'}"


def ingest_text_fn(content: str, title: str, paper_id_input: str) -> str:
    if not title.strip():
        return "❌ Paper title is required."
    if not content.strip():
        return "❌ Content cannot be empty."
    pid = paper_id_input.strip() or "paper_" + hashlib.md5(title.lower().strip().encode()).hexdigest()[:8]
    result, err = api_post("/papers/ingest", {"paper_id": pid, "title": title, "content": content}, timeout=120)
    if err:
        return f"❌ {err}"
    if result and result.get("status") == "success":
        return (
            f"✅ Ingested successfully!\n\n"
            f"- **Paper ID:** `{pid}`\n"
            f"- **Chunks created:** {result.get('chunks_created', 0)}\n"
            f"- **Embeddings stored:** {result.get('embeddings_stored', 0)}"
        )
    return f"❌ Ingestion failed: {result.get('message', 'Unknown') if result else 'No response'}"


# ── Memory ──────────────────────────────────────────────────────────────────

def memory_chronological_fn(session_id: str, limit: int) -> str:
    result, err = api_post("/memory/retrieve", {"session_id": session_id, "limit": int(limit)}, timeout=20)
    if err:
        return f"❌ {err}"
    entries = result.get("entries", []) if result else []
    if not entries:
        return "No memory entries found for this session. Start chatting in Paper Discovery or RAG Q&A!"
    lines = [f"**Retrieved {len(entries)} entries**\n"]
    for e in entries:
        role = e.get("role", "unknown").capitalize()
        ts = e.get("timestamp", "")[:19].replace("T", " ")
        content = e.get("content", "")[:400]
        lines.append(f"**{role}** · `{ts}`\n{content}\n\n---")
    return "\n".join(lines)


def memory_semantic_fn(session_id: str, query: str, limit: int) -> str:
    if not query.strip():
        return "Enter a search query."
    result, err = api_post("/memory/retrieve", {"session_id": session_id, "limit": int(limit), "query": query}, timeout=30)
    if err:
        return f"❌ {err}"
    entries = result.get("entries", []) if result else []
    if not entries:
        return "No relevant memory found for that query."
    lines = [f"**Found {len(entries)} relevant entries**\n"]
    for e in entries:
        role = e.get("role", "unknown").capitalize()
        content = e.get("content", "")[:400]
        score = e.get("metadata", {}).get("score")
        score_str = f" · Similarity: `{score:.3f}`" if score is not None else ""
        lines.append(f"**{role}**{score_str}\n{content}\n\n---")
    return "\n".join(lines)


# ── Report ──────────────────────────────────────────────────────────────────

_PLACEHOLDER_RE = re.compile(r'%%%[A-Z_0-9]+%%%')
_INLINE_NOSPACE_RE = re.compile(r'(\S)(\$[^$\n]+?\$)(\S)')
_EXISTING_DISPLAY = re.compile(r'(\$\$.*?\$\$)', re.DOTALL)
_LATEX_CMD_RE = re.compile(r'\\[a-zA-Z]+')
_LATEX_KEYWORDS_RE = re.compile(
    r'\\(frac|left|right|prod|sum|int|partial|nabla|sqrt|leq|geq|approx|propto|cdot|quad|qquad|'
    r'text|mathcal|mathbb|begin|end|times|infty|'
    r'alpha|beta|gamma|delta|epsilon|zeta|eta|theta|iota|kappa|lambda|mu|nu|xi|pi|rho|sigma|tau|upsilon|phi|chi|psi|omega|'
    r'Gamma|Delta|Theta|Lambda|Xi|Pi|Sigma|Phi|Psi|Omega|'
    r'exp|log|ln|sin|cos|tan|lim|max|min|argmax|argmin|'
    r'mathbf|mathrm|boldsymbol|hat|bar|tilde|vec|diag|softmax|relu|'
    r'equiv|neq|sim|cong|ll|gg|in|notin|subset|cup|cap|forall|exists)\b'
)
_MATH_RELATION_RE = re.compile(r'(=|\\approx|\\equiv|\\propto|\\leq|\\geq|\\neq|\\sim|\\rightarrow|\\Rightarrow|\\ll|\\gg)')
_SUBSCRIPT_RE = re.compile(r'[a-zA-Z0-9][_\^]\{')
_SUBSCRIPT_SIMPLE_RE = re.compile(r'[a-zA-Z0-9](?:_[a-zA-Z0-9]+|\^[a-zA-Z0-9]+)')
_BRACKET_DISPLAY_RE = re.compile(r'\\\[(.*?)\\\]', re.DOTALL)


def _is_raw_latex(line: str) -> bool:
    s = line.strip()
    if not s or '$' in s:
        return False
    if s.startswith(('#', '>', '|')):
        return False
    # Skip bullet/numbered list items that are plain text
    if s.startswith(('-', '*')) and not _LATEX_CMD_RE.search(s):
        return False
    kw = len(_LATEX_KEYWORDS_RE.findall(s))
    has_cmd = bool(_LATEX_CMD_RE.search(s))
    has_sub = bool(_SUBSCRIPT_RE.search(s) or _SUBSCRIPT_SIMPLE_RE.search(s))
    has_relation = bool(_MATH_RELATION_RE.search(s))
    n_words = len(s.split())
    # Line starts with a LaTeX command → almost certainly an equation
    if s.startswith('\\') and has_cmd:
        return True
    # Plain English paragraphs: many words, no LaTeX commands
    if n_words > 15 and not has_cmd:
        return False
    # 2+ named keywords → definitely LaTeX
    if kw >= 2:
        return True
    # 1 keyword + a relation or subscript → LaTeX
    if kw >= 1 and (has_relation or has_sub):
        return True
    # LaTeX command + math relation → LaTeX
    if has_cmd and has_relation:
        return True
    # Subscripts + relation + short line → likely LaTeX
    if has_sub and has_relation and n_words < 12:
        return True
    return False


def _is_latex_continuation(line: str) -> bool:
    s = line.strip()
    if not s or '$' in s:
        return False
    if s.startswith('&'):
        return True
    if s.endswith(r'\\'):
        return True
    if s.startswith((r'\\begin', r'\\end')):
        return True
    return False


def _normalize_display_expr(expr: str) -> str:
    # Convert commands like \\text and \\frac to \text and \frac.
    return re.sub(r'\\\\([A-Za-z])', r'\\\1', expr)


def _process_section(text: str) -> str:
    """Wrap raw LaTeX lines in $$...$$ within a non-display-math text section."""
    lines = text.split('\n')
    result = []
    i = 0
    while i < len(lines):
        if _is_raw_latex(lines[i]):
            block = [lines[i].strip()]
            i += 1
            # Keep only true continuation lines in the same display block.
            while i < len(lines) and _is_latex_continuation(lines[i]):
                block.append(lines[i].strip())
                i += 1
            expr = _normalize_display_expr('\n'.join(block))
            result.append('\n$$\n' + expr + '\n$$\n')
        else:
            result.append(lines[i])
            i += 1
    return '\n'.join(result)


def _wrap_raw_latex_lines(text: str) -> str:
    """Split on existing $$...$$ blocks, then fix raw LaTeX in remaining text."""
    text = _BRACKET_DISPLAY_RE.sub(lambda m: f"$$\n{m.group(1).strip()}\n$$", text)
    parts = _EXISTING_DISPLAY.split(text)
    out = []
    for part in parts:
        if part.startswith('$$') and part.endswith('$$'):
            out.append(part)  # already wrapped, keep as-is
        else:
            out.append(_process_section(part))
    return ''.join(out)


def _clean_report(text: str) -> str:
    text = _PLACEHOLDER_RE.sub('', text)
    text = _wrap_raw_latex_lines(text)
    for _ in range(3):
        text = _INLINE_NOSPACE_RE.sub(r'\1 \2 \3', text)
    return text


def generate_report_fn(session_id: str, topic: str, max_length: int, fmt: str):
    def _msg(text):
        return [{"role": "assistant", "content": text}]

    if not topic.strip():
        return _msg("❌ Please enter a research topic.")
    result, err = api_post(
        "/report/generate",
        {"topic": topic, "session_id": session_id, "max_length": int(max_length), "format": fmt},
        timeout=3600,
    )
    if err:
        return _msg(f"❌ {err}")
    if result and result.get("report"):
        sources = result.get("sources_used", [])
        report = _clean_report(result["report"])
        if sources:
            report += f"\n\n---\n_Sources used: {len(sources)}_"
        return _msg(report)
    return _msg("❌ Failed to generate report.")


# ── Gradio UI ────────────────────────────────────────────────────────────────

CSS = """
footer { display: none !important; }
.session-panel { background: #1e1e2e; border-radius: 8px; padding: 8px; }
"""

with gr.Blocks(title="ResearchCopilot") as demo:

    user_id_state = gr.State("")
    session_id = gr.State("")

    # ── Header ───────────────────────────────────────────────────────────────
    with gr.Row():
        with gr.Column():
            gr.Markdown("# 🔬 ResearchCopilot\n*Production-Grade AI Research Assistant*")

    # ── Login row ─────────────────────────────────────────────────────────────
    with gr.Row():
        username_input = gr.Textbox(
            placeholder="Enter username to save & restore chats…",
            label="Username",
            scale=4,
        )
        login_btn = gr.Button("🔑 Login", variant="primary", scale=1)
        login_status = gr.Markdown("_Enter a username to enable persistent chat history._")

    # ── Session management row ────────────────────────────────────────────────
    with gr.Row():
        sessions_dropdown = gr.Dropdown(
            choices=[], label="💬 Your Chats", scale=4, interactive=True,
            info="Select a past chat to reload it"
        )
        new_chat_btn = gr.Button("✏️ New Chat", variant="primary", scale=1)

    with gr.Row():
        rename_input = gr.Textbox(placeholder="New chat name…", label="Rename current chat", scale=4)
        rename_btn = gr.Button("✏️ Rename", scale=1)
        delete_btn = gr.Button("🗑️ Delete Chat", variant="stop", scale=1)

    gr.Markdown("---")

    # ── Knowledge Base Stats ──────────────────────────────────────────────────
    with gr.Row():
        kb_stats_md = gr.Markdown(get_kb_stats())
        show_papers_btn = gr.Button("📋 Show Papers", size="sm", variant="secondary", scale=0)
        refresh_kb_btn = gr.Button("🔄 Refresh", size="sm", variant="secondary", scale=0)

    papers_list_md = gr.Markdown(visible=False)

    gr.Markdown("---")

    # ── Tabs ─────────────────────────────────────────────────────────────────
    with gr.Tabs():

        # ── Tab 1: Paper Discovery ────────────────────────────────────────────
        with gr.Tab("🔎 Paper Discovery"):
            gr.Markdown(
                "Ask any research question. Papers are found on arXiv automatically, "
                "ingested, and used to answer your question."
            )
            discovery_bot = gr.Chatbot(
                height=520,
                label="Research Chat",
                render_markdown=True,
                latex_delimiters=[
                    {"left": "$$", "right": "$$", "display": True},
                    {"left": "$", "right": "$", "display": False},
                ],
                buttons=["copy", "copy_all"],
                layout="bubble",
            )
            with gr.Row():
                discovery_input = gr.Textbox(
                    placeholder="Ask a research question…",
                    label="",
                    scale=5,
                    submit_btn="Send",
                )
                discovery_clear = gr.Button("🗑️ New Chat", variant="secondary", scale=1)
            discovery_max_papers = gr.Slider(1, 30, value=10, step=1, label="Max papers to fetch")

        # ── Tab 2: PDF Ingestion ──────────────────────────────────────────────
        with gr.Tab("📄 PDF Ingestion"):
            gr.Markdown("Upload a PDF or paste text to embed and store in the ResearchCopilot knowledge base.")
            with gr.Tabs():
                with gr.Tab("📎 Upload PDF"):
                    with gr.Row():
                        pdf_title = gr.Textbox(label="Paper Title *", placeholder="Attention Is All You Need", scale=3)
                        pdf_paper_id = gr.Textbox(label="Paper ID (optional)", placeholder="arxiv_1706.03762", scale=2)
                    pdf_file = gr.File(label="Choose PDF file", file_types=[".pdf"])
                    pdf_btn = gr.Button("📥 Ingest PDF", variant="primary")
                    pdf_result = gr.Markdown()

                with gr.Tab("📝 Paste Text"):
                    with gr.Row():
                        text_title = gr.Textbox(label="Paper Title *", placeholder="My Research Paper", scale=3)
                        text_paper_id = gr.Textbox(label="Paper ID (optional)", placeholder="my_paper_001", scale=2)
                    text_content = gr.Textbox(
                        label="Paper text / abstract *",
                        placeholder="Paste the full text or abstract here…",
                        lines=10,
                    )
                    text_btn = gr.Button("📥 Ingest Text", variant="primary")
                    text_result = gr.Markdown()

        # ── Tab 3: RAG Q&A ────────────────────────────────────────────────────
        with gr.Tab("💬 RAG Q&A"):
            gr.Markdown(
                "Ask questions about your ingested papers. "
                "The assistant retrieves relevant chunks and grounds its answer in the literature."
            )
            with gr.Accordion("⚙️ Query Settings", open=False):
                with gr.Row():
                    rag_top_k = gr.Slider(1, 15, value=5, step=1, label="Retrieved chunks (top-k)")
                    rag_use_memory = gr.Checkbox(value=True, label="Use conversation memory")
                rag_filter = gr.Textbox(
                    label="Filter to paper IDs (comma-separated, leave blank for all)",
                    placeholder="arxiv_2310.06825, ss_abc123",
                )

            rag_bot = gr.Chatbot(
                height=450,
                label="Q&A Chat",
                render_markdown=True,
                latex_delimiters=[
                    {"left": "$$", "right": "$$", "display": True},
                    {"left": "$", "right": "$", "display": False},
                ],
                buttons=["copy", "copy_all"],
                layout="bubble",
            )
            with gr.Row():
                rag_input = gr.Textbox(
                    placeholder="Ask a research question…",
                    label="",
                    scale=5,
                    submit_btn="Send",
                )
                rag_clear = gr.Button("🗑️ Clear", variant="secondary", scale=1)

            gr.Markdown("---")
            gr.Markdown("### 📋 Generate Research Report")
            with gr.Row():
                report_topic = gr.Textbox(
                    label="Research topic",
                    placeholder="e.g. Transformer attention mechanisms in NLP",
                    scale=3,
                )
                report_fmt = gr.Dropdown(["markdown", "plain"], value="markdown", label="Format", scale=1)
                report_length = gr.Slider(200, 50000, value=2000, step=500, label="Max words", scale=2)
            report_btn = gr.Button("📝 Generate Report", variant="primary")
            report_out = gr.Chatbot(
                label="Report",
                height=700,
                latex_delimiters=[
                    {"left": "$$", "right": "$$", "display": True},
                    {"left": "$", "right": "$", "display": False},
                ],
            )

        # ── Tab 4: Author Discovery ───────────────────────────────────────────
        with gr.Tab("👤 Author Discovery"):
            gr.Markdown("Search any researcher by name — see their profile and top papers via Semantic Scholar.")
            with gr.Row():
                author_input = gr.Textbox(label="Author Name", placeholder="e.g. Yann LeCun", scale=4)
                author_max = gr.Slider(5, 20, value=10, step=1, label="Max papers", scale=1)
                author_search_btn = gr.Button("🔍 Search", variant="primary", scale=1)
            author_profile_md = gr.Markdown()
            author_papers_md = gr.Markdown()

        # ── Tab 5: Memory History ─────────────────────────────────────────────
        with gr.Tab("🧠 Memory History"):
            gr.Markdown(
                "Browse or semantically search your session's conversation history "
                "stored in ChromaDB and DynamoDB."
            )
            with gr.Tabs():
                with gr.Tab("📜 Chronological"):
                    mem_limit = gr.Slider(3, 50, value=10, step=1, label="Number of turns")
                    mem_refresh_btn = gr.Button("🔄 Load Memory", variant="primary")
                    mem_chron_out = gr.Markdown()

                with gr.Tab("🔍 Semantic Search"):
                    mem_query = gr.Textbox(label="Search your conversation", placeholder="What did we discuss about BERT?")
                    mem_sem_limit = gr.Slider(1, 20, value=5, step=1, label="Max results")
                    mem_search_btn = gr.Button("🔍 Search", variant="primary")
                    mem_sem_out = gr.Markdown()

                with gr.Tab("🗑️ Forget Topic"):
                    gr.Markdown(
                        "**Explicit forgetting** — remove memories about a specific topic from ChromaDB.\n\n"
                        "e.g. type `GANs` to delete all conversation chunks related to GANs."
                    )
                    forget_input = gr.Textbox(label="Topic to forget", placeholder="e.g. GANs, BERT, transformers")
                    forget_k = gr.Slider(5, 50, value=20, step=5, label="Max entries to scan")
                    forget_btn = gr.Button("🗑️ Forget", variant="stop")
                    forget_out = gr.Markdown()

    # ── Event wiring ──────────────────────────────────────────────────────────

    # Author Discovery
    author_search_btn.click(
        fn=search_author,
        inputs=[author_input, author_max],
        outputs=[author_profile_md, author_papers_md],
    )
    author_input.submit(
        fn=search_author,
        inputs=[author_input, author_max],
        outputs=[author_profile_md, author_papers_md],
    )

    # KB stats refresh
    refresh_kb_btn.click(fn=get_kb_stats, outputs=kb_stats_md)

    # Show/hide paper list (state tracks visibility)
    papers_visible_state = gr.State(False)

    def toggle_papers(is_visible):
        if is_visible:
            return False, gr.update(visible=False), gr.update(value="📋 Show Papers")
        return True, gr.update(visible=True, value=get_papers_list()), gr.update(value="🔼 Hide Papers")

    show_papers_btn.click(
        fn=toggle_papers,
        inputs=[papers_visible_state],
        outputs=[papers_visible_state, papers_list_md, show_papers_btn],
    )

    # Login
    login_btn.click(
        fn=login_fn,
        inputs=[username_input],
        outputs=[user_id_state, sessions_dropdown, login_status],
    )
    username_input.submit(
        fn=login_fn,
        inputs=[username_input],
        outputs=[user_id_state, sessions_dropdown, login_status],
    )

    # New chat
    new_chat_btn.click(
        fn=new_chat_fn,
        inputs=[user_id_state],
        outputs=[session_id, sessions_dropdown, discovery_bot, rag_bot],
    )

    # Switch session (load history)
    sessions_dropdown.change(
        fn=lambda sid: (sid, *switch_session_fn(sid)),
        inputs=[sessions_dropdown],
        outputs=[session_id, discovery_bot, rag_bot],
    )

    # Rename session
    rename_btn.click(
        fn=rename_session_fn,
        inputs=[user_id_state, session_id, rename_input],
        outputs=[sessions_dropdown],
    ).then(fn=lambda: "", outputs=rename_input)

    # Delete session
    delete_btn.click(
        fn=delete_session_fn,
        inputs=[user_id_state, session_id],
        outputs=[session_id, sessions_dropdown, discovery_bot, rag_bot],
    )

    # Paper Discovery — submit on Enter or Send
    discovery_input.submit(
        fn=paper_discovery_fn,
        inputs=[discovery_input, discovery_bot, session_id, user_id_state, discovery_max_papers],
        outputs=[discovery_bot, session_id],
    ).then(fn=lambda: "", outputs=discovery_input)

    discovery_clear.click(fn=lambda: [], outputs=discovery_bot)

    # RAG Q&A
    rag_input.submit(
        fn=rag_fn,
        inputs=[rag_input, rag_bot, session_id, user_id_state, rag_top_k, rag_use_memory, rag_filter],
        outputs=rag_bot,
    ).then(fn=lambda: "", outputs=rag_input)

    rag_clear.click(fn=lambda: [], outputs=rag_bot)

    # PDF Ingestion
    pdf_btn.click(
        fn=ingest_pdf_fn,
        inputs=[pdf_file, pdf_title, pdf_paper_id],
        outputs=pdf_result,
    )
    text_btn.click(
        fn=ingest_text_fn,
        inputs=[text_content, text_title, text_paper_id],
        outputs=text_result,
    )

    # Report
    report_btn.click(
        fn=generate_report_fn,
        inputs=[session_id, report_topic, report_length, report_fmt],
        outputs=report_out,
    )

    # Memory
    mem_refresh_btn.click(
        fn=memory_chronological_fn,
        inputs=[session_id, mem_limit],
        outputs=mem_chron_out,
    )
    mem_search_btn.click(
        fn=memory_semantic_fn,
        inputs=[session_id, mem_query, mem_sem_limit],
        outputs=mem_sem_out,
    )

    # Forget topic
    forget_btn.click(
        fn=forget_topic_fn,
        inputs=[session_id, forget_input, forget_k],
        outputs=forget_out,
    )

if __name__ == "__main__":
    demo.launch(theme=gr.themes.Soft(), css=CSS)
