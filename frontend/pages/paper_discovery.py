"""
Paper Discovery — Claude-style research chat.
User asks a question → papers auto-searched → auto-ingested → RAG answer.
"""
import streamlit as st
from frontend.render_utils import render_answer


def render(api_post, api_base: str):
    # ── Init discovery-specific state ─────────────────────────────────────
    if "discovery_messages" not in st.session_state:
        st.session_state.discovery_messages = []
    if "discovery_paper_ids" not in st.session_state:
        st.session_state.discovery_paper_ids = []
    # ── Header row ────────────────────────────────────────────────────────
    col_title, col_btn = st.columns([5, 1])
    with col_title:
        st.header("Paper Discovery")
        st.caption(
            "Ask any research question. Relevant papers are found automatically, "
            "ingested, and used to answer your question."
        )
    with col_btn:
        st.write("")
        if st.session_state.discovery_messages:
            if st.button("New Chat", use_container_width=True, type="primary"):
                st.session_state.discovery_messages = []
                st.session_state.discovery_paper_ids = []
                st.rerun()

    st.divider()

    # ── Chat history ──────────────────────────────────────────────────────
    for msg in st.session_state.discovery_messages:
        with st.chat_message(msg["role"]):
            if msg["role"] == "assistant":
                render_answer(msg["content"])
            else:
                st.markdown(msg["content"])
            if msg.get("papers_found"):
                _render_paper_cards(msg["papers_found"])
            if msg.get("sources"):
                _render_sources(msg["sources"])

    # ── Chat input (Enter or Send button both submit) ─────────────────────
    with st.form(key="discovery_form", clear_on_submit=True):
        col_input, col_send = st.columns([6, 1])
        with col_input:
            user_input_text = st.text_input(
                "question",
                placeholder="Ask a research question...",
                label_visibility="collapsed",
            )
        with col_send:
            send_clicked = st.form_submit_button("Send", use_container_width=True, type="primary")
        max_papers = st.slider("Max papers to fetch", min_value=1, max_value=30, value=10, step=1)

    user_input = user_input_text.strip() if send_clicked and user_input_text.strip() else None

    if user_input:

        # Show user message
        st.session_state.discovery_messages.append({
            "role": "user",
            "content": user_input,
        })
        with st.chat_message("user"):
            st.markdown(user_input)

        # Assistant pipeline
        with st.chat_message("assistant"):
            papers_found = []
            ingested_ids = []

            # Step 1: Search
            with st.status("Searching for relevant papers...", expanded=True) as status:
                st.write("Querying arXiv and Semantic Scholar...")
                search_result = api_post(
                    "/papers/search",
                    {"query": user_input, "source": "arxiv", "max_results": max_papers},
                    timeout=60,
                )
                papers_found = search_result.get("papers", []) if search_result else []

                if not papers_found:
                    status.update(label="No papers found — answering from existing knowledge.", state="error")
                else:
                    st.write(f"Found **{len(papers_found)}** papers. Ingesting...")

                    # Step 2: Ingest each paper
                    for paper in papers_found:
                        short_title = paper["title"][:55] + ("..." if len(paper["title"]) > 55 else "")
                        st.write(f"Ingesting: *{short_title}*")
                        authors = ", ".join(paper.get("authors", [])[:5])
                        content = (
                            f"Title: {paper['title']}\n\n"
                            f"Authors: {authors}\n\n"
                            f"Published: {paper.get('published_date', 'N/A')}\n\n"
                            f"Abstract:\n{paper.get('abstract', '')}"
                        )
                        ingest_result = api_post(
                            "/papers/ingest",
                            {
                                "paper_id": paper["paper_id"],
                                "title": paper["title"],
                                "content": content,
                            },
                            timeout=60,
                        )
                        if ingest_result and ingest_result.get("status") == "success":
                            ingested_ids.append(paper["paper_id"])
                            # Track globally for summary
                            if paper["paper_id"] not in st.session_state.discovery_paper_ids:
                                st.session_state.discovery_paper_ids.append(paper["paper_id"])

                    st.write(f"Generating answer from **{len(ingested_ids)}** paper(s)...")
                    status.update(
                        label=f"Loaded {len(ingested_ids)} paper(s). Generating answer...",
                        state="running",
                    )

            # Step 3: RAG query — search ALL papers in ChromaDB (no filter)
            rag_result = api_post(
                "/rag/query",
                {
                    "query": user_input,
                    "session_id": st.session_state.session_id,
                    "top_k": 5,
                    "use_memory": True,
                },
                timeout=300,
            )

            if rag_result:
                answer = rag_result.get("answer", "").strip()
                if not answer:
                    answer = (
                        "The model is warming up — please ask again in a few seconds."
                    )
                sources = rag_result.get("sources", [])
                model = rag_result.get("model_used", "unknown")

                render_answer(answer)
                st.caption(
                    f"Model: `{model}` · "
                    f"Papers ingested: {len(ingested_ids)} · "
                    f"Sources retrieved: {len(sources)}"
                )

                if papers_found:
                    _render_paper_cards(papers_found)
                if sources:
                    _render_sources(sources)

                st.session_state.discovery_messages.append({
                    "role": "assistant",
                    "content": answer,
                    "papers_found": papers_found,
                    "sources": sources,
                })
            else:
                error = "Failed to get a response from the backend."
                st.error(error)
                st.session_state.discovery_messages.append({
                    "role": "assistant",
                    "content": error,
                })


# ── Summary ────────────────────────────────────────────────────────────────

def _render_summary(api_post):
    st.subheader("Chat Summary")

    # Build a text block from all turns
    conversation_text = ""
    for msg in st.session_state.discovery_messages:
        role = "User" if msg["role"] == "user" else "Assistant"
        conversation_text += f"{role}: {msg['content']}\n\n"

    # Use report endpoint with the conversation as context, or generate locally
    with st.spinner("Generating summary..."):
        result = api_post(
            "/report/generate",
            {
                "topic": "Summary of research conversation",
                "session_id": st.session_state.session_id,
                "paper_ids": st.session_state.discovery_paper_ids or None,
                "max_length": 400,
                "format": "markdown",
            },
            timeout=300,
        )

    if result and result.get("report"):
        st.markdown(result["report"])
        st.caption(f"Sources used: {len(result.get('sources_used', []))}")
        st.download_button(
            "Download Summary",
            data=result["report"],
            file_name="research_summary.md",
            mime="text/markdown",
            use_container_width=True,
        )
    else:
        # Fallback: show conversation digest
        st.markdown("**Questions asked in this session:**")
        for msg in st.session_state.discovery_messages:
            if msg["role"] == "user":
                st.markdown(f"- {msg['content']}")
        st.markdown("**Papers referenced:**")
        for msg in st.session_state.discovery_messages:
            for paper in msg.get("papers_found", []):
                st.markdown(f"- {paper['title']} ({paper.get('published_date', '')})")


# ── Helper renderers ───────────────────────────────────────────────────────

def _render_paper_cards(papers: list):
    if not papers:
        return
    with st.expander(f"Papers found ({len(papers)})", expanded=False):
        for paper in papers:
            st.markdown(f"**{paper['title']}**")
            authors = ", ".join(paper.get("authors", [])[:3])
            st.caption(
                f"{authors} · {paper.get('published_date', '')} · {paper['source'].upper()}"
            )
            if paper.get("abstract"):
                st.text(paper["abstract"][:300] + "...")
            links = []
            if paper.get("url"):
                links.append(f"[View]({paper['url']})")
            if paper.get("pdf_url"):
                links.append(f"[PDF]({paper['pdf_url']})")
            if links:
                st.markdown(" · ".join(links))
            st.markdown("---")


def _render_sources(sources: list):
    if not sources:
        return
    with st.expander(f"{len(sources)} source chunk(s) retrieved", expanded=False):
        for i, src in enumerate(sources):
            st.markdown(
                f"**[{i+1}]** `{src.get('paper_id', '')}` — "
                f"*{src.get('paper_title', '') or 'Untitled'}* · "
                f"Score: `{src.get('score', 0):.3f}`"
            )
            st.text(src.get("text", "")[:300] + ("..." if len(src.get("text", "")) > 300 else ""))
            if i < len(sources) - 1:
                st.markdown("---")
