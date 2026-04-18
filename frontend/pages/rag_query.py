"""
RAG Q&A Page
Full chat interface: user asks a question → backend runs the RAG pipeline
(embed → retrieve → Llama 3) → displays answer with source citations.
Also includes research report generation.
"""
import streamlit as st
from frontend.render_utils import render_answer


def render(api_post):
    st.header("💬 RAG Q&A")
    st.caption(
        "Ask questions about your ingested papers. "
        "The assistant retrieves relevant chunks and grounds its answer in the literature."
    )

    # ── Settings sidebar panel ─────────────────────────────────────────────
    with st.expander("⚙️ Query Settings", expanded=False):
        col1, col2 = st.columns(2)
        with col1:
            top_k = st.slider("Retrieved chunks (top-k)", 1, 15, 5)
            use_memory = st.checkbox("Use conversation memory", value=True)
        with col2:
            paper_filter = st.text_input(
                "Filter to paper IDs (comma-separated, leave blank for all)",
                placeholder="arxiv_2310.06825, ss_abc123",
            )

    paper_ids = (
        [p.strip() for p in paper_filter.split(",") if p.strip()]
        if paper_filter.strip() else None
    )

    st.divider()

    # ── Chat history display ───────────────────────────────────────────────
    for msg in st.session_state.chat_messages:
        with st.chat_message(msg["role"]):
            if msg["role"] == "assistant":
                render_answer(msg["content"])
            else:
                st.markdown(msg["content"])
            if msg.get("sources"):
                _render_sources(msg["sources"])

    # ── Chat input ─────────────────────────────────────────────────────────
    user_input = st.chat_input("Ask a research question...")

    if user_input:
        # Show user message immediately
        st.session_state.chat_messages.append({"role": "user", "content": user_input})
        with st.chat_message("user"):
            st.markdown(user_input)

        # Call RAG endpoint
        with st.chat_message("assistant"):
            with st.spinner("Retrieving context and generating answer..."):
                payload = {
                    "query": user_input,
                    "session_id": st.session_state.session_id,
                    "top_k": top_k,
                    "use_memory": use_memory,
                    "paper_ids": paper_ids,
                }
                result = api_post("/rag/query", payload, timeout=300)

            if result:
                answer = result.get("answer", "").strip()
                if not answer:
                    answer = "⚠️ The model did not return an answer. The model may still be loading — please try again in a few seconds."
                sources = result.get("sources", [])
                model = result.get("model_used", "unknown")

                render_answer(answer)
                st.caption(f"Model: `{model}` · Sources retrieved: {len(sources)}")

                if sources:
                    _render_sources(sources)

                st.session_state.chat_messages.append({
                    "role": "assistant",
                    "content": answer,
                    "sources": sources,
                })
            else:
                error_msg = "Failed to get a response from the backend."
                st.error(error_msg)
                st.session_state.chat_messages.append({
                    "role": "assistant",
                    "content": error_msg,
                })

    # ── Clear chat ─────────────────────────────────────────────────────────
    if st.session_state.chat_messages:
        st.divider()
        if st.button("🗑️ Clear conversation", use_container_width=False):
            st.session_state.chat_messages = []
            st.rerun()

    st.divider()

    # ── Report generation ──────────────────────────────────────────────────
    st.subheader("📋 Generate Research Report")
    st.caption("Generate a structured markdown report on a topic using ingested papers as context.")

    with st.form("report_form"):
        topic = st.text_input("Research topic", placeholder="e.g. Transformer attention mechanisms in NLP")
        col1, col2 = st.columns(2)
        with col1:
            max_length = st.slider("Max words", 200, 50000, 2000, step=500)
        with col2:
            fmt = st.selectbox("Format", ["markdown", "plain"])
        gen_submitted = st.form_submit_button("📝 Generate Report", use_container_width=True, type="primary")

    if gen_submitted and topic.strip():
        with st.spinner("Generating report..."):
            payload = {
                "topic": topic,
                "session_id": st.session_state.session_id,
                "paper_ids": paper_ids,
                "max_length": max_length,
                "format": fmt,
            }
            report_result = api_post("/report/generate", payload, timeout=3600)

        if report_result:
            st.subheader(f"Report: {report_result['topic']}")
            st.caption(
                f"Generated at {report_result.get('generated_at', 'N/A')} · "
                f"Sources: {len(report_result.get('sources_used', []))}"
            )
            if fmt == "markdown":
                render_answer(report_result["report"])
            else:
                st.text(report_result["report"])

            with st.expander("Sources used"):
                for src in report_result.get("sources_used", []):
                    st.code(src, language=None)

            st.download_button(
                "⬇️ Download Report",
                data=report_result["report"],
                file_name=f"report_{topic[:30].replace(' ', '_')}.md",
                mime="text/markdown",
            )


def _render_sources(sources: list):
    """Render retrieved source chunks in a collapsible block."""
    if not sources:
        return
    with st.expander(f"📚 {len(sources)} source(s) retrieved", expanded=False):
        for i, src in enumerate(sources):
            st.markdown(
                f"**[{i+1}]** `{src.get('paper_id', 'unknown')}` — "
                f"*{src.get('paper_title', '') or 'Untitled'}*  "
                f"Score: `{src.get('score', 0):.3f}`"
            )
            st.text(src.get("text", "")[:400] + ("..." if len(src.get("text", "")) > 400 else ""))
            if i < len(sources) - 1:
                st.markdown("---")
