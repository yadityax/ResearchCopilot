"""
PDF Ingestion Page
Upload a PDF file or paste raw text to chunk, embed, and store in the vector DB.
"""
import hashlib
import streamlit as st


def render(api_post, api_post_file):
    st.header("📄 PDF Ingestion")
    st.caption("Upload a PDF or paste text to embed and store in the ResearchCopilot knowledge base.")

    mode = st.radio("Input mode", ["📎 Upload PDF", "📝 Paste Text"], horizontal=True)
    st.divider()

    if mode == "📎 Upload PDF":
        _render_pdf_upload(api_post_file)
    else:
        _render_text_paste(api_post)

    # ── Ingested papers list ───────────────────────────────────────────────
    if st.session_state.get("ingested_papers"):
        st.divider()
        st.subheader(f"📚 Ingested Papers ({len(st.session_state.ingested_papers)})")
        for pid in st.session_state.ingested_papers:
            st.code(pid, language=None)


def _render_pdf_upload(api_post_file):
    with st.form("pdf_upload_form"):
        col1, col2 = st.columns(2)
        with col1:
            paper_id_input = st.text_input(
                "Paper ID",
                placeholder="e.g. arxiv_2310.06825  (leave blank to auto-generate)",
            )
        with col2:
            title = st.text_input("Paper Title *", placeholder="Attention Is All You Need")

        uploaded_file = st.file_uploader("Choose a PDF file", type=["pdf"])
        submitted = st.form_submit_button("📥 Ingest PDF", use_container_width=True, type="primary")

    if submitted:
        if not title.strip():
            st.error("Paper title is required.")
            return
        if not uploaded_file:
            st.error("Please upload a PDF file.")
            return

        pid = paper_id_input.strip() or _auto_id(title)

        with st.spinner("Extracting text, chunking and embedding..."):
            result = api_post_file(
                "/papers/ingest/pdf",
                files={"file": (uploaded_file.name, uploaded_file.getvalue(), "application/pdf")},
                data={"paper_id": pid, "title": title},
                timeout=120,
            )

        if result:
            _show_ingest_result(result)
            if result.get("status") == "success":
                if pid not in st.session_state.ingested_papers:
                    st.session_state.ingested_papers.append(pid)


def _render_text_paste(api_post):
    with st.form("text_ingest_form"):
        col1, col2 = st.columns(2)
        with col1:
            paper_id_input = st.text_input(
                "Paper ID",
                placeholder="e.g. my_paper_001  (leave blank to auto-generate)",
            )
        with col2:
            title = st.text_input("Paper Title *", placeholder="My Research Paper")

        content = st.text_area(
            "Paper text / abstract *",
            height=300,
            placeholder="Paste the full text or abstract here...",
        )
        submitted = st.form_submit_button("📥 Ingest Text", use_container_width=True, type="primary")

    if submitted:
        if not title.strip():
            st.error("Paper title is required.")
            return
        if not content.strip():
            st.error("Content cannot be empty.")
            return

        pid = paper_id_input.strip() or _auto_id(title)

        with st.spinner("Chunking and embedding..."):
            result = api_post(
                "/papers/ingest",
                {"paper_id": pid, "title": title, "content": content},
                timeout=120,
            )

        if result:
            _show_ingest_result(result)
            if result.get("status") == "success":
                if pid not in st.session_state.ingested_papers:
                    st.session_state.ingested_papers.append(pid)


def _show_ingest_result(result: dict):
    if result.get("status") == "success":
        col1, col2, col3 = st.columns(3)
        col1.metric("Status", "✅ Success")
        col2.metric("Chunks Created", result.get("chunks_created", 0))
        col3.metric("Embeddings Stored", result.get("embeddings_stored", 0))
        st.info(result.get("message", ""))
    else:
        st.error(f"Ingestion failed: {result.get('message', 'Unknown error')}")


def _auto_id(title: str) -> str:
    return "paper_" + hashlib.md5(title.lower().strip().encode()).hexdigest()[:8]
