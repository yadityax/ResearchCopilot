"""
Memory History Page
View conversation history for the current session.
Supports both chronological list view and semantic search within memory.
"""
import streamlit as st


def render(api_post):
    st.header("🧠 Adaptive Memory")
    st.caption(
        "The adaptive memory system stores every conversation turn in DynamoDB (full history) "
        "and ChromaDB (semantic search). Browse or semantically search your session history."
    )

    session_id = st.session_state.session_id
    st.info(f"Current session: `{session_id}`")

    # ── Manual memory entry ────────────────────────────────────────────────
    with st.expander("➕ Add memory entry manually", expanded=False):
        with st.form("add_memory_form"):
            role = st.selectbox("Role", ["user", "assistant"])
            content = st.text_area("Content", height=100)
            if st.form_submit_button("Store"):
                if content.strip():
                    result = api_post(
                        "/memory/store",
                        {"session_id": session_id, "role": role, "content": content, "metadata": {}},
                        timeout=30,
                    )
                    if result:
                        st.success(f"Stored entry `{result.get('entry_id', '')[:16]}...`")

    st.divider()

    # ── Retrieve mode toggle ───────────────────────────────────────────────
    mode = st.radio(
        "Retrieval mode",
        ["📜 Chronological (last N turns)", "🔍 Semantic search"],
        horizontal=True,
    )

    if mode == "📜 Chronological (last N turns)":
        _render_chronological(api_post, session_id)
    else:
        _render_semantic(api_post, session_id)

    # ── Visualise in-session chat messages ─────────────────────────────────
    if st.session_state.chat_messages:
        st.divider()
        st.subheader("💬 In-Session Chat Log")
        st.caption("Messages from the current browser session (not yet persisted if backend is offline).")
        for i, msg in enumerate(st.session_state.chat_messages):
            icon = "🧑" if msg["role"] == "user" else "🤖"
            st.markdown(f"**{icon} {msg['role'].capitalize()}:** {msg['content'][:300]}"
                        + ("..." if len(msg["content"]) > 300 else ""))


def _render_chronological(api_post, session_id: str):
    col1, col2 = st.columns([2, 1])
    with col1:
        limit = st.slider("Number of turns to retrieve", 3, 50, 10)
    with col2:
        if st.button("🔄 Refresh", use_container_width=True):
            st.rerun()

    result = api_post(
        "/memory/retrieve",
        {"session_id": session_id, "limit": limit},
        timeout=20,
    )

    if result is None:
        return

    entries = result.get("entries", [])
    total = result.get("total", 0)

    st.caption(f"Retrieved {len(entries)} / {total} entries")

    if not entries:
        st.info("No memory entries found for this session. Start chatting in the Q&A tab!")
        return

    for entry in entries:
        role = entry.get("role", "unknown")
        icon = "🧑" if role == "user" else "🤖"
        ts = entry.get("timestamp", "")[:19].replace("T", " ")
        with st.container():
            st.markdown(
                f"{icon} **{role.capitalize()}**  <span style='color:grey;font-size:0.8em'>{ts}</span>",
                unsafe_allow_html=True,
            )
            st.text(entry.get("content", "")[:500])
            st.caption(f"Entry ID: `{entry.get('entry_id', '')[:16]}...`")
            st.divider()


def _render_semantic(api_post, session_id: str):
    with st.form("semantic_memory_form"):
        query = st.text_input("Search your conversation history", placeholder="What did we discuss about BERT?")
        limit = st.slider("Max results", 1, 20, 5)
        submitted = st.form_submit_button("🔍 Search Memory", use_container_width=True, type="primary")

    if submitted and query.strip():
        with st.spinner("Searching memory..."):
            result = api_post(
                "/memory/retrieve",
                {"session_id": session_id, "limit": limit, "query": query},
                timeout=30,
            )

        if result is None:
            return

        entries = result.get("entries", [])
        st.caption(f"Found {len(entries)} semantically relevant turns")

        if not entries:
            st.info("No relevant memory found for that query.")
            return

        for entry in entries:
            role = entry.get("role", "unknown")
            icon = "🧑" if role == "user" else "🤖"
            score = entry.get("metadata", {}).get("score", None)
            score_str = f" · Similarity: `{score:.3f}`" if score is not None else ""
            st.markdown(f"{icon} **{role.capitalize()}**{score_str}")
            st.text(entry.get("content", "")[:500])
            st.divider()
