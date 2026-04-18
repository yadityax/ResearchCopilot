"""
ResearchCopilot — Streamlit Frontend
ChatGPT-style: persistent sessions per user, sidebar chat list, history reload.
"""
import uuid
import os
import streamlit as st
import requests

st.set_page_config(
    page_title="ResearchCopilot",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("<style>[data-testid='stSidebarNav']{display:none}</style>", unsafe_allow_html=True)

API_BASE = os.getenv("API_BASE_URL", "http://localhost:8000") + "/api/v1"


# ── API helpers ────────────────────────────────────────────────────────────

def api_post(endpoint: str, payload: dict, timeout: int = 60):
    try:
        r = requests.post(f"{API_BASE}{endpoint}", json=payload, timeout=timeout)
        r.raise_for_status()
        return r.json()
    except requests.exceptions.ConnectionError:
        st.error("Cannot reach backend.")
        return None
    except requests.exceptions.HTTPError as e:
        st.error(f"API error {e.response.status_code}: {e.response.text[:200]}")
        return None
    except Exception as e:
        st.error(f"Error: {e}")
        return None


def api_get(endpoint: str, timeout: int = 10):
    try:
        r = requests.get(f"{API_BASE}{endpoint}", timeout=timeout)
        r.raise_for_status()
        return r.json()
    except Exception:
        return None


def api_put(endpoint: str, payload: dict, timeout: int = 10):
    try:
        r = requests.put(f"{API_BASE}{endpoint}", json=payload, timeout=timeout)
        r.raise_for_status()
        return r.json()
    except Exception:
        return None


def api_delete(endpoint: str, timeout: int = 10):
    try:
        r = requests.delete(f"{API_BASE}{endpoint}", timeout=timeout)
        r.raise_for_status()
        return r.json()
    except Exception:
        return None


def api_post_file(endpoint: str, files: dict, data: dict, timeout: int = 120):
    try:
        r = requests.post(f"{API_BASE}{endpoint}", files=files, data=data, timeout=timeout)
        r.raise_for_status()
        return r.json()
    except Exception as e:
        st.error(f"Error: {e}")
        return None


# ── Session helpers ────────────────────────────────────────────────────────

def load_session_history(session_id: str):
    """Load chat history for a session from backend memory."""
    resp = api_post("/memory/retrieve", {"session_id": session_id, "limit": 50})
    if not resp:
        return []
    messages = []
    for entry in resp.get("entries", []):
        messages.append({"role": entry["role"], "content": entry["content"]})
    return messages


def create_session_on_backend(user_id: str, session_id: str, name: str = "New Chat"):
    api_post("/memory/sessions", {
        "user_id": user_id,
        "session_id": session_id,
        "session_name": name,
    })


def fetch_user_sessions(user_id: str):
    resp = api_get(f"/memory/sessions/{user_id}")
    if resp:
        return resp.get("sessions", [])
    return []


def rename_session_on_backend(user_id: str, session_id: str, name: str):
    api_put(f"/memory/sessions/{user_id}/{session_id}/rename", {"session_name": name})


# ── Login screen ───────────────────────────────────────────────────────────

def show_login():
    st.markdown("## 👋 Welcome to ResearchCopilot")
    st.markdown("Enter a username to get started. Your chats will be saved under this name.")
    col1, col2 = st.columns([3, 1])
    with col1:
        username = st.text_input("Username", placeholder="e.g. aditya", label_visibility="collapsed")
    with col2:
        if st.button("Continue", type="primary", use_container_width=True):
            if username.strip():
                st.session_state.user_id = username.strip().lower().replace(" ", "_")
                st.rerun()
            else:
                st.error("Please enter a username.")


# ── Init state ─────────────────────────────────────────────────────────────

def init_state():
    if "user_id" not in st.session_state:
        st.session_state.user_id = None
    if "session_id" not in st.session_state:
        st.session_state.session_id = None
    if "chat_messages" not in st.session_state:
        st.session_state.chat_messages = []
    if "ingested_papers" not in st.session_state:
        st.session_state.ingested_papers = []
    if "active_tab" not in st.session_state:
        st.session_state.active_tab = "chat"


init_state()

# ── Not logged in ──────────────────────────────────────────────────────────

if not st.session_state.user_id:
    show_login()
    st.stop()

user_id = st.session_state.user_id

# ── Sidebar: chat history list ─────────────────────────────────────────────

with st.sidebar:
    st.markdown(f"### 🤖 ResearchCopilot")
    st.caption(f"Logged in as **{user_id}**")
    if st.button("🚪 Switch User", use_container_width=True):
        for key in list(st.session_state.keys()):
            del st.session_state[key]
        st.rerun()

    st.divider()

    # Backend status
    health = api_get("/health".replace("/api/v1", ""), timeout=5)
    if health and health.get("status") == "ok":
        st.success("Backend Online", icon="✅")
    else:
        st.warning("Backend Offline", icon="⚠️")

    st.divider()

    # New chat button
    if st.button("✏️ New Chat", type="primary", use_container_width=True):
        new_sid = str(uuid.uuid4())
        st.session_state.session_id = new_sid
        st.session_state.chat_messages = []
        create_session_on_backend(user_id, new_sid, "New Chat")
        st.rerun()

    st.markdown("#### 💬 Your Chats")

    sessions = fetch_user_sessions(user_id)

    if not sessions:
        st.caption("No chats yet. Start a new chat above.")
    else:
        for sesh in sessions:
            sid = sesh["session_id"]
            name = sesh.get("session_name", "New Chat")
            count = sesh.get("message_count", 0)
            is_active = sid == st.session_state.session_id

            col_btn, col_del = st.columns([5, 1])
            with col_btn:
                label = f"**{name}**" if is_active else name
                if st.button(
                    f"{'▶ ' if is_active else ''}{name[:30]}",
                    key=f"sess_{sid}",
                    use_container_width=True,
                    type="primary" if is_active else "secondary",
                ):
                    st.session_state.session_id = sid
                    st.session_state.chat_messages = load_session_history(sid)
                    st.rerun()
            with col_del:
                if st.button("🗑", key=f"del_{sid}", help="Delete chat"):
                    api_delete(f"/memory/sessions/{user_id}/{sid}")
                    if st.session_state.session_id == sid:
                        st.session_state.session_id = None
                        st.session_state.chat_messages = []
                    st.rerun()

    st.divider()
    st.caption("Built for MLOps Project — April 2026")

# ── Auto-create session if none active ────────────────────────────────────

if not st.session_state.session_id:
    new_sid = str(uuid.uuid4())
    st.session_state.session_id = new_sid
    st.session_state.chat_messages = []
    create_session_on_backend(user_id, new_sid, "New Chat")

session_id = st.session_state.session_id

# ── Main tabs ──────────────────────────────────────────────────────────────

tab1, tab2, tab3, tab4 = st.tabs([
    "💬 Research Chat",
    "🔎 Paper Discovery",
    "📄 PDF Ingestion",
    "🧠 Memory History",
])

from pages.paper_discovery import render as render_discovery
from pages.pdf_ingestion   import render as render_ingestion
from pages.memory_history  import render as render_memory
from frontend.render_utils import render_answer

# ── Tab 1: Main RAG Chat (ChatGPT-style) ───────────────────────────────────

with tab1:
    # Session rename
    col_title, col_rename = st.columns([5, 2])
    with col_title:
        current_name = next(
            (s["session_name"] for s in sessions if s["session_id"] == session_id),
            "New Chat"
        )
        st.subheader(f"💬 {current_name}")
    with col_rename:
        with st.expander("✏️ Rename"):
            new_name = st.text_input("Chat name", value=current_name, label_visibility="collapsed")
            if st.button("Save", key="rename_btn"):
                rename_session_on_backend(user_id, session_id, new_name)
                st.rerun()

    st.divider()

    # Chat history
    for msg in st.session_state.chat_messages:
        with st.chat_message(msg["role"]):
            if msg["role"] == "assistant":
                render_answer(msg["content"])
            else:
                st.markdown(msg["content"])
            if msg.get("sources"):
                _render_sources = lambda sources: None  # defined below
                with st.expander(f"📚 {len(msg['sources'])} source(s)", expanded=False):
                    for i, src in enumerate(msg["sources"]):
                        st.markdown(f"**[{i+1}]** *{src.get('paper_title','') or src.get('paper_id','')}* — Score: `{src.get('score',0):.3f}`")
                        st.text(src.get("text","")[:300] + "...")

    # Chat input
    user_input = st.chat_input("Ask a research question...")

    if user_input:
        # Auto-name session from first message
        if not st.session_state.chat_messages and current_name == "New Chat":
            auto_name = user_input[:40].strip()
            rename_session_on_backend(user_id, session_id, auto_name)

        st.session_state.chat_messages.append({"role": "user", "content": user_input})
        with st.chat_message("user"):
            st.markdown(user_input)

        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                result = api_post("/rag/query", {
                    "query": user_input,
                    "session_id": session_id,
                    "user_id": user_id,
                    "top_k": 8,
                    "use_memory": True,
                }, timeout=300)

            if result:
                answer = result.get("answer", "").strip() or "⚠️ No answer returned. Try again."
                sources = result.get("sources", [])
                model = result.get("model_used", "unknown")

                render_answer(answer)
                st.caption(f"Model: `{model}` · Sources: {len(sources)}")

                if sources:
                    with st.expander(f"📚 {len(sources)} source(s)", expanded=False):
                        for i, src in enumerate(sources):
                            st.markdown(f"**[{i+1}]** *{src.get('paper_title','') or src.get('paper_id','')}* — Score: `{src.get('score',0):.3f}`")
                            st.text(src.get("text","")[:300] + "...")

                st.session_state.chat_messages.append({
                    "role": "assistant",
                    "content": answer,
                    "sources": sources,
                })

# ── Other tabs ────────────────────────────────────────────────────────────

with tab2:
    render_discovery(api_post, API_BASE)

with tab3:
    render_ingestion(api_post, api_post_file)

with tab4:
    render_memory(api_post)
