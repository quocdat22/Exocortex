"""Streamlit demo UI for Exocortex RAG system with multi-turn conversational support."""

import httpx
import streamlit as st

API_BASE_URL = "http://localhost:8000"

st.set_page_config(
    page_title="Exocortex — Ebook RAG & Conversational Assistant",
    page_icon="🧠",
    layout="wide",
)


def api_request(method: str, endpoint: str, **kwargs) -> dict | None:
    """Make an HTTP API request to FastAPI backend."""
    url = f"{API_BASE_URL}{endpoint}"
    try:
        with httpx.Client(timeout=120.0) as client:
            response = getattr(client, method)(url, **kwargs)
            if response.status_code in (200, 409):
                return response.json()
            else:
                detail = "Unknown error"
                try:
                    detail = response.json().get("detail", str(response.text))
                except Exception:
                    detail = response.text
                st.error(f"API Error ({response.status_code}): {detail}")
                return None
    except httpx.ConnectError:
        st.error(f"Cannot connect to API at {API_BASE_URL}. Is the FastAPI server running?")
        return None
    except Exception as e:
        st.error(f"Request failed: {e}")
        return None


# Query existing sessions and initialize active session state before sidebar
sessions_resp = api_request("get", "/sessions")
sessions = sessions_resp.get("sessions", []) if sessions_resp else []

if not st.session_state.get("current_session_id"):
    if sessions:
        st.session_state["current_session_id"] = sessions[0]["id"]
    else:
        new_sess = api_request("post", "/sessions", json={})
        if new_sess:
            st.session_state["current_session_id"] = new_sess["id"]
            sessions_resp = api_request("get", "/sessions")
            sessions = sessions_resp.get("sessions", []) if sessions_resp else []

# --- Sidebar: Sessions & System Management ---
with st.sidebar:
    st.title("🧠 Exocortex")
    st.caption("Conversational RAG for English Ebooks")

    if st.button("➕ New Chat", type="primary", use_container_width=True):
        new_sess = api_request("post", "/sessions", json={})
        if new_sess:
            st.session_state["current_session_id"] = new_sess["id"]
            st.rerun()

    st.subheader("💬 Conversations")
    if sessions:
        for s in sessions:
            col_title, col_del = st.columns([4, 1])
            is_active = s["id"] == st.session_state.get("current_session_id")
            prefix = "👉 " if is_active else ""
            with col_title:
                if st.button(f"{prefix}{s['title'][:25]}", key=f"sess_{s['id']}", use_container_width=True):
                    st.session_state["current_session_id"] = s["id"]
                    st.rerun()
            with col_del:
                if st.button("🗑️", key=f"del_sess_{s['id']}"):
                    api_request("delete", f"/sessions/{s['id']}")
                    if st.session_state.get("current_session_id") == s["id"]:
                        st.session_state["current_session_id"] = None
                    st.rerun()
    else:
        st.caption("No conversations yet.")

    st.divider()

    with st.expander("⚙️ System Health & Status"):
        if st.button("🔄 Check Health"):
            health = api_request("get", "/health")
            if health:
                status = health.get("status", "unknown")
                if status == "healthy":
                    st.success("System is healthy")
                else:
                    st.warning("System is degraded")
                col1, col2, col3 = st.columns(3)
                col1.metric("Ollama", "✅" if health.get("ollama") else "❌")
                col2.metric("ChromaDB", "✅" if health.get("chromadb") else "❌")
                col3.metric("LLM", "✅" if health.get("llm") else "❌")
                if "details" in health:
                    st.json(health["details"])

    with st.expander("📚 Indexed Documents"):
        docs = api_request("get", "/documents")
        if docs:
            st.metric("Documents", docs.get("total_documents", 0))
            st.metric("Chunks", docs.get("total_chunks", 0))
            for doc in docs.get("documents", []):
                st.markdown(f"**{doc['filename']}** ({doc.get('chunk_count', 0)} chunks)")
                if st.button("🗑️ Delete Doc", key=f"del_doc_{doc['document_id']}"):
                    api_request("delete", f"/documents/{doc['document_id']}")
                    st.rerun()


# --- Main Area ---
tab_chat, tab_upload = st.tabs(["💬 Chat", "📤 Upload Ebook"])

with tab_chat:
    current_session = None
    curr_id = st.session_state.get("current_session_id")
    if curr_id:
        current_session = api_request("get", f"/sessions/{curr_id}")
        if not current_session:
            st.session_state["current_session_id"] = None

    if current_session:
        st.header(f"💬 {current_session.get('title', 'Chat Session')}")

        # Render message history
        for msg in current_session.get("messages", []):
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])
                if msg["role"] == "assistant":
                    with st.expander("🔍 Details & Citations"):
                        st.caption(
                            f"**Model:** {msg.get('model', 'N/A')} | **Retrieval:** {'Active' if msg.get('needs_retrieval') else 'Bypassed'}"
                        )
                        if msg.get("standalone_query"):
                            st.caption(f"**Rewritten Query:** {msg['standalone_query']}")
                        if msg.get("sources"):
                            st.markdown("**Sources:**")
                            for idx, src in enumerate(msg["sources"], 1):
                                preview = src.get("text_preview") or src.get("text") or ""
                                st.markdown(
                                    f"- **Source {idx}:** `{src.get('filename')}` (p. {src.get('page_numbers')})\n> {preview}"
                                )
                        if msg.get("usage"):
                            st.json(msg["usage"])

    # Chat Input
    user_input = st.chat_input("Ask a question about your books...")
    if user_input:
        if current_session:
            # Render user message optimistically
            with st.chat_message("user"):
                st.markdown(user_input)

            # Send to chat endpoint
            with st.spinner("Thinking & searching books..."):
                resp = api_request(
                    "post",
                    f"/sessions/{current_session['id']}/chat",
                    json={"question": user_input},
                )
            if resp:
                st.rerun()

with tab_upload:
    st.header("Upload Ebook (PDF)")
    st.caption("Upload an English ebook in PDF format. The system will extract text, create chunks, and index them.")

    uploaded_file = st.file_uploader(
        "Choose a PDF file",
        type=["pdf"],
        help="Only PDF files with extractable text are supported. Scanned/image PDFs are not supported.",
    )
    if uploaded_file is not None:
        file_key = f"{uploaded_file.name}_{uploaded_file.size}"
        st.info(f"File: {uploaded_file.name} ({uploaded_file.size / 1024 / 1024:.2f} MB)")

        strategy = st.selectbox(
            "Chunking Strategy",
            ["recursive", "fixed", "sentence_paragraph", "semantic"],
            index=0,
            help="Choose the chunking strategy (optimal: recursive character splitting).",
        )

        dup_state = st.session_state.get(f"dup_detected_{file_key}")
        if dup_state:
            st.warning("⚠️ **Duplicate Document Detected!**")
            st.markdown(
                f"File content matches existing document(s) (Hash: `{dup_state.get('file_hash', '')[:12]}...`):"
            )
            for doc in dup_state.get("existing_documents", []):
                st.markdown(
                    f"- 📄 **{doc['filename']}** (ID: `{doc['document_id']}`, Chunks: {doc.get('chunk_count', 'N/A')})"
                )
            st.caption("You can force ingestion to re-index or add duplicate content.")

            col1, col2 = st.columns([1, 1])
            with col1:
                if st.button("⚡ Force Ingest & Index", type="primary"):
                    with st.spinner(f"Ingesting {uploaded_file.name} (Force)..."):
                        files = {"file": (uploaded_file.name, uploaded_file.getvalue(), "application/pdf")}
                        result = api_request(
                            "post",
                            "/ingest",
                            files=files,
                            params={"strategy": strategy, "force": "true"},
                        )
                    if result and not result.get("duplicate"):
                        st.session_state.pop(f"dup_detected_{file_key}", None)
                        st.success(result.get("message", "Ingested successfully"))
                        st.rerun()
            with col2:
                if st.button("❌ Cancel"):
                    st.session_state.pop(f"dup_detected_{file_key}", None)
                    st.rerun()
        else:
            if st.button("📥 Ingest & Index"):
                with st.spinner(f"Processing {uploaded_file.name} with '{strategy}' chunking..."):
                    files = {"file": (uploaded_file.name, uploaded_file.getvalue(), "application/pdf")}
                    result = api_request(
                        "post",
                        "/ingest",
                        files=files,
                        params={"strategy": strategy, "force": "false"},
                    )
                if result:
                    if result.get("duplicate"):
                        st.session_state[f"dup_detected_{file_key}"] = result
                        st.rerun()
                    else:
                        st.success(result.get("message", "Ingested successfully"))
                        st.rerun()

