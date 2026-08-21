"""Streamlit demo UI for Exocortex RAG system.

Provides a web interface for:
- Uploading PDF ebooks
- Viewing indexed documents
- Asking questions with source-cited answers

Requires the FastAPI server to be running on localhost:8000.
"""

import httpx
import streamlit as st

# --- Configuration ---
API_BASE_URL = "http://localhost:8000"

st.set_page_config(
    page_title="Exocortex — Ebook RAG",
    page_icon="🧠",
    layout="wide",
)


def api_request(method: str, endpoint: str, **kwargs) -> dict | None:
    """Make an API request to the FastAPI backend.

    Args:
        method: HTTP method (get, post, delete).
        endpoint: API endpoint path (e.g., '/health').
        **kwargs: Additional arguments for httpx.

    Returns:
        Response JSON dict, or None if request failed.
    """
    url = f"{API_BASE_URL}{endpoint}"
    try:
        with httpx.Client(timeout=120.0) as client:
            response = getattr(client, method)(url, **kwargs)
            if response.status_code in (200, 409):
                return response.json()
            else:
                st.error(
                    f"API Error ({response.status_code}): {response.json().get('detail', 'Unknown error')}"
                )
                return None
    except httpx.ConnectError:
        st.error(
            f"Cannot connect to API at {API_BASE_URL}. Is the FastAPI server running?"
        )
        return None
    except Exception as e:
        st.error(f"Request failed: {e}")
        return None


# --- Sidebar: System Status ---
with st.sidebar:
    st.title("🧠 Exocortex")
    st.caption("RAG System for English Ebooks")

    st.divider()

    # Health check
    if st.button("🔄 Check System Health"):
        health = api_request("get", "/health")
        if health:
            status = health["status"]
            if status == "healthy":
                st.success("System is healthy")
            else:
                st.warning("System is degraded")

            col1, col2, col3 = st.columns(3)
            col1.metric("Ollama", "✅" if health["ollama"] else "❌")
            col2.metric("ChromaDB", "✅" if health["chromadb"] else "❌")
            col3.metric("LLM", "✅" if health["llm"] else "❌")

            st.json(health["details"])

    st.divider()

    # Document list
    st.subheader("📚 Indexed Documents")
    docs = api_request("get", "/documents")
    if docs:
        if docs["total_documents"] == 0:
            st.info("No documents indexed yet. Upload a PDF below.")
        else:
            st.metric("Total Documents", docs["total_documents"])
            st.metric("Total Chunks", docs["total_chunks"])
            for doc in docs["documents"]:
                with st.expander(f"📄 {doc['filename']}"):
                    st.text(f"ID: {doc['document_id']}")
                    st.text(f"Chunks: {doc['chunk_count']}")
                    if doc.get("file_hash"):
                        st.caption(f"Hash: {doc['file_hash'][:12]}...")
                    if st.button("🗑️ Delete", key=f"del_{doc['document_id']}"):
                        result = api_request(
                            "delete", f"/documents/{doc['document_id']}"
                        )
                        if result:
                            st.success(result["message"])
                            st.rerun()


# --- Main Area ---
tab_query, tab_upload = st.tabs(["💬 Ask Questions", "📤 Upload Ebook"])

# --- Tab 1: Query ---
with tab_query:
    st.header("Ask a Question")
    st.caption(
        "Ask questions about your indexed ebooks. Answers are grounded in the document content."
    )

    question = st.text_input(
        "Your question:",
        placeholder="e.g., What are the main concepts discussed in chapter 1?",
    )

    if st.button("🔍 Ask", disabled=not question):
        with st.spinner("Searching documents and generating answer..."):
            result = api_request("post", "/query", json={"question": question})

        if result:
            # Display answer
            st.subheader("Answer")
            st.markdown(result["answer"])

            # Display sources
            st.subheader("Sources")
            st.caption(
                f"{result['num_chunks_retrieved']} chunks retrieved | Model: {result['model']}"
            )

            for i, source in enumerate(result["sources"], 1):
                with st.expander(
                    f"📖 Source {i}: {source['filename']} (page {source['page_numbers']})"
                ):
                    chunk_text = source.get("text") or source.get("text_preview", "")
                    st.markdown(chunk_text)

            # Display token usage
            if result.get("usage"):
                with st.expander("📊 Token Usage"):
                    st.json(result["usage"])

# --- Tab 2: Upload ---
with tab_upload:
    st.header("Upload Ebook (PDF)")
    st.caption(
        "Upload an English ebook in PDF format. The system will extract text, create chunks, and index them."
    )

    uploaded_file = st.file_uploader(
        "Choose a PDF file",
        type=["pdf"],
        help="Only PDF files with extractable text are supported. Scanned/image PDFs are not supported.",
    )

    if uploaded_file is not None:
        file_key = f"{uploaded_file.name}_{uploaded_file.size}"
        st.info(
            f"File: {uploaded_file.name} ({uploaded_file.size / 1024 / 1024:.2f} MB)"
        )

        strategy = st.selectbox(
            "Chunking Strategy",
            ["recursive", "fixed", "sentence_paragraph", "semantic"],
            index=0,
            help="Choose the chunking strategy (optimal: recursive character splitting).",
        )

        dup_state = st.session_state.get(f"dup_detected_{file_key}")

        if dup_state:
            st.warning("⚠️ **Phát hiện tài liệu trùng lặp (Duplicate Document Detected)!**")
            st.markdown(
                f"Nội dung của file này trùng khớp (Hash: `{dup_state.get('file_hash', '')[:12]}...`) "
                f"với các tài liệu đã tồn tại trong hệ thống:"
            )
            for doc in dup_state.get("existing_documents", []):
                st.markdown(
                    f"- 📄 **{doc['filename']}** (ID: `{doc['document_id']}`, Chunks: {doc['chunk_count']})"
                )
            st.caption(
                "Bạn vẫn có thể tiếp tục Ingest & Index nếu muốn index lại hoặc lưu dưới dạng tài liệu bổ sung."
            )

            col1, col2 = st.columns([1, 1])
            with col1:
                if st.button("⚡ Vẫn tiếp tục (Force Ingest & Index)", type="primary"):
                    with st.spinner(f"Đang Ingest {uploaded_file.name} (Force Ingest)..."):
                        files = {
                            "file": (
                                uploaded_file.name,
                                uploaded_file.getvalue(),
                                "application/pdf",
                            )
                        }
                        result = api_request(
                            "post",
                            "/ingest",
                            files=files,
                            params={"strategy": strategy, "force": "true"},
                        )

                    if result and not result.get("duplicate"):
                        st.session_state.pop(f"dup_detected_{file_key}", None)
                        st.success(result.get("message", "Ingested successfully"))
                        st.json(
                            {
                                "filename": result["filename"],
                                "document_id": result["document_id"],
                                "chunk_count": result["chunk_count"],
                                "strategy": strategy,
                                "forced": True,
                            }
                        )
                        st.rerun()

            with col2:
                if st.button("❌ Hủy bỏ"):
                    st.session_state.pop(f"dup_detected_{file_key}", None)
                    st.rerun()

        else:
            if st.button("📥 Ingest & Index"):
                with st.spinner(f"Processing {uploaded_file.name} with '{strategy}' chunking..."):
                    files = {
                        "file": (
                            uploaded_file.name,
                            uploaded_file.getvalue(),
                            "application/pdf",
                        )
                    }
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
                        st.json(
                            {
                                "filename": result["filename"],
                                "document_id": result["document_id"],
                                "chunk_count": result["chunk_count"],
                                "strategy": strategy,
                            }
                        )
                        st.rerun()  # Refresh sidebar document list
