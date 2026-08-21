"""Retrieval engine — orchestrates query → embed → search → LLM answer.

This is the main entry point for the RAG pipeline. It ties together
the embedding client, vector store, and LLM client to answer user queries.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

from exocortex.config import Settings
from exocortex.embedding import EmbeddingClient
from exocortex.llm import LLMClient
from exocortex.session import SessionStore
from exocortex.vectorstore import VectorStore

logger = logging.getLogger(__name__)


@dataclass
class ChatResponse:
    """Response to a conversational multi-turn query within a session."""

    answer: str
    sources: list[dict]
    query: str
    standalone_query: str
    needs_retrieval: bool
    session_id: str
    num_chunks_retrieved: int
    model: str
    usage: dict | None = None


@dataclass
class QueryResponse:
    """Complete response to a user query."""

    answer: str  # LLM-generated answer
    sources: list[dict]  # Source documents with metadata
    query: str  # Original query
    num_chunks_retrieved: int  # How many chunks were retrieved
    model: str  # LLM model used
    usage: dict | None = None  # Token usage stats


class DuplicateDocumentError(Exception):
    """Raised when an uploaded document has identical content to an existing document."""

    def __init__(self, message: str, file_hash: str, existing_documents: list[dict]) -> None:
        super().__init__(message)
        self.file_hash = file_hash
        self.existing_documents = existing_documents


class RAGEngine:
    """Retrieval-Augmented Generation engine.

    Orchestrates the full RAG pipeline:
    1. Embed the user query (via Ollama)
    2. Search ChromaDB for relevant chunks
    3. Send context + query to DeepSeek LLM
    4. Return structured response with sources

    Args:
        settings: Application settings.
        embedding_client: Pre-initialized embedding client (optional).
        vector_store: Pre-initialized vector store (optional).
        llm_client: Pre-initialized LLM client (optional).
    """

    def __init__(
        self,
        settings: Settings,
        embedding_client: EmbeddingClient | None = None,
        vector_store: VectorStore | None = None,
        llm_client: LLMClient | None = None,
        session_store: SessionStore | None = None,
    ) -> None:
        self.settings = settings
        self.embedding_client = embedding_client or EmbeddingClient(settings)
        self.vector_store = vector_store or VectorStore(settings)
        self.llm_client = llm_client or LLMClient(settings)
        self.session_store = session_store or SessionStore(settings.sessions_db_path)

    def chat(self, session_id: str, question: str) -> ChatResponse:
        """Process a conversational question with history, rewrite, and retrieval routing.

        Args:
            session_id: The UUID of the conversation session.
            question: The user's latest follow-up question.

        Returns:
            ChatResponse with the answer, citations, standalone query, and session info.
        """
        logger.info(f"Processing chat turn for session {session_id}: {question[:100]}...")

        # 1. Ensure session exists
        session = self.session_store.get_session(session_id, include_messages=False)
        is_new_session = False
        if session is None:
            session = self.session_store.create_session(title=question[:40].strip() or "New Conversation")
            session_id = session.id
            is_new_session = True

        # 2. Fetch recent conversation history (sliding window: 2 * window_size)
        history_limit = max(1, self.settings.chat_history_window * 2)
        recent_history = self.session_store.get_recent_messages(session_id, limit=history_limit)

        # 3. Rewrite query and determine routing
        standalone_query, needs_retrieval = self.llm_client.rewrite_and_route(
            history=recent_history,
            question=question,
        )
        logger.info(
            f"Query routed: needs_retrieval={needs_retrieval}, standalone_query='{standalone_query[:100]}'"
        )

        # 4. Perform vector retrieval if needed
        search_results = []
        if needs_retrieval and self.vector_store.count() > 0:
            query_embedding = self.embedding_client.embed_query(standalone_query)
            search_results = self.vector_store.query(
                query_embedding=query_embedding,
                top_k=self.settings.top_k,
            )
            logger.info(f"Retrieved {len(search_results)} chunks for standalone query")

        # 5. Generate answer using LLM with context + history
        llm_response = self.llm_client.generate_with_history(
            messages_history=recent_history,
            query=question,
            search_results=search_results,
        )

        # 6. Persist user and assistant turns to database
        self.session_store.add_message(
            session_id=session_id,
            role="user",
            content=question,
        )
        self.session_store.add_message(
            session_id=session_id,
            role="assistant",
            content=llm_response.answer,
            standalone_query=standalone_query,
            needs_retrieval=needs_retrieval,
            sources=llm_response.sources,
            model=llm_response.model,
            usage=llm_response.usage,
        )

        # Auto-update session title if it was default or initial placeholder
        if not is_new_session and (session.title in ("New Conversation", "Initial") or not session.title):
            clean_title = question.strip().split("\n")[0][:40].strip()
            if clean_title:
                self.session_store.update_session_title(session_id, clean_title)

        return ChatResponse(
            answer=llm_response.answer,
            sources=llm_response.sources,
            query=question,
            standalone_query=standalone_query,
            needs_retrieval=needs_retrieval,
            session_id=session_id,
            num_chunks_retrieved=len(search_results),
            model=llm_response.model,
            usage=llm_response.usage,
        )

    def query(self, question: str) -> QueryResponse:
        """Process a user question through the full RAG pipeline.

        Args:
            question: The user's question in English.

        Returns:
            QueryResponse with the answer and source information.

        Raises:
            ConnectionError: If Ollama is not reachable.
            RuntimeError: If any pipeline step fails.
        """
        logger.info(f"Processing query: {question[:100]}...")

        # Step 1: Embed the query
        query_embedding = self.embedding_client.embed_query(question)
        logger.info(f"Query embedded (dim={len(query_embedding)})")

        # Step 2: Search vector store
        search_results = self.vector_store.query(
            query_embedding=query_embedding,
            top_k=self.settings.top_k,
        )
        logger.info(f"Retrieved {len(search_results)} chunks")

        # Step 3: Generate answer with LLM
        llm_response = self.llm_client.generate(
            query=question,
            search_results=search_results,
        )
        logger.info(f"LLM generated answer ({len(llm_response.answer)} chars)")

        return QueryResponse(
            answer=llm_response.answer,
            sources=llm_response.sources,
            query=question,
            num_chunks_retrieved=len(search_results),
            model=llm_response.model,
            usage=llm_response.usage,
        )

    def ingest_and_index(
        self,
        pdf_path: str | Path,
        strategy: str | None = None,
        force: bool = False,
    ) -> dict:
        """Full ingestion and indexing pipeline for a PDF.

        Extracts text, chunks, computes embeddings, and stores in ChromaDB.

        Args:
            pdf_path: Path to the PDF file.
            strategy: Optional chunking strategy name. If None, uses self.settings.chunking_strategy.
            force: If True, bypass duplicate detection and ingest anyway.

        Returns:
            Dict with ingestion results: {filename, document_id, chunk_count}.

        Raises:
            DuplicateDocumentError: If document content matches existing document and force=False.
        """
        from exocortex.ingestion import compute_file_hash, ingest_pdf

        path = Path(pdf_path)
        file_hash = compute_file_hash(path)

        if not force:
            existing_docs = self.vector_store.find_by_file_hash(file_hash)
            if existing_docs:
                filenames = ", ".join(d["filename"] for d in existing_docs)
                raise DuplicateDocumentError(
                    f"Duplicate document detected: content matches existing document(s) ({filenames})",
                    file_hash=file_hash,
                    existing_documents=existing_docs,
                )

        chosen_strategy = strategy or self.settings.chunking_strategy
        chunks = ingest_pdf(
            pdf_path=path,
            chunk_size=self.settings.chunk_size,
            chunk_overlap=self.settings.chunk_overlap,
            strategy=chosen_strategy,
        )

        if not chunks:
            return {
                "filename": path.name,
                "document_id": "",
                "chunk_count": 0,
            }

        # Embed all chunks
        texts = [c.text for c in chunks]
        embeddings = self.embedding_client.embed_documents(texts)

        # Store in vector store
        self.vector_store.add_chunks(chunks, embeddings)

        logger.info(
            f"Ingested {path.name} (strategy={chosen_strategy}): {len(chunks)} chunks embedded and stored"
        )

        return {
            "filename": path.name,
            "document_id": chunks[0].document_id,
            "chunk_count": len(chunks),
        }
