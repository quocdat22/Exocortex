"""Retrieval engine — orchestrates query → embed → search → LLM answer.

This is the main entry point for the RAG pipeline. It ties together
the embedding client, vector store, and LLM client to answer user queries.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

from exocortex.config import Settings
from exocortex.embedding import EmbeddingClient
from exocortex.llm import LLMClient, LLMResponse
from exocortex.vectorstore import SearchResult, VectorStore

logger = logging.getLogger(__name__)


@dataclass
class QueryResponse:
    """Complete response to a user query."""

    answer: str  # LLM-generated answer
    sources: list[dict]  # Source documents with metadata
    query: str  # Original query
    num_chunks_retrieved: int  # How many chunks were retrieved
    model: str  # LLM model used
    usage: dict | None = None  # Token usage stats


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
    ) -> None:
        self.settings = settings
        self.embedding_client = embedding_client or EmbeddingClient(settings)
        self.vector_store = vector_store or VectorStore(settings)
        self.llm_client = llm_client or LLMClient(settings)

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

    def ingest_and_index(self, pdf_path: str | object) -> dict:
        """Convenience method: ingest a PDF and index it in the vector store.

        Combines Phase 2 (ingestion) and Phase 3 (embedding + storage).

        Args:
            pdf_path: Path to the PDF file.

        Returns:
            Dict with ingestion results: {filename, document_id, chunk_count}.
        """
        from pathlib import Path

        from exocortex.ingestion import ingest_pdf

        path = Path(pdf_path)
        chunks = ingest_pdf(
            pdf_path=path,
            chunk_size=self.settings.chunk_size,
            chunk_overlap=self.settings.chunk_overlap,
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

        logger.info(f"Ingested {path.name}: {len(chunks)} chunks embedded and stored")

        return {
            "filename": path.name,
            "document_id": chunks[0].document_id,
            "chunk_count": len(chunks),
        }
