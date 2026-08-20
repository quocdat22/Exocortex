"""ChromaDB vector store operations.

Manages a ChromaDB collection for storing and querying document chunk embeddings.
Runs in embedded mode (no separate server required).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import chromadb

from exocortex.config import Settings
from exocortex.ingestion import Chunk

logger = logging.getLogger(__name__)


@dataclass
class SearchResult:
    """A single search result from the vector store."""

    text: str  # The chunk text content
    metadata: dict  # Chunk metadata (filename, page_numbers, etc.)
    distance: float  # Distance score (lower = more similar)
    chunk_id: str  # Unique chunk ID in ChromaDB


class VectorStore:
    """ChromaDB vector store for document chunks.

    Provides methods to add, query, list, and delete document chunks.
    Uses ChromaDB in embedded persistent mode.

    Args:
        settings: Application settings with ChromaDB configuration.
    """

    def __init__(self, settings: Settings) -> None:
        self.persist_dir = settings.chroma_path
        self.collection_name = settings.chroma_collection_name

        # Ensure persist directory exists
        self.persist_dir.mkdir(parents=True, exist_ok=True)

        # Initialize ChromaDB client (embedded, persistent)
        self.client = chromadb.PersistentClient(
            path=str(self.persist_dir),
        )

        # Get or create the collection
        # We manage embeddings ourselves (from Ollama), so no embedding function
        self.collection = self.client.get_or_create_collection(
            name=self.collection_name,
            metadata={"hnsw:space": "cosine"},  # cosine similarity
        )

        logger.info(
            f"ChromaDB collection '{self.collection_name}' initialized "
            f"with {self.collection.count()} existing documents"
        )

    def add_chunks(
        self,
        chunks: list[Chunk],
        embeddings: list[list[float]],
    ) -> int:
        """Add chunks with their embeddings to the vector store.

        Each chunk gets a unique ID: {document_id}_{chunk_index}.
        If a chunk ID already exists, it will be updated (upserted).

        Args:
            chunks: List of Chunk objects from the ingestion pipeline.
            embeddings: Corresponding embedding vectors (same order as chunks).

        Returns:
            Number of chunks added/updated.

        Raises:
            ValueError: If chunks and embeddings have different lengths.
        """
        if len(chunks) != len(embeddings):
            raise ValueError(
                f"Chunks ({len(chunks)}) and embeddings ({len(embeddings)}) "
                f"must have the same length"
            )

        if not chunks:
            return 0

        ids = [f"{c.document_id}_{c.chunk_index}" for c in chunks]
        documents = [c.text for c in chunks]
        metadatas = [c.to_metadata_dict() for c in chunks]

        self.collection.upsert(
            ids=ids,
            documents=documents,
            embeddings=embeddings,
            metadatas=metadatas,
        )

        logger.info(f"Added/updated {len(chunks)} chunks to ChromaDB")
        return len(chunks)

    def query(
        self,
        query_embedding: list[float],
        top_k: int = 5,
    ) -> list[SearchResult]:
        """Query the vector store for the most similar chunks.

        Args:
            query_embedding: The query embedding vector.
            top_k: Number of results to return.

        Returns:
            List of SearchResult objects, ordered by similarity (most similar first).
        """
        count = self.collection.count()
        if count == 0:
            return []

        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=min(top_k, count),
            include=["documents", "metadatas", "distances"],
        )

        search_results: list[SearchResult] = []

        if not results["ids"][0]:
            return search_results

        for i, chunk_id in enumerate(results["ids"][0]):
            search_results.append(
                SearchResult(
                    text=results["documents"][0][i],
                    metadata=results["metadatas"][0][i],
                    distance=results["distances"][0][i],
                    chunk_id=chunk_id,
                )
            )

        return search_results

    def list_documents(self) -> list[dict]:
        """List all unique documents in the vector store.

        Returns:
            List of dicts with document info: {document_id, filename, chunk_count}.
        """
        all_metadata = self.collection.get(include=["metadatas"])

        # Group by document_id
        doc_map: dict[str, dict] = {}
        for meta in all_metadata["metadatas"]:
            doc_id = meta.get("document_id", "unknown")
            if doc_id not in doc_map:
                doc_map[doc_id] = {
                    "document_id": doc_id,
                    "filename": meta.get("filename", "unknown"),
                    "chunk_count": 0,
                }
            doc_map[doc_id]["chunk_count"] += 1

        return list(doc_map.values())

    def delete_document(self, document_id: str) -> int:
        """Delete all chunks belonging to a document.

        Args:
            document_id: The document ID to delete.

        Returns:
            Number of chunks deleted.
        """
        # Find all chunk IDs for this document
        results = self.collection.get(
            where={"document_id": document_id},
            include=[],
        )

        if not results["ids"]:
            return 0

        count = len(results["ids"])
        self.collection.delete(ids=results["ids"])
        logger.info(f"Deleted {count} chunks for document {document_id}")
        return count

    def count(self) -> int:
        """Return total number of chunks in the collection."""
        return self.collection.count()

    def health_check(self) -> bool:
        """Check if ChromaDB is operational.

        Returns:
            True if the collection is accessible, False otherwise.
        """
        try:
            self.collection.count()
            return True
        except Exception:
            return False
