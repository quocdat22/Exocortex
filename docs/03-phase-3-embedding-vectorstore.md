# Phase 3: Embedding + Vector Store

## Objective

Build the embedding client (Ollama) and vector store operations (ChromaDB) so that
chunks from Phase 2 can be embedded into vectors and stored/queried in ChromaDB.

## Prerequisites

- Phase 2 completed (ingestion pipeline producing `Chunk` objects)
- Ollama running locally with `qwen3-embedding:0.6b` pulled
- `uv sync` completed

---

## Concepts

### Embedding with qwen3-embedding:0.6b

Key technical details:
- **Model:** `qwen3-embedding:0.6b` running in Ollama
- **API:** `POST {ollama_base_url}/api/embed`
- **Dimension:** 1024 (output vector length)
- **Context window:** 32,768 tokens (set via `num_ctx` option)
- **Instruction-aware:** Queries benefit from instruction prefixes; documents are embedded as-is

**Important — Instruction Prefix for Queries:**

The qwen3-embedding model is instruction-aware. For better retrieval accuracy:
- **Documents (chunks):** Embed directly, no prefix needed
- **Queries:** Prepend an instruction prefix:
  ```
  Instruct: Given a web search query, retrieve relevant passages that answer the query
  Query: {user_query}
  ```

### ChromaDB Embedded Mode

ChromaDB runs in-process (no separate server needed):
- Data persisted to `chroma_data/` directory
- Collection named `exocortex_ebooks`
- Stores: document text, embedding vector, metadata
- Each chunk gets a unique ID: `{document_id}_{chunk_index}`

---

## Implementation

### File: `src/exocortex/embedding.py`

```python
"""Embedding client using Ollama API.

Connects to a local Ollama instance to generate embeddings using the
qwen3-embedding:0.6b model. Supports both single and batch embedding.
"""

from __future__ import annotations

import logging

import httpx

from exocortex.config import Settings

logger = logging.getLogger(__name__)

# Instruction prefix for query embedding (improves retrieval accuracy)
QUERY_INSTRUCTION = (
    "Instruct: Given a web search query, retrieve relevant passages "
    "that answer the query\nQuery: "
)


class EmbeddingClient:
    """Client for generating embeddings via Ollama API.

    Args:
        settings: Application settings with Ollama configuration.
    """

    def __init__(self, settings: Settings) -> None:
        self.base_url = settings.ollama_base_url
        self.model = settings.embedding_model
        self.num_ctx = settings.embedding_num_ctx
        self.expected_dim = settings.embedding_dim

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """Embed a list of document texts (no instruction prefix).

        Args:
            texts: List of document/chunk texts to embed.

        Returns:
            List of embedding vectors (each is a list of floats).

        Raises:
            ConnectionError: If Ollama is not reachable.
            RuntimeError: If the embedding request fails.
        """
        if not texts:
            return []

        return self._call_embed_api(texts)

    def embed_query(self, query: str) -> list[float]:
        """Embed a single query text (with instruction prefix).

        Prepends the instruction prefix for better retrieval accuracy
        with qwen3-embedding's instruction-aware architecture.

        Args:
            query: The user's query text.

        Returns:
            A single embedding vector (list of floats).

        Raises:
            ConnectionError: If Ollama is not reachable.
            RuntimeError: If the embedding request fails.
        """
        prefixed_query = f"{QUERY_INSTRUCTION}{query}"
        embeddings = self._call_embed_api([prefixed_query])
        return embeddings[0]

    def _call_embed_api(self, texts: list[str]) -> list[list[float]]:
        """Call the Ollama /api/embed endpoint.

        Args:
            texts: List of texts to embed.

        Returns:
            List of embedding vectors.

        Raises:
            ConnectionError: If Ollama server is unreachable.
            RuntimeError: If the API returns an error.
        """
        url = f"{self.base_url}/api/embed"
        payload = {
            "model": self.model,
            "input": texts,
            "options": {
                "num_ctx": self.num_ctx,
            },
        }

        try:
            with httpx.Client(timeout=120.0) as client:
                response = client.post(url, json=payload)
        except httpx.ConnectError as e:
            raise ConnectionError(
                f"Cannot connect to Ollama at {self.base_url}. "
                f"Is Ollama running? Error: {e}"
            ) from e

        if response.status_code != 200:
            raise RuntimeError(
                f"Ollama embedding failed (HTTP {response.status_code}): "
                f"{response.text}"
            )

        data = response.json()
        embeddings = data.get("embeddings", [])

        if len(embeddings) != len(texts):
            raise RuntimeError(
                f"Expected {len(texts)} embeddings, got {len(embeddings)}"
            )

        # Validate dimension
        if embeddings and len(embeddings[0]) != self.expected_dim:
            logger.warning(
                f"Embedding dimension mismatch: expected {self.expected_dim}, "
                f"got {len(embeddings[0])}"
            )

        return embeddings

    def health_check(self) -> bool:
        """Check if Ollama is reachable and the embedding model is available.

        Returns:
            True if Ollama is healthy and model is available, False otherwise.
        """
        try:
            with httpx.Client(timeout=10.0) as client:
                # Check Ollama is running
                response = client.get(f"{self.base_url}/api/tags")
                if response.status_code != 200:
                    return False

                # Check model is available
                models = response.json().get("models", [])
                model_names = [m.get("name", "") for m in models]
                # Match with or without tag
                model_base = self.model.split(":")[0]
                return any(model_base in name for name in model_names)

        except (httpx.ConnectError, httpx.TimeoutException):
            return False
```

### File: `src/exocortex/vectorstore.py`

```python
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
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=min(top_k, self.collection.count()),
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
```

---

## Tests

### File: `tests/test_embedding.py`

```python
"""Tests for Phase 3: Embedding client."""

import logging

import pytest

from exocortex.config import Settings
from exocortex.embedding import EmbeddingClient, QUERY_INSTRUCTION

logger = logging.getLogger(__name__)


@pytest.fixture
def settings() -> Settings:
    return Settings(deepseek_api_key="test-key")


@pytest.fixture
def client(settings) -> EmbeddingClient:
    return EmbeddingClient(settings)


def test_embed_client_initialization(client):
    """Client should initialize with correct settings."""
    assert client.model == "qwen3-embedding:0.6b"
    assert client.expected_dim == 1024


def test_embed_empty_list(client):
    """Embedding empty list should return empty list."""
    result = client.embed_documents([])
    assert result == []


def test_query_instruction_prefix():
    """Query instruction prefix should be defined."""
    assert "Instruct:" in QUERY_INSTRUCTION
    assert "Query:" in QUERY_INSTRUCTION


# --- Integration tests (require Ollama running) ---


@pytest.fixture
def ollama_available(client) -> bool:
    """Check if Ollama is available for integration tests."""
    if not client.health_check():
        pytest.skip("Ollama not available or model not pulled")
    return True


def test_embed_documents_integration(client, ollama_available):
    """Should produce embeddings with correct dimension."""
    texts = ["Hello world", "This is a test document about machine learning."]
    embeddings = client.embed_documents(texts)

    assert len(embeddings) == 2
    assert len(embeddings[0]) == 1024
    assert len(embeddings[1]) == 1024
    assert all(isinstance(v, float) for v in embeddings[0])
    logger.info(f"Embedded {len(texts)} texts, dim={len(embeddings[0])}")


def test_embed_query_integration(client, ollama_available):
    """Query embedding should have instruction prefix applied."""
    embedding = client.embed_query("What is machine learning?")

    assert len(embedding) == 1024
    assert all(isinstance(v, float) for v in embedding)
    logger.info(f"Query embedding dim={len(embedding)}")


def test_embed_similarity_integration(client, ollama_available):
    """Similar texts should have closer embeddings than dissimilar texts."""
    texts = [
        "Machine learning is a subset of artificial intelligence.",
        "Deep learning uses neural networks with many layers.",
        "The weather today is sunny and warm.",
    ]
    embeddings = client.embed_documents(texts)

    # Compute cosine similarity (dot product since embeddings are L2-normalized)
    def cosine_sim(a, b):
        return sum(x * y for x, y in zip(a, b))

    sim_related = cosine_sim(embeddings[0], embeddings[1])  # ML vs DL
    sim_unrelated = cosine_sim(embeddings[0], embeddings[2])  # ML vs weather

    assert sim_related > sim_unrelated, (
        f"Related texts should be more similar: "
        f"ML-DL={sim_related:.4f} vs ML-weather={sim_unrelated:.4f}"
    )
    logger.info(f"Similarity: ML-DL={sim_related:.4f}, ML-weather={sim_unrelated:.4f}")


def test_health_check_integration(client):
    """Health check should return a boolean."""
    result = client.health_check()
    assert isinstance(result, bool)
    logger.info(f"Ollama health check: {'OK' if result else 'FAILED'}")
```

### File: `tests/test_vectorstore.py`

```python
"""Tests for Phase 3: ChromaDB vector store."""

import logging
import tempfile

import pytest

from exocortex.config import Settings
from exocortex.ingestion import Chunk
from exocortex.vectorstore import VectorStore, SearchResult

logger = logging.getLogger(__name__)


@pytest.fixture
def temp_chroma_dir():
    """Create a temporary directory for ChromaDB in tests."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield tmpdir


@pytest.fixture
def settings(temp_chroma_dir) -> Settings:
    return Settings(
        deepseek_api_key="test-key",
        chroma_persist_dir=temp_chroma_dir,
        chroma_collection_name="test_collection",
    )


@pytest.fixture
def store(settings) -> VectorStore:
    return VectorStore(settings)


@pytest.fixture
def sample_chunks() -> list[Chunk]:
    """Create sample chunks for testing."""
    return [
        Chunk(
            text="Machine learning is a branch of AI.",
            document_id="doc1",
            filename="ml_book.pdf",
            page_numbers=[1],
            chunk_index=0,
        ),
        Chunk(
            text="Neural networks process data in layers.",
            document_id="doc1",
            filename="ml_book.pdf",
            page_numbers=[1, 2],
            chunk_index=1,
        ),
        Chunk(
            text="Python is a popular programming language.",
            document_id="doc2",
            filename="python_book.pdf",
            page_numbers=[5],
            chunk_index=0,
        ),
    ]


@pytest.fixture
def sample_embeddings() -> list[list[float]]:
    """Create fake embeddings (1024-dim) for testing."""
    import random

    random.seed(42)
    return [[random.random() for _ in range(1024)] for _ in range(3)]


def test_store_initialization(store):
    """Store should initialize with empty collection."""
    assert store.count() == 0
    assert store.health_check() is True


def test_add_chunks(store, sample_chunks, sample_embeddings):
    """Should add chunks and increase count."""
    added = store.add_chunks(sample_chunks, sample_embeddings)
    assert added == 3
    assert store.count() == 3
    logger.info(f"Added {added} chunks, total: {store.count()}")


def test_add_chunks_mismatch(store, sample_chunks, sample_embeddings):
    """Should raise ValueError if chunks and embeddings have different lengths."""
    with pytest.raises(ValueError, match="same length"):
        store.add_chunks(sample_chunks, sample_embeddings[:2])


def test_add_chunks_upsert(store, sample_chunks, sample_embeddings):
    """Adding same chunks twice should upsert (not duplicate)."""
    store.add_chunks(sample_chunks, sample_embeddings)
    store.add_chunks(sample_chunks, sample_embeddings)
    assert store.count() == 3  # Still 3, not 6


def test_query(store, sample_chunks, sample_embeddings):
    """Should return results ordered by similarity."""
    store.add_chunks(sample_chunks, sample_embeddings)

    # Query with the first chunk's embedding (should match itself best)
    results = store.query(sample_embeddings[0], top_k=2)

    assert len(results) == 2
    assert all(isinstance(r, SearchResult) for r in results)
    assert results[0].chunk_id == "doc1_0"  # Most similar to itself
    logger.info(f"Query returned {len(results)} results")


def test_query_empty_store(store, sample_embeddings):
    """Querying empty store should return empty list."""
    results = store.query(sample_embeddings[0], top_k=5)
    assert results == []


def test_list_documents(store, sample_chunks, sample_embeddings):
    """Should list unique documents with chunk counts."""
    store.add_chunks(sample_chunks, sample_embeddings)

    docs = store.list_documents()
    assert len(docs) == 2  # doc1 and doc2

    doc_map = {d["document_id"]: d for d in docs}
    assert doc_map["doc1"]["filename"] == "ml_book.pdf"
    assert doc_map["doc1"]["chunk_count"] == 2
    assert doc_map["doc2"]["filename"] == "python_book.pdf"
    assert doc_map["doc2"]["chunk_count"] == 1
    logger.info(f"Documents: {docs}")


def test_delete_document(store, sample_chunks, sample_embeddings):
    """Should delete all chunks for a document."""
    store.add_chunks(sample_chunks, sample_embeddings)
    assert store.count() == 3

    deleted = store.delete_document("doc1")
    assert deleted == 2
    assert store.count() == 1  # Only doc2 remains
    logger.info(f"Deleted {deleted} chunks, remaining: {store.count()}")


def test_delete_nonexistent_document(store):
    """Deleting non-existent document should return 0."""
    deleted = store.delete_document("nonexistent")
    assert deleted == 0
```

---

## Success Criteria

Phase 3 is complete when ALL of the following are true:

### Automated Tests
```bash
# Vector store tests (no Ollama required — uses fake embeddings)
uv run pytest tests/test_vectorstore.py -v --log-cli-level=INFO

# Embedding client tests (Ollama required for integration tests)
uv run pytest tests/test_embedding.py -v --log-cli-level=INFO
```

**Expected:**
- `test_vectorstore.py`: All tests PASSED
- `test_embedding.py`: Unit tests PASSED; integration tests PASSED if Ollama running, SKIPPED otherwise

### Manual Verification
```bash
# Test embedding with Ollama
uv run python -c "
from exocortex.config import get_settings
from exocortex.embedding import EmbeddingClient

client = EmbeddingClient(get_settings())
print(f'Ollama health: {client.health_check()}')

emb = client.embed_documents(['Hello world'])
print(f'Embedding dim: {len(emb[0])}')
print(f'First 5 values: {emb[0][:5]}')
"
```

**Expected output:**
```
Ollama health: True
Embedding dim: 1024
First 5 values: [0.123..., -0.456..., ...]
```

### Checklist
- [X] `src/exocortex/embedding.py` exists with `EmbeddingClient` class
- [X] `src/exocortex/vectorstore.py` exists with `VectorStore` class
- [X] `EmbeddingClient.embed_documents()` returns 1024-dim vectors
- [X] `EmbeddingClient.embed_query()` prepends instruction prefix
- [X] `EmbeddingClient.health_check()` verifies Ollama + model availability
- [X] `VectorStore` uses ChromaDB persistent mode
- [X] `VectorStore.add_chunks()` stores chunks with embeddings and metadata
- [X] `VectorStore.query()` returns `SearchResult` objects sorted by similarity
- [X] `VectorStore.list_documents()` returns unique documents with chunk counts
- [X] `VectorStore.delete_document()` removes all chunks for a document
- [X] All tests pass
