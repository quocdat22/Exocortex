"""Tests for Phase 3: ChromaDB vector store."""

import logging
import tempfile

import pytest

from exocortex.config import Settings
from exocortex.ingestion import Chunk
from exocortex.vectorstore import SearchResult, VectorStore

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
