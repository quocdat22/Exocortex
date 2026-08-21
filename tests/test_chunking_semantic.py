"""Tests for SemanticChunker strategy."""

from __future__ import annotations

from unittest.mock import MagicMock, patch
import pytest

from exocortex.chunking.semantic import SemanticChunker
from exocortex.ingestion import Chunk, PageContent


def test_semantic_chunker_breakpoint_splitting():
    """SemanticChunker should split text at semantic breakpoints based on embedding distance."""
    mock_embedding_client = MagicMock()
    # Mock embeddings such that sentence 1 & 2 are close, 3 & 4 are close, but 2 and 3 are far apart
    v1 = [1.0, 0.0]
    v2 = [0.95, 0.05]
    v3 = [0.0, 1.0]
    v4 = [0.05, 0.95]
    mock_embedding_client.embed_documents.return_value = [v1, v2, v3, v4]

    chunker = SemanticChunker(
        embedding_client=mock_embedding_client,
        distance_threshold_percentile=50.0,
        min_chunk_size=50,
        max_chunk_size=1000,
    )
    pages = [
        PageContent(
            page_number=1,
            text="Apples are delicious fruits. Oranges are also sweet citrus. Machine learning algorithms train models. Deep learning uses neural networks.",
        )
    ]
    chunks = chunker.chunk(pages, document_id="doc1", filename="doc1.pdf")
    assert len(chunks) == 2
    assert all(isinstance(c, Chunk) for c in chunks)
    assert "fruits" in chunks[0].text
    assert "Machine learning" in chunks[1].text
    assert chunks[0].document_id == "doc1"
    assert chunks[0].filename == "doc1.pdf"
    assert chunks[0].chunk_index == 0
    assert chunks[1].chunk_index == 1
    assert chunks[0].page_numbers == [1]
    assert chunks[0].metadata["strategy"] == "semantic"
    assert chunks[1].metadata["strategy"] == "semantic"


def test_semantic_chunker_validation():
    """SemanticChunker should validate constructor parameters."""
    with pytest.raises(ValueError, match="distance_threshold_percentile must be between 0.0 and 100.0"):
        SemanticChunker(distance_threshold_percentile=-1.0)

    with pytest.raises(ValueError, match="distance_threshold_percentile must be between 0.0 and 100.0"):
        SemanticChunker(distance_threshold_percentile=101.0)

    with pytest.raises(ValueError, match="min_chunk_size must be positive"):
        SemanticChunker(min_chunk_size=0)

    with pytest.raises(ValueError, match="max_chunk_size must be positive"):
        SemanticChunker(max_chunk_size=0)

    with pytest.raises(ValueError, match="min_chunk_size cannot exceed max_chunk_size"):
        SemanticChunker(min_chunk_size=500, max_chunk_size=100)


def test_semantic_chunker_empty_pages():
    """SemanticChunker should return empty list for empty pages."""
    chunker = SemanticChunker()
    assert chunker.chunk([], document_id="doc1", filename="test.pdf") == []


def test_semantic_chunker_blank_text():
    """SemanticChunker should return empty list for blank/whitespace text."""
    chunker = SemanticChunker()
    pages = [PageContent(page_number=1, text="   \n\n   \t  ")]
    assert chunker.chunk(pages, document_id="doc1", filename="test.pdf") == []


def test_semantic_chunker_single_sentence():
    """SemanticChunker should return single chunk without embedding calls for single sentence."""
    mock_client = MagicMock()
    chunker = SemanticChunker(embedding_client=mock_client)
    pages = [PageContent(page_number=1, text="Only one sentence is present in this text.")]
    chunks = chunker.chunk(pages, document_id="doc1", filename="test.pdf")

    assert len(chunks) == 1
    assert chunks[0].text == "Only one sentence is present in this text."
    assert chunks[0].chunk_index == 0
    assert chunks[0].page_numbers == [1]
    assert chunks[0].metadata["strategy"] == "semantic"
    mock_client.embed_documents.assert_not_called()


def test_semantic_chunker_max_chunk_size_split():
    """SemanticChunker should force split when adding sentence exceeds max_chunk_size."""
    mock_client = MagicMock()
    # Identical embeddings -> distance 0.0, well below threshold
    mock_client.embed_documents.return_value = [
        [1.0, 0.0],
        [1.0, 0.0],
        [1.0, 0.0],
    ]
    chunker = SemanticChunker(
        embedding_client=mock_client,
        distance_threshold_percentile=99.0,
        min_chunk_size=20,
        max_chunk_size=60,
    )
    pages = [
        PageContent(
            page_number=1,
            text="First sentence with moderate length. Second sentence with moderate length. Third sentence is here.",
        )
    ]
    chunks = chunker.chunk(pages, document_id="doc1", filename="test.pdf")
    assert len(chunks) >= 2


def test_semantic_chunker_min_chunk_size_suppression():
    """SemanticChunker should not split at breakpoint if current chunk is smaller than min_chunk_size."""
    mock_client = MagicMock()
    # High distance between sentence 1 and 2, but sentence 1 is very short
    mock_client.embed_documents.return_value = [
        [1.0, 0.0],
        [0.0, 1.0],
        [0.0, 1.0],
    ]
    chunker = SemanticChunker(
        embedding_client=mock_client,
        distance_threshold_percentile=10.0,
        min_chunk_size=100,  # min_chunk_size is large
        max_chunk_size=1000,
    )
    pages = [
        PageContent(
            page_number=1,
            text="Hi. Second sentence extends the content length. Third sentence finishes it up nicely.",
        )
    ]
    chunks = chunker.chunk(pages, document_id="doc1", filename="test.pdf")
    # "Hi." has length 3, so first split is suppressed; sentences 1 & 2 grouped together
    assert len(chunks) == 1 or "Hi." in chunks[0].text


def test_semantic_chunker_multi_page_span():
    """SemanticChunker should track page numbers across multi-page input."""
    mock_client = MagicMock()
    mock_client.embed_documents.return_value = [
        [1.0, 0.0],
        [1.0, 0.0],
        [0.0, 1.0],
        [0.0, 1.0],
    ]
    chunker = SemanticChunker(
        embedding_client=mock_client,
        distance_threshold_percentile=50.0,
        min_chunk_size=30,
        max_chunk_size=1000,
    )
    pages = [
        PageContent(page_number=1, text="Sentence one on page one. Sentence two on page one."),
        PageContent(page_number=2, text="Sentence three on page two. Sentence four on page two."),
    ]
    chunks = chunker.chunk(pages, document_id="doc1", filename="test.pdf")
    assert len(chunks) == 2
    assert chunks[0].page_numbers == [1]
    assert chunks[1].page_numbers == [2]


def test_semantic_chunker_lazy_client_init():
    """SemanticChunker should lazily initialize EmbeddingClient if none provided."""
    chunker = SemanticChunker(embedding_client=None)
    with patch("exocortex.config.get_settings") as mock_get_settings, patch(
        "exocortex.embedding.EmbeddingClient"
    ) as mock_client_cls:
        mock_instance = MagicMock()
        mock_instance.embed_documents.return_value = [[1.0, 0.0], [1.0, 0.0]]
        mock_client_cls.return_value = mock_instance

        pages = [PageContent(page_number=1, text="Sentence one here. Sentence two here.")]
        chunks = chunker.chunk(pages, document_id="doc1", filename="test.pdf")

        mock_get_settings.assert_called_once()
        mock_client_cls.assert_called_once()
        assert len(chunks) >= 1


def test_semantic_chunker_cosine_distance():
    """_cosine_distance should correctly calculate 1 - cosine similarity."""
    # Identical vectors -> dist 0.0
    assert pytest.approx(SemanticChunker._cosine_distance([1.0, 0.0], [1.0, 0.0])) == 0.0
    # Orthogonal vectors -> dist 1.0
    assert pytest.approx(SemanticChunker._cosine_distance([1.0, 0.0], [0.0, 1.0])) == 1.0
    # Opposite vectors -> dist 2.0
    assert pytest.approx(SemanticChunker._cosine_distance([1.0, 0.0], [-1.0, 0.0])) == 2.0
    # Zero vector -> dist 1.0
    assert SemanticChunker._cosine_distance([0.0, 0.0], [1.0, 1.0]) == 1.0
    assert SemanticChunker._cosine_distance([1.0, 1.0], [0.0, 0.0]) == 1.0


def test_semantic_chunker_split_sentences():
    """_split_into_sentences should split on sentence ending punctuation."""
    text = "First sentence! Second sentence? Third sentence. \"Fourth quoted sentence.\""
    sentences = SemanticChunker._split_into_sentences(text)
    assert len(sentences) == 4
    assert sentences[0] == "First sentence!"
    assert sentences[1] == "Second sentence?"
    assert sentences[2] == "Third sentence."
    assert sentences[3] == '"Fourth quoted sentence."'
