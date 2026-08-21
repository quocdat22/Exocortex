"""Tests for Chunking Factory and Ingestion Pipeline Integration."""

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from exocortex.chunking import (
    BaseChunker,
    FixedSizeChunker,
    RecursiveCharacterChunker,
    SemanticChunker,
    SentenceParagraphChunker,
    get_chunker,
)
from exocortex.config import Settings
from exocortex.ingestion import PageContent, ingest_pdf

# --- Factory Strategy Tests ---


def test_get_chunker_strategies():
    """get_chunker should return correct Chunker instance for all supported strategy aliases."""
    # Fixed size
    fixed1 = get_chunker("fixed")
    assert isinstance(fixed1, FixedSizeChunker)
    assert isinstance(fixed1, BaseChunker)
    assert fixed1.chunk_size == 512
    assert fixed1.chunk_overlap == 50

    fixed2 = get_chunker("fixed_size")
    assert isinstance(fixed2, FixedSizeChunker)

    # Recursive
    rec1 = get_chunker("recursive")
    assert isinstance(rec1, RecursiveCharacterChunker)
    assert isinstance(rec1, BaseChunker)
    assert rec1.chunk_size == 512
    assert rec1.chunk_overlap == 50

    rec2 = get_chunker("recursive_character")
    assert isinstance(rec2, RecursiveCharacterChunker)

    # Sentence / Paragraph
    sent1 = get_chunker("sentence_paragraph")
    assert isinstance(sent1, SentenceParagraphChunker)
    assert isinstance(sent1, BaseChunker)
    assert sent1.chunk_size == 512
    assert sent1.sentence_overlap == 1

    sent2 = get_chunker("sentence")
    assert isinstance(sent2, SentenceParagraphChunker)

    sent3 = get_chunker("paragraph")
    assert isinstance(sent3, SentenceParagraphChunker)

    # Semantic
    sem = get_chunker("semantic")
    assert isinstance(sem, SemanticChunker)
    assert isinstance(sem, BaseChunker)
    assert sem.distance_threshold_percentile == 85.0
    assert sem.min_chunk_size == 100
    assert sem.max_chunk_size == 1000


def test_get_chunker_case_insensitive_and_whitespace():
    """Strategy name matching should be case-insensitive and tolerate whitespace."""
    assert isinstance(get_chunker("  FIXED  "), FixedSizeChunker)
    assert isinstance(get_chunker("Recursive_Character\n"), RecursiveCharacterChunker)
    assert isinstance(get_chunker("  Sentence_Paragraph  "), SentenceParagraphChunker)
    assert isinstance(get_chunker(" SEMANTIC "), SemanticChunker)


def test_get_chunker_custom_kwargs():
    """get_chunker should pass custom kwargs to chunker constructors."""
    fixed = get_chunker("fixed", chunk_size=256, chunk_overlap=30)
    assert fixed.chunk_size == 256
    assert fixed.chunk_overlap == 30

    rec = get_chunker(
        "recursive", chunk_size=300, chunk_overlap=40, separators=["\n\n", " "]
    )
    assert rec.chunk_size == 300
    assert rec.chunk_overlap == 40
    assert rec.separators == ["\n\n", " "]

    sent = get_chunker("sentence_paragraph", chunk_size=400, sentence_overlap=3)
    assert sent.chunk_size == 400
    assert sent.sentence_overlap == 3

    mock_client = MagicMock()
    sem = get_chunker(
        "semantic",
        embedding_client=mock_client,
        distance_threshold_percentile=90.0,
        min_chunk_size=80,
        max_chunk_size=800,
    )
    assert sem.embedding_client is mock_client
    assert sem.distance_threshold_percentile == 90.0
    assert sem.min_chunk_size == 80
    assert sem.max_chunk_size == 800


def test_get_chunker_invalid():
    """Unknown strategy name should raise ValueError with informative message."""
    with pytest.raises(ValueError, match="Unknown chunking strategy"):
        get_chunker("unknown_strategy")

    with pytest.raises(ValueError, match="Unknown chunking strategy"):
        get_chunker("")


def test_chunking_module_exports():
    """exocortex.chunking should expose all public classes and get_chunker in __all__."""
    import exocortex.chunking as pkg

    expected_exports = {
        "BaseChunker",
        "FixedSizeChunker",
        "RecursiveCharacterChunker",
        "SemanticChunker",
        "SentenceParagraphChunker",
        "get_chunker",
    }
    assert set(pkg.__all__) == expected_exports
    for name in expected_exports:
        assert hasattr(pkg, name)


# --- Config Integration Tests ---


def test_config_chunking_strategy_default():
    """Settings should include chunking_strategy with default 'recursive'."""
    settings = Settings(deepseek_api_key="test-key")
    assert hasattr(settings, "chunking_strategy")
    assert settings.chunking_strategy == "recursive"


def test_config_chunking_strategy_override(monkeypatch):
    """Settings chunking_strategy should be overridable via environment variable."""
    monkeypatch.setenv("CHUNKING_STRATEGY", "fixed")
    settings = Settings(deepseek_api_key="test-key")
    assert settings.chunking_strategy == "fixed"


# --- Ingestion Pipeline Integration Tests ---


def test_ingest_pdf_with_strategy():
    """ingest_pdf should accept strategy parameter and delegate to get_chunker."""
    dummy_pages = [
        PageContent(page_number=1, text="First paragraph of text.\n\nSecond paragraph of text."),
        PageContent(page_number=2, text="Third paragraph of text on page 2."),
    ]

    with patch("exocortex.ingestion.extract_text_from_pdf", return_value=dummy_pages):
        # Test default fixed strategy
        chunks_fixed = ingest_pdf(Path("dummy.pdf"), chunk_size=100, chunk_overlap=20)
        assert len(chunks_fixed) > 0
        assert all(c.filename == "dummy.pdf" for c in chunks_fixed)

        # Test recursive strategy
        chunks_rec = ingest_pdf(
            Path("dummy.pdf"),
            chunk_size=100,
            chunk_overlap=20,
            strategy="recursive",
        )
        assert len(chunks_rec) > 0
        assert all(c.filename == "dummy.pdf" for c in chunks_rec)

        # Test sentence_paragraph strategy
        chunks_sent = ingest_pdf(
            Path("dummy.pdf"),
            chunk_size=100,
            strategy="sentence_paragraph",
            sentence_overlap=1,
        )
        assert len(chunks_sent) > 0
        assert all(c.filename == "dummy.pdf" for c in chunks_sent)


def test_ingest_pdf_invalid_strategy():
    """ingest_pdf should raise ValueError when an invalid strategy is specified."""
    dummy_pages = [PageContent(page_number=1, text="Sample text.")]
    with (
        patch("exocortex.ingestion.extract_text_from_pdf", return_value=dummy_pages),
        pytest.raises(ValueError, match="Unknown chunking strategy"),
    ):
        ingest_pdf(Path("dummy.pdf"), strategy="invalid_strategy")
