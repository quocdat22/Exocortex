"""Tests for BaseChunker and FixedSizeChunker strategy."""

from __future__ import annotations

import pytest

from exocortex.chunking.base import BaseChunker
from exocortex.chunking.fixed_size import FixedSizeChunker
from exocortex.ingestion import Chunk, PageContent


def test_base_chunker_is_abstract():
    """BaseChunker should not be directly instantiable without chunk()."""
    with pytest.raises(TypeError):
        BaseChunker()  # type: ignore[abstract]


def test_base_chunker_build_page_segments():
    """build_page_segments should return joined text and offset segments."""
    pages = [
        PageContent(page_number=1, text="Hello world"),
        PageContent(page_number=2, text="Second page text"),
    ]
    full_text, segments = BaseChunker.build_page_segments(pages)

    assert full_text == "Hello world Second page text"
    assert len(segments) == 2
    assert segments[0] == (0, 1, "Hello world")
    assert segments[1] == (12, 2, "Second page text")


def test_base_chunker_find_pages_for_range():
    """find_pages_for_range should correctly identify overlapping pages."""
    segments = [
        (0, 1, "Page one text here."),      # len 19 -> [0, 19)
        (20, 2, "Page two content text."),   # len 22 -> [20, 42)
    ]
    # Spans only page 1
    assert BaseChunker.find_pages_for_range(segments, 0, 10) == [1]
    # Spans boundary between page 1 and page 2
    assert BaseChunker.find_pages_for_range(segments, 15, 25) == [1, 2]
    # Spans only page 2
    assert BaseChunker.find_pages_for_range(segments, 22, 30) == [2]


def test_fixed_size_chunker_basic():
    """Basic chunking with FixedSizeChunker."""
    chunker = FixedSizeChunker(chunk_size=100, chunk_overlap=20)
    pages = [
        PageContent(page_number=1, text="This is page 1 content with some text to be split."),
        PageContent(page_number=2, text="This is page 2 content which continues the discussion."),
    ]
    chunks = chunker.chunk(pages, document_id="doc123", filename="test.pdf")

    assert len(chunks) > 0
    assert all(isinstance(c, Chunk) for c in chunks)
    assert chunks[0].document_id == "doc123"
    assert chunks[0].filename == "test.pdf"
    assert chunks[0].chunk_index == 0
    assert 1 in chunks[0].page_numbers
    assert chunks[0].metadata.get("strategy") == "fixed_size"


def test_fixed_size_chunker_validation():
    """FixedSizeChunker should reject invalid parameter combinations."""
    with pytest.raises(ValueError, match="chunk_size must be positive"):
        FixedSizeChunker(chunk_size=0, chunk_overlap=0)

    with pytest.raises(ValueError, match="chunk_size must be positive"):
        FixedSizeChunker(chunk_size=-10, chunk_overlap=0)

    with pytest.raises(ValueError, match="chunk_overlap must be non-negative"):
        FixedSizeChunker(chunk_size=50, chunk_overlap=-5)

    with pytest.raises(ValueError, match="chunk_overlap must be less than chunk_size"):
        FixedSizeChunker(chunk_size=50, chunk_overlap=50)

    with pytest.raises(ValueError, match="chunk_overlap must be less than chunk_size"):
        FixedSizeChunker(chunk_size=50, chunk_overlap=60)


def test_fixed_size_chunker_empty_pages():
    """FixedSizeChunker should return empty list for empty pages."""
    chunker = FixedSizeChunker(chunk_size=100, chunk_overlap=20)
    assert chunker.chunk([], document_id="doc1", filename="test.pdf") == []


def test_fixed_size_chunker_multi_page_span():
    """FixedSizeChunker should track all pages spanned by a chunk."""
    chunker = FixedSizeChunker(chunk_size=150, chunk_overlap=20)
    pages = [
        PageContent(page_number=1, text="a" * 100),
        PageContent(page_number=2, text="b" * 100),
    ]
    chunks = chunker.chunk(pages, document_id="doc1", filename="test.pdf")
    multi_page_chunks = [c for c in chunks if len(c.page_numbers) > 1]
    assert len(multi_page_chunks) > 0
    assert 1 in multi_page_chunks[0].page_numbers
    assert 2 in multi_page_chunks[0].page_numbers


def test_fixed_size_chunker_overlap_content():
    """Consecutive chunks should have expected overlapping characters."""
    text = "abcdefghijklmnopqrstuvwxyz" * 10  # 260 chars
    pages = [PageContent(page_number=1, text=text)]
    chunker = FixedSizeChunker(chunk_size=100, chunk_overlap=20)
    chunks = chunker.chunk(pages, document_id="test", filename="test.pdf")

    assert len(chunks) > 1
    for i in range(len(chunks) - 1):
        overlap_from_current = chunks[i].text[-20:]
        overlap_in_next = chunks[i + 1].text[:20]
        assert overlap_from_current == overlap_in_next
