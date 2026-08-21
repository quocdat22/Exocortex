"""Tests for RecursiveCharacterChunker strategy."""

from __future__ import annotations

import pytest

from exocortex.chunking.recursive import RecursiveCharacterChunker
from exocortex.ingestion import Chunk, PageContent


def test_recursive_character_chunker_splits_paragraphs_and_sentences():
    """RecursiveCharacterChunker should split hierarchical text cleanly."""
    chunker = RecursiveCharacterChunker(chunk_size=120, chunk_overlap=20)
    text = (
        "Paragraph 1 is about ML systems.\n\n"
        "Paragraph 2 is about training and inference bottlenecks.\n\n"
        "Paragraph 3 discusses latency and throughput."
    )
    pages = [PageContent(page_number=1, text=text)]
    chunks = chunker.chunk(pages, document_id="doc1", filename="doc1.pdf")

    assert len(chunks) >= 2
    assert all(isinstance(c, Chunk) for c in chunks)
    assert all(len(c.text) <= 150 for c in chunks)
    assert all(c.page_numbers == [1] for c in chunks)
    assert chunks[0].chunk_index == 0
    assert chunks[0].document_id == "doc1"
    assert chunks[0].filename == "doc1.pdf"
    assert chunks[0].metadata.get("strategy") == "recursive"


def test_recursive_character_chunker_validation():
    """RecursiveCharacterChunker should validate chunk_size and chunk_overlap."""
    with pytest.raises(ValueError, match="chunk_size must be positive"):
        RecursiveCharacterChunker(chunk_size=0, chunk_overlap=0)

    with pytest.raises(ValueError, match="chunk_size must be positive"):
        RecursiveCharacterChunker(chunk_size=-10, chunk_overlap=0)

    with pytest.raises(ValueError, match="chunk_overlap must be non-negative"):
        RecursiveCharacterChunker(chunk_size=50, chunk_overlap=-5)

    with pytest.raises(ValueError, match="chunk_overlap must be less than chunk_size"):
        RecursiveCharacterChunker(chunk_size=50, chunk_overlap=50)

    with pytest.raises(ValueError, match="chunk_overlap must be less than chunk_size"):
        RecursiveCharacterChunker(chunk_size=50, chunk_overlap=60)


def test_recursive_character_chunker_empty_pages():
    """RecursiveCharacterChunker should return empty list for empty pages."""
    chunker = RecursiveCharacterChunker(chunk_size=100, chunk_overlap=20)
    assert chunker.chunk([], document_id="doc1", filename="test.pdf") == []


def test_recursive_character_chunker_blank_text():
    """RecursiveCharacterChunker should return empty list for blank text."""
    chunker = RecursiveCharacterChunker(chunk_size=100, chunk_overlap=20)
    pages = [PageContent(page_number=1, text="   \n\n   \n\t  ")]
    assert chunker.chunk(pages, document_id="doc1", filename="test.pdf") == []


def test_recursive_character_chunker_multi_page_span():
    """RecursiveCharacterChunker should track all pages spanned by a chunk."""
    chunker = RecursiveCharacterChunker(chunk_size=150, chunk_overlap=20)
    pages = [
        PageContent(page_number=1, text="This is the first page of content for testing. " * 2),
        PageContent(page_number=2, text="This is the second page of content for testing. " * 2),
    ]
    chunks = chunker.chunk(pages, document_id="doc1", filename="test.pdf")
    assert len(chunks) > 0
    multi_page_chunks = [c for c in chunks if len(c.page_numbers) > 1]
    assert len(multi_page_chunks) > 0
    assert 1 in multi_page_chunks[0].page_numbers
    assert 2 in multi_page_chunks[0].page_numbers


def test_recursive_character_chunker_custom_separators():
    """RecursiveCharacterChunker should honor custom separators list."""
    chunker = RecursiveCharacterChunker(
        chunk_size=50,
        chunk_overlap=10,
        separators=["|", ";", ""],
    )
    text = "Section A|Section B;SubSection B1;SubSection B2|Section C"
    pages = [PageContent(page_number=1, text=text)]
    chunks = chunker.chunk(pages, document_id="doc_custom", filename="custom.pdf")

    assert len(chunks) >= 2
    assert all(c.metadata.get("strategy") == "recursive" for c in chunks)


def test_recursive_character_chunker_fallback_character_split():
    """RecursiveCharacterChunker should split by character when no separator fits."""
    chunker = RecursiveCharacterChunker(chunk_size=50, chunk_overlap=10)
    long_string = "X" * 120
    pages = [PageContent(page_number=1, text=long_string)]
    chunks = chunker.chunk(pages, document_id="doc_fallback", filename="fallback.pdf")

    assert len(chunks) >= 2
    assert all(len(c.text) <= 50 for c in chunks)


def test_recursive_character_chunker_sequential_indices():
    """RecursiveCharacterChunker should generate sequential chunk indices."""
    chunker = RecursiveCharacterChunker(chunk_size=60, chunk_overlap=10)
    text = (
        "Line 1 has some words to fill up character count.\n\n"
        "Line 2 has even more words to exceed the chunk size.\n\n"
        "Line 3 continues with additional information.\n\n"
        "Line 4 wraps up the paragraph content."
    )
    pages = [PageContent(page_number=1, text=text)]
    chunks = chunker.chunk(pages, document_id="doc_idx", filename="idx.pdf")

    assert len(chunks) > 1
    for expected_idx, c in enumerate(chunks):
        assert c.chunk_index == expected_idx
