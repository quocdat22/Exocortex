"""Tests for SentenceParagraphChunker strategy."""

from __future__ import annotations

import pytest

from exocortex.chunking.sentence_paragraph import SentenceParagraphChunker
from exocortex.ingestion import Chunk, PageContent


def test_sentence_paragraph_chunker_basic():
    """SentenceParagraphChunker should split text across sentences and maintain overlap."""
    chunker = SentenceParagraphChunker(chunk_size=100, sentence_overlap=1)
    text = (
        "Sentence one is here. Sentence two follows it. Sentence three is right next. "
        "Sentence four continues the discussion. Sentence five wraps up."
    )
    pages = [PageContent(page_number=1, text=text)]
    chunks = chunker.chunk(pages, document_id="doc1", filename="doc1.pdf")

    assert len(chunks) >= 2
    assert all(isinstance(c, Chunk) for c in chunks)
    assert "Sentence one is here." in chunks[0].text
    assert chunks[0].document_id == "doc1"
    assert chunks[0].filename == "doc1.pdf"
    assert chunks[0].chunk_index == 0
    assert chunks[0].page_numbers == [1]
    assert chunks[0].metadata["strategy"] == "sentence_paragraph"


def test_sentence_paragraph_chunker_validation():
    """SentenceParagraphChunker should reject invalid parameter combinations."""
    with pytest.raises(ValueError, match="chunk_size must be positive"):
        SentenceParagraphChunker(chunk_size=0, sentence_overlap=0)

    with pytest.raises(ValueError, match="chunk_size must be positive"):
        SentenceParagraphChunker(chunk_size=-10, sentence_overlap=1)

    with pytest.raises(ValueError, match="sentence_overlap must be non-negative"):
        SentenceParagraphChunker(chunk_size=100, sentence_overlap=-1)


def test_sentence_paragraph_chunker_empty_pages():
    """SentenceParagraphChunker should return empty list for empty pages."""
    chunker = SentenceParagraphChunker(chunk_size=100, sentence_overlap=1)
    assert chunker.chunk([], document_id="doc1", filename="test.pdf") == []


def test_sentence_paragraph_chunker_blank_text():
    """SentenceParagraphChunker should return empty list for blank/whitespace text."""
    chunker = SentenceParagraphChunker(chunk_size=100, sentence_overlap=1)
    pages = [PageContent(page_number=1, text="   \n\n   \n\t  ")]
    assert chunker.chunk(pages, document_id="doc1", filename="test.pdf") == []


def test_sentence_paragraph_chunker_multi_page_span():
    """SentenceParagraphChunker should track all pages spanned by a chunk."""
    chunker = SentenceParagraphChunker(chunk_size=200, sentence_overlap=1)
    pages = [
        PageContent(page_number=1, text="First sentence on page one. Second sentence on page one."),
        PageContent(page_number=2, text="Third sentence on page two. Fourth sentence on page two."),
    ]
    chunks = chunker.chunk(pages, document_id="doc1", filename="test.pdf")
    assert len(chunks) > 0
    multi_page_chunks = [c for c in chunks if len(c.page_numbers) > 1]
    assert len(multi_page_chunks) > 0
    assert 1 in multi_page_chunks[0].page_numbers
    assert 2 in multi_page_chunks[0].page_numbers


def test_sentence_paragraph_chunker_split_sentences():
    """_split_into_sentences should split on punctuation followed by uppercase/quotes/digits."""
    text = 'Sentence one! Sentence two. Sentence 3 is here. "Sentence 4 is quoted."'
    sentences = SentenceParagraphChunker._split_into_sentences(text)
    assert len(sentences) == 4
    assert sentences[0] == "Sentence one!"
    assert sentences[1] == "Sentence two."
    assert sentences[2] == "Sentence 3 is here."
    assert sentences[3] == '"Sentence 4 is quoted."'


def test_sentence_paragraph_chunker_paragraph_boundaries():
    """SentenceParagraphChunker should handle multi-paragraph texts separated by double newlines."""
    chunker = SentenceParagraphChunker(chunk_size=120, sentence_overlap=1)
    text = (
        "Paragraph one has sentence A. Paragraph one has sentence B.\n\n"
        "Paragraph two has sentence C. Paragraph two has sentence D.\n\n"
        "Paragraph three has sentence E."
    )
    pages = [PageContent(page_number=1, text=text)]
    chunks = chunker.chunk(pages, document_id="doc_para", filename="para.pdf")

    assert len(chunks) >= 2
    assert all(c.metadata["strategy"] == "sentence_paragraph" for c in chunks)


def test_sentence_paragraph_chunker_sentence_overlap():
    """SentenceParagraphChunker should overlap the configured number of sentences."""
    chunker = SentenceParagraphChunker(chunk_size=100, sentence_overlap=1)
    text = (
        "Alpha sentence is first. Beta sentence is second. "
        "Gamma sentence is third. Delta sentence is fourth."
    )
    pages = [PageContent(page_number=1, text=text)]
    chunks = chunker.chunk(pages, document_id="doc_overlap", filename="overlap.pdf")

    assert len(chunks) >= 2
    # Verify the overlap sentence is present in both chunks
    assert any(
        "Beta sentence is second." in chunks[0].text and "Beta sentence is second." in chunks[1].text
        for _ in [1]
    ) or any(
        "Gamma sentence is third." in chunks[0].text and "Gamma sentence is third." in chunks[1].text
        for _ in [1]
    )


def test_sentence_paragraph_chunker_zero_overlap():
    """SentenceParagraphChunker should support zero sentence overlap."""
    chunker = SentenceParagraphChunker(chunk_size=80, sentence_overlap=0)
    text = (
        "First sentence here. Second sentence here. "
        "Third sentence here. Fourth sentence here."
    )
    pages = [PageContent(page_number=1, text=text)]
    chunks = chunker.chunk(pages, document_id="doc_zero", filename="zero.pdf")

    assert len(chunks) >= 2
    # No sentence in chunk[0] should be in chunk[1]
    sentences_0 = SentenceParagraphChunker._split_into_sentences(chunks[0].text)
    sentences_1 = SentenceParagraphChunker._split_into_sentences(chunks[1].text)
    assert set(sentences_0).isdisjoint(set(sentences_1))


def test_sentence_paragraph_chunker_oversized_single_sentence():
    """A sentence longer than chunk_size should still be emitted without infinite loop."""
    chunker = SentenceParagraphChunker(chunk_size=50, sentence_overlap=1)
    long_sentence = "This is an extremely long sentence that definitely exceeds the chunk size limit of fifty characters."
    pages = [PageContent(page_number=1, text=long_sentence)]
    chunks = chunker.chunk(pages, document_id="doc_long", filename="long.pdf")

    assert len(chunks) == 1
    assert chunks[0].text == long_sentence
    assert chunks[0].chunk_index == 0


def test_sentence_paragraph_chunker_sequential_indices():
    """SentenceParagraphChunker should assign sequential chunk indices."""
    chunker = SentenceParagraphChunker(chunk_size=60, sentence_overlap=1)
    text = (
        "Sentence one is here. Sentence two is here. "
        "Sentence three is here. Sentence four is here. "
        "Sentence five is here."
    )
    pages = [PageContent(page_number=1, text=text)]
    chunks = chunker.chunk(pages, document_id="doc_idx", filename="idx.pdf")

    assert len(chunks) > 1
    for expected_idx, c in enumerate(chunks):
        assert c.chunk_index == expected_idx


def test_sentence_paragraph_chunker_repeated_sentences_page_mapping():
    """SentenceParagraphChunker should map pages correctly even when sentences repeat."""
    chunker = SentenceParagraphChunker(chunk_size=70, sentence_overlap=0)
    pages = [
        PageContent(page_number=1, text="Repeated header text. Specific page one content."),
        PageContent(page_number=2, text="Repeated header text. Specific page two content."),
    ]
    chunks = chunker.chunk(pages, document_id="doc_repeat", filename="repeat.pdf")

    assert len(chunks) >= 2
    # The first chunk should map to page 1
    assert chunks[0].page_numbers == [1]
    # The last chunk should map to page 2
    assert 2 in chunks[-1].page_numbers


def test_sentence_paragraph_chunker_multiple_consecutive_newlines():
    """SentenceParagraphChunker should handle texts with varied whitespace and newlines."""
    chunker = SentenceParagraphChunker(chunk_size=100, sentence_overlap=1)
    text = "\n\n\n   Sentence A is here.   \n\n\n\n   Sentence B is here.   \n\n"
    pages = [PageContent(page_number=1, text=text)]
    chunks = chunker.chunk(pages, document_id="doc_space", filename="space.pdf")

    assert len(chunks) >= 1
    assert all(isinstance(c, Chunk) for c in chunks)

