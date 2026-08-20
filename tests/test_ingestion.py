"""Tests for Phase 2: PDF Ingestion pipeline."""

import logging
from pathlib import Path

import pytest

from exocortex.ingestion import (
    Chunk,
    PageContent,
    chunk_text,
    extract_text_from_pdf,
    generate_document_id,
    ingest_pdf,
)

logger = logging.getLogger(__name__)


# --- Unit Tests: generate_document_id ---


def test_document_id_deterministic():
    """Same filename should produce same ID."""
    id1 = generate_document_id("book.pdf")
    id2 = generate_document_id("book.pdf")
    assert id1 == id2
    assert len(id1) == 16


def test_document_id_unique():
    """Different filenames should produce different IDs."""
    id1 = generate_document_id("book1.pdf")
    id2 = generate_document_id("book2.pdf")
    assert id1 != id2


# --- Unit Tests: chunk_text ---


def test_chunk_text_basic():
    """Chunking should split text into expected number of chunks."""
    pages = [PageContent(page_number=1, text="a" * 1000)]
    chunks = chunk_text(
        pages=pages,
        chunk_size=200,
        chunk_overlap=50,
        document_id="test-doc",
        filename="test.pdf",
    )

    assert len(chunks) > 1
    assert all(len(c.text) <= 200 for c in chunks)
    assert all(c.document_id == "test-doc" for c in chunks)
    assert all(c.filename == "test.pdf" for c in chunks)
    logger.info(f"Created {len(chunks)} chunks from 1000 chars (size=200, overlap=50)")


def test_chunk_text_overlap():
    """Consecutive chunks should have overlapping content."""
    text = "abcdefghijklmnopqrstuvwxyz" * 10  # 260 chars
    pages = [PageContent(page_number=1, text=text)]
    chunks = chunk_text(
        pages=pages,
        chunk_size=100,
        chunk_overlap=20,
        document_id="test",
        filename="test.pdf",
    )

    # Verify overlap: end of chunk N should match start of chunk N+1
    for i in range(len(chunks) - 1):
        overlap_from_current = chunks[i].text[-20:]
        overlap_in_next = chunks[i + 1].text[:20]
        assert overlap_from_current == overlap_in_next, (
            f"Chunk {i} and {i + 1} should overlap"
        )


def test_chunk_text_preserves_page_numbers():
    """Chunks spanning multiple pages should track all page numbers."""
    pages = [
        PageContent(page_number=1, text="a" * 100),
        PageContent(page_number=2, text="b" * 100),
    ]
    chunks = chunk_text(
        pages=pages,
        chunk_size=150,
        chunk_overlap=20,
        document_id="test",
        filename="test.pdf",
    )

    # At least one chunk should span both pages
    multi_page_chunks = [c for c in chunks if len(c.page_numbers) > 1]
    assert len(multi_page_chunks) > 0, "Should have at least one multi-page chunk"


def test_chunk_text_empty_pages():
    """Empty pages list should return empty chunks."""
    chunks = chunk_text(
        pages=[],
        chunk_size=100,
        chunk_overlap=10,
        document_id="test",
        filename="test.pdf",
    )
    assert chunks == []


def test_chunk_text_invalid_params():
    """Invalid chunk parameters should raise ValueError."""
    pages = [PageContent(page_number=1, text="hello")]

    with pytest.raises(ValueError, match="chunk_size must be positive"):
        chunk_text(pages, chunk_size=0, chunk_overlap=0, document_id="x", filename="x")

    with pytest.raises(ValueError, match="chunk_overlap must be less"):
        chunk_text(
            pages, chunk_size=10, chunk_overlap=10, document_id="x", filename="x"
        )


def test_chunk_metadata_dict():
    """Chunk.to_metadata_dict() should return a flat dict."""
    chunk = Chunk(
        text="hello",
        document_id="doc123",
        filename="book.pdf",
        page_numbers=[1, 2, 3],
        chunk_index=5,
    )
    meta = chunk.to_metadata_dict()
    assert meta["document_id"] == "doc123"
    assert meta["filename"] == "book.pdf"
    assert meta["page_numbers"] == "1,2,3"
    assert meta["chunk_index"] == 5


# --- Integration Test: extract_text_from_pdf ---


def test_extract_text_file_not_found():
    """Should raise FileNotFoundError for missing files."""
    with pytest.raises(FileNotFoundError):
        extract_text_from_pdf(Path("/nonexistent/book.pdf"))


def test_extract_text_not_pdf():
    """Should raise ValueError for non-PDF files."""
    # Create a temp text file
    import tempfile

    with tempfile.NamedTemporaryFile(suffix=".txt", delete=False) as f:
        f.write(b"not a pdf")
        temp_path = Path(f.name)

    try:
        with pytest.raises(ValueError, match="Not a PDF"):
            extract_text_from_pdf(temp_path)
    finally:
        temp_path.unlink()


# --- Integration Test: Full Pipeline ---
# These tests require a real PDF file. They are skipped if no test PDF exists.

TEST_PDF_DIR = Path("data/ebooks")


@pytest.fixture
def sample_pdf() -> Path | None:
    """Return path to a sample PDF if one exists in data/ebooks/."""
    if not TEST_PDF_DIR.exists():
        pytest.skip("No data/ebooks/ directory found")

    pdfs = list(TEST_PDF_DIR.glob("*.pdf"))
    if not pdfs:
        pytest.skip("No PDF files found in data/ebooks/")

    return pdfs[0]


def test_ingest_pdf_integration(sample_pdf):
    """Full pipeline: PDF → chunks with metadata."""
    chunks = ingest_pdf(sample_pdf, chunk_size=512, chunk_overlap=50)

    assert len(chunks) > 0
    assert all(isinstance(c, Chunk) for c in chunks)
    assert all(c.filename == sample_pdf.name for c in chunks)
    assert all(len(c.text) > 0 for c in chunks)
    assert all(len(c.page_numbers) > 0 for c in chunks)

    # Log results for observability
    logger.info(f"File: {sample_pdf.name}")
    logger.info(f"Total chunks: {len(chunks)}")
    logger.info(f"First chunk preview: {chunks[0].text[:100]}...")
    logger.info(f"Last chunk page(s): {chunks[-1].page_numbers}")
