# Phase 2: PDF Ingestion

## Objective

Build the ingestion pipeline that takes a PDF ebook file, extracts text page by page
using PyMuPDF, and splits the text into fixed-size chunks with overlap. This phase
produces structured `Chunk` objects ready for embedding in Phase 3.

## Prerequisites

- Phase 1 completed (project setup, config module working)
- `uv sync` ran successfully

---

## Concepts

### Chunking Strategy

We use **fixed-size character chunking with overlap**:

```
Document Text:  [==========|==========|==========|==========]
                                                              
Chunk 1:        [==========|==]                               
Chunk 2:                [==|==========|==]                    
Chunk 3:                         [==|==========|==]           
Chunk 4:                                  [==|==========]     
                    ▲                                         
                    └── overlap region                        
```

- **chunk_size**: Maximum number of characters per chunk (default: 512)
- **chunk_overlap**: Number of overlapping characters between consecutive chunks (default: 50)
- Overlap ensures context is not lost at chunk boundaries

> **Note on tokens vs characters:** The config specifies `chunk_size=512` as a token
> target. For English text, 1 token ≈ 4 characters. However, for simplicity in this
> basic RAG, we chunk by **characters** (not tokens). This is a known limitation.
> A future improvement would use a proper tokenizer. For now, use `chunk_size` as
> character count with the understanding that 512 chars ≈ 128 tokens, which is well
> within the embedding model's context window.

### Metadata

Each chunk carries metadata for source attribution:
- `document_id`: Unique identifier for the source document
- `filename`: Original PDF filename
- `page_number`: Page(s) the chunk came from
- `chunk_index`: Sequential index within the document

---

## Implementation

### File: `src/exocortex/ingestion.py`

```python
"""PDF ingestion and text chunking pipeline.

Handles:
1. Extracting text from PDF files using PyMuPDF (fitz)
2. Splitting extracted text into fixed-size chunks with overlap
3. Attaching metadata (filename, page number, chunk index) to each chunk
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from pathlib import Path

import fitz  # PyMuPDF


@dataclass
class PageContent:
    """Represents extracted text from a single PDF page."""

    page_number: int  # 1-indexed page number
    text: str  # Extracted text content


@dataclass
class Chunk:
    """A text chunk ready for embedding, with source metadata."""

    text: str  # The chunk text content
    document_id: str  # Unique document identifier (hash of filename)
    filename: str  # Original PDF filename
    page_numbers: list[int]  # Page(s) this chunk spans (1-indexed)
    chunk_index: int  # Sequential index within the document
    metadata: dict = field(default_factory=dict)  # Additional metadata

    def to_metadata_dict(self) -> dict:
        """Convert chunk metadata to a flat dict for ChromaDB storage."""
        return {
            "document_id": self.document_id,
            "filename": self.filename,
            "page_numbers": ",".join(str(p) for p in self.page_numbers),
            "chunk_index": self.chunk_index,
        }


def generate_document_id(filename: str) -> str:
    """Generate a stable document ID from filename.

    Uses SHA-256 hash of the filename to create a unique, deterministic ID.

    Args:
        filename: The PDF filename (not full path).

    Returns:
        A hex string document ID (first 16 chars of SHA-256).
    """
    return hashlib.sha256(filename.encode()).hexdigest()[:16]


def extract_text_from_pdf(pdf_path: Path) -> list[PageContent]:
    """Extract text from a PDF file, page by page.

    Uses PyMuPDF (fitz) to extract text with layout preservation.
    Skips pages with no extractable text (e.g., image-only pages).

    Args:
        pdf_path: Path to the PDF file.

    Returns:
        List of PageContent objects, one per page with text.

    Raises:
        FileNotFoundError: If the PDF file does not exist.
        ValueError: If the file is not a valid PDF or has no extractable text.
    """
    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF file not found: {pdf_path}")

    if not pdf_path.suffix.lower() == ".pdf":
        raise ValueError(f"Not a PDF file: {pdf_path}")

    pages: list[PageContent] = []

    doc = fitz.open(pdf_path)
    try:
        for page_num in range(len(doc)):
            page = doc[page_num]
            text = page.get_text("text")  # Extract as plain text

            # Clean up: normalize whitespace, strip leading/trailing
            text = text.strip()

            if text:  # Skip empty pages
                pages.append(
                    PageContent(
                        page_number=page_num + 1,  # 1-indexed
                        text=text,
                    )
                )
    finally:
        doc.close()

    if not pages:
        raise ValueError(
            f"No extractable text found in {pdf_path}. "
            "The PDF may be scanned/image-based (OCR not supported)."
        )

    return pages


def chunk_text(
    pages: list[PageContent],
    chunk_size: int,
    chunk_overlap: int,
    document_id: str,
    filename: str,
) -> list[Chunk]:
    """Split page text into fixed-size chunks with overlap.

    Concatenates all page texts with page boundary markers, then splits
    into chunks of `chunk_size` characters with `chunk_overlap` character overlap.
    Tracks which page(s) each chunk spans.

    Args:
        pages: List of PageContent from extract_text_from_pdf().
        chunk_size: Maximum characters per chunk.
        chunk_overlap: Overlap characters between consecutive chunks.
        document_id: Unique document identifier.
        filename: Original PDF filename.

    Returns:
        List of Chunk objects ready for embedding.

    Raises:
        ValueError: If chunk_size <= chunk_overlap or inputs are invalid.
    """
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    if chunk_overlap < 0:
        raise ValueError("chunk_overlap must be non-negative")
    if chunk_overlap >= chunk_size:
        raise ValueError("chunk_overlap must be less than chunk_size")
    if not pages:
        return []

    # Build a list of (char_offset, page_number) for page tracking
    # Concatenate all page texts with a space separator
    segments: list[tuple[int, int, str]] = []  # (start_offset, page_number, text)
    current_offset = 0

    for page in pages:
        segments.append((current_offset, page.page_number, page.text))
        current_offset += len(page.text) + 1  # +1 for the space separator

    full_text = " ".join(page.text for page in pages)

    # Slide window across full_text
    chunks: list[Chunk] = []
    step = chunk_size - chunk_overlap
    chunk_index = 0

    for start in range(0, len(full_text), step):
        end = min(start + chunk_size, len(full_text))
        chunk_text_content = full_text[start:end].strip()

        if not chunk_text_content:
            continue

        # Determine which pages this chunk spans
        chunk_pages = _find_pages_for_range(segments, start, end)

        chunks.append(
            Chunk(
                text=chunk_text_content,
                document_id=document_id,
                filename=filename,
                page_numbers=chunk_pages,
                chunk_index=chunk_index,
            )
        )
        chunk_index += 1

        # Stop if we've reached the end
        if end >= len(full_text):
            break

    return chunks


def _find_pages_for_range(
    segments: list[tuple[int, int, str]],
    start: int,
    end: int,
) -> list[int]:
    """Find which page numbers a character range spans.

    Args:
        segments: List of (start_offset, page_number, text) tuples.
        start: Start character offset.
        end: End character offset.

    Returns:
        Sorted list of page numbers the range spans.
    """
    pages = set()
    for seg_start, page_num, text in segments:
        seg_end = seg_start + len(text)
        # Check if chunk range overlaps with this segment
        if start < seg_end and end > seg_start:
            pages.add(page_num)
    return sorted(pages)


def ingest_pdf(
    pdf_path: Path,
    chunk_size: int = 512,
    chunk_overlap: int = 50,
) -> list[Chunk]:
    """Full ingestion pipeline: PDF → pages → chunks.

    This is the main entry point for PDF ingestion. It:
    1. Extracts text from the PDF
    2. Generates a document ID
    3. Chunks the text with overlap

    Args:
        pdf_path: Path to the PDF file.
        chunk_size: Maximum characters per chunk.
        chunk_overlap: Overlap characters between chunks.

    Returns:
        List of Chunk objects ready for embedding.

    Example:
        >>> chunks = ingest_pdf(Path("data/ebooks/my_book.pdf"))
        >>> print(f"Created {len(chunks)} chunks from {chunks[0].filename}")
    """
    filename = pdf_path.name
    document_id = generate_document_id(filename)

    pages = extract_text_from_pdf(pdf_path)
    chunks = chunk_text(
        pages=pages,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        document_id=document_id,
        filename=filename,
    )

    return chunks
```

---

## Tests

### File: `tests/test_ingestion.py`

```python
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
```

---

## Success Criteria

Phase 2 is complete when ALL of the following are true:

### Automated Tests
```bash
# Unit tests (no PDF required)
uv run pytest tests/test_ingestion.py -v -k "not integration" --log-cli-level=INFO

# Integration test (requires at least 1 PDF in data/ebooks/)
uv run pytest tests/test_ingestion.py -v -k "integration" --log-cli-level=INFO
```

**Expected:** All unit tests PASSED. Integration test PASSED if a PDF exists, SKIPPED otherwise.

### Manual Verification
```bash
uv run python -c "
from pathlib import Path
from exocortex.ingestion import ingest_pdf

# Replace with an actual PDF path
pdf = next(Path('data/ebooks').glob('*.pdf'), None)
if pdf:
    chunks = ingest_pdf(pdf)
    print(f'Parsed: {pdf.name}')
    print(f'Total chunks: {len(chunks)}')
    print(f'Chunk 0 pages: {chunks[0].page_numbers}')
    print(f'Chunk 0 preview: {chunks[0].text[:200]}')
else:
    print('No PDF found in data/ebooks/')
"
```

### Checklist
- [ ] `src/exocortex/ingestion.py` exists with all specified functions
- [ ] `extract_text_from_pdf()` correctly extracts text page by page
- [ ] `chunk_text()` splits text into chunks with correct overlap
- [ ] Each `Chunk` has correct metadata (document_id, filename, page_numbers, chunk_index)
- [ ] Error handling: FileNotFoundError, ValueError for invalid inputs
- [ ] `tests/test_ingestion.py` — all unit tests pass
- [ ] Integration test passes with a real PDF (if available)
