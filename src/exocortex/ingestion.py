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
from typing import Any

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
    file_hash: str = ""  # SHA-256 content hash of the source PDF
    metadata: dict = field(default_factory=dict)  # Additional metadata

    def to_metadata_dict(self) -> dict:
        """Convert chunk metadata to a flat dict for ChromaDB storage."""
        data = {
            "document_id": self.document_id,
            "filename": self.filename,
            "page_numbers": ",".join(str(p) for p in self.page_numbers),
            "chunk_index": self.chunk_index,
        }
        if self.file_hash:
            data["file_hash"] = self.file_hash
        return data


def compute_file_hash(pdf_source: Path | bytes | str) -> str:
    """Compute SHA-256 hash of PDF raw bytes.

    Args:
        pdf_source: Path to PDF or raw bytes.

    Returns:
        Hex string of SHA-256 hash.
    """
    if isinstance(pdf_source, (str, Path)):
        path = Path(pdf_source)
        if not path.exists():
            return hashlib.sha256(str(path).encode()).hexdigest()
        return hashlib.sha256(path.read_bytes()).hexdigest()
    elif isinstance(pdf_source, bytes):
        return hashlib.sha256(pdf_source).hexdigest()
    else:
        raise TypeError(f"Unsupported type for compute_file_hash: {type(pdf_source)}")


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
    chunk_size: int | None = None,
    chunk_overlap: int | None = None,
    strategy: str | None = None,
    **kwargs: Any,
) -> list[Chunk]:
    """Full ingestion pipeline: PDF → pages → chunks.

    This is the main entry point for PDF ingestion. It:
    1. Extracts text from the PDF
    2. Generates a document ID
    3. Chunks the text using the specified chunking strategy (default: from Settings or "recursive")

    Args:
        pdf_path: Path to the PDF file.
        chunk_size: Maximum characters per chunk (default: from Settings or 512).
        chunk_overlap: Overlap characters between chunks (default: from Settings or 50).
        strategy: Chunking strategy name (default: from Settings or "recursive").
        **kwargs: Additional strategy-specific arguments passed to get_chunker.

    Returns:
        List of Chunk objects ready for embedding.

    Example:
        >>> chunks = ingest_pdf(Path("data/ebooks/my_book.pdf"))
        >>> print(f"Created {len(chunks)} chunks from {chunks[0].filename}")
    """
    from exocortex.chunking.factory import get_chunker

    filename = pdf_path.name
    document_id = generate_document_id(filename)

    pages = extract_text_from_pdf(pdf_path)

    # Resolve defaults from Settings if not explicitly provided
    resolved_strategy = strategy
    resolved_chunk_size = chunk_size
    resolved_chunk_overlap = chunk_overlap

    if resolved_strategy is None or resolved_chunk_size is None or resolved_chunk_overlap is None:
        try:
            from exocortex.config import get_settings

            settings = get_settings()
            if resolved_strategy is None:
                resolved_strategy = settings.chunking_strategy
            if resolved_chunk_size is None:
                resolved_chunk_size = settings.chunk_size
            if resolved_chunk_overlap is None:
                resolved_chunk_overlap = settings.chunk_overlap
        except (ImportError, AttributeError, ValueError, TypeError):
            if resolved_strategy is None:
                resolved_strategy = "recursive"
            if resolved_chunk_size is None:
                resolved_chunk_size = 512
            if resolved_chunk_overlap is None:
                resolved_chunk_overlap = 50

    chunker = get_chunker(
        strategy=resolved_strategy,
        chunk_size=resolved_chunk_size,
        chunk_overlap=resolved_chunk_overlap,
        **kwargs,
    )
    chunks = chunker.chunk(pages=pages, document_id=document_id, filename=filename)
    file_hash = compute_file_hash(pdf_path)
    for c in chunks:
        c.file_hash = file_hash
    return chunks
