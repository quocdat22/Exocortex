"""Fixed-size window chunking strategy with sliding overlap."""

from __future__ import annotations

from exocortex.chunking.base import BaseChunker
from exocortex.ingestion import Chunk, PageContent


class FixedSizeChunker(BaseChunker):
    """Fixed character window chunking with sliding overlap."""

    def __init__(self, chunk_size: int = 512, chunk_overlap: int = 50) -> None:
        """Initialize FixedSizeChunker.

        Args:
            chunk_size: Maximum characters per chunk (must be positive).
            chunk_overlap: Overlap characters between consecutive chunks (must be >= 0 and < chunk_size).

        Raises:
            ValueError: If chunk_size <= 0, chunk_overlap < 0, or chunk_overlap >= chunk_size.
        """
        if chunk_size <= 0:
            raise ValueError("chunk_size must be positive")
        if chunk_overlap < 0:
            raise ValueError("chunk_overlap must be non-negative")
        if chunk_overlap >= chunk_size:
            raise ValueError("chunk_overlap must be less than chunk_size")
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def chunk(
        self,
        pages: list[PageContent],
        document_id: str,
        filename: str,
    ) -> list[Chunk]:
        """Split page contents into fixed-size chunks with overlap.

        Args:
            pages: List of PageContent from extract_text_from_pdf().
            document_id: Unique document identifier.
            filename: Original PDF filename.

        Returns:
            List of Chunk objects.
        """
        if not pages:
            return []

        full_text, segments = self.build_page_segments(pages)
        chunks: list[Chunk] = []
        step = self.chunk_size - self.chunk_overlap
        chunk_index = 0

        for start in range(0, len(full_text), step):
            end = min(start + self.chunk_size, len(full_text))
            chunk_text_content = full_text[start:end].strip()

            if not chunk_text_content:
                continue

            chunk_pages = self.find_pages_for_range(segments, start, end)
            chunks.append(
                Chunk(
                    text=chunk_text_content,
                    document_id=document_id,
                    filename=filename,
                    page_numbers=chunk_pages,
                    chunk_index=chunk_index,
                    metadata={"strategy": "fixed_size"},
                )
            )
            chunk_index += 1
            if end >= len(full_text):
                break

        return chunks
