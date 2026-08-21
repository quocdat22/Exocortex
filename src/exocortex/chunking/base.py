"""Base interface for chunking strategies in Exocortex."""

from __future__ import annotations

from abc import ABC, abstractmethod

from exocortex.ingestion import Chunk, PageContent


class BaseChunker(ABC):
    """Abstract base class for all chunking strategies."""

    @abstractmethod
    def chunk(
        self,
        pages: list[PageContent],
        document_id: str,
        filename: str,
    ) -> list[Chunk]:
        """Split page contents into a list of Chunks.

        Args:
            pages: List of PageContent extracted from PDF.
            document_id: Unique document identifier.
            filename: Original PDF filename.

        Returns:
            List of Chunk objects.
        """
        pass

    @staticmethod
    def build_page_segments(
        pages: list[PageContent],
    ) -> tuple[str, list[tuple[int, int, str]]]:
        """Build concatenated text and offset mappings for page tracking.

        Args:
            pages: List of PageContent extracted from PDF.

        Returns:
            Tuple of (full_text, segments) where segments is a list of
            (start_offset, page_number, page_text).
        """
        segments: list[tuple[int, int, str]] = []
        current_offset = 0
        for page in pages:
            segments.append((current_offset, page.page_number, page.text))
            current_offset += len(page.text) + 1
        full_text = " ".join(page.text for page in pages)
        return full_text, segments

    @staticmethod
    def find_pages_for_range(
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
            Sorted list of 1-indexed page numbers the range spans.
        """
        pages = set()
        for seg_start, page_num, text in segments:
            seg_end = seg_start + len(text)
            if start < seg_end and end > seg_start:
                pages.add(page_num)
        return sorted(pages)
