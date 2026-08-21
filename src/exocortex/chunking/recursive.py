"""Recursive character text splitting strategy."""

from __future__ import annotations

from exocortex.chunking.base import BaseChunker
from exocortex.ingestion import Chunk, PageContent


class RecursiveCharacterChunker(BaseChunker):
    """Recursively splits text on hierarchical separators (paragraphs -> sentences -> words)."""

    def __init__(
        self,
        chunk_size: int = 512,
        chunk_overlap: int = 50,
        separators: list[str] | None = None,
    ) -> None:
        """Initialize RecursiveCharacterChunker.

        Args:
            chunk_size: Maximum characters per chunk (must be positive).
            chunk_overlap: Overlap characters between consecutive chunks (must be >= 0 and < chunk_size).
            separators: List of separators in priority order. Defaults to ["\n\n", "\n", ". ", " ", ""].

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
        self.separators = separators if separators is not None else ["\n\n", "\n", ". ", " ", ""]

    def _split_text(self, text: str, separators: list[str]) -> list[str]:
        """Split text recursively using the given separators."""
        final_chunks: list[str] = []
        separator = separators[-1]
        new_separators = []
        for i, sep in enumerate(separators):
            if sep == "":
                separator = ""
                break
            if sep in text:
                separator = sep
                new_separators = separators[i + 1:]
                break

        splits = text.split(separator) if separator else list(text)
        good_splits: list[str] = []
        _separator = "" if separator == "" else separator

        for s in splits:
            if len(s) < self.chunk_size:
                good_splits.append(s)
            else:
                if good_splits:
                    merged = self._merge_splits(good_splits, _separator)
                    final_chunks.extend(merged)
                    good_splits = []
                if not new_separators:
                    final_chunks.append(s)
                else:
                    other_info = self._split_text(s, new_separators)
                    final_chunks.extend(other_info)

        if good_splits:
            merged = self._merge_splits(good_splits, _separator)
            final_chunks.extend(merged)

        return final_chunks

    def _merge_splits(self, splits: list[str], separator: str) -> list[str]:
        """Combine small splits into chunks up to chunk_size with overlap."""
        docs: list[str] = []
        current_doc: list[str] = []
        total = 0

        for d in splits:
            _len = len(d)
            if total + _len + (len(separator) if current_doc else 0) > self.chunk_size:
                if total > 0:
                    doc = separator.join(current_doc).strip()
                    if doc:
                        docs.append(doc)
                    # Keep overlap from the end of current_doc
                    while total > self.chunk_overlap and current_doc:
                        popped = current_doc.pop(0)
                        total -= len(popped) + len(separator)
                        if total < 0:
                            total = 0
                current_doc.append(d)
                total += _len + (len(separator) if len(current_doc) > 1 else 0)
            else:
                current_doc.append(d)
                total += _len + (len(separator) if len(current_doc) > 1 else 0)

        if current_doc:
            doc = separator.join(current_doc).strip()
            if doc:
                docs.append(doc)

        return docs

    def chunk(
        self,
        pages: list[PageContent],
        document_id: str,
        filename: str,
    ) -> list[Chunk]:
        """Split page contents into recursive character chunks.

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
        raw_chunks = self._split_text(full_text, self.separators)

        chunks: list[Chunk] = []
        current_search_start = 0

        for text_chunk in raw_chunks:
            text_chunk = text_chunk.strip()
            if not text_chunk:
                continue

            # Locate position in full_text to get page mapping
            start_pos = full_text.find(text_chunk[:min(50, len(text_chunk))], current_search_start)
            if start_pos == -1:
                start_pos = current_search_start
            end_pos = start_pos + len(text_chunk)
            current_search_start = max(0, start_pos + max(1, len(text_chunk) - self.chunk_overlap))

            chunk_pages = self.find_pages_for_range(segments, start_pos, end_pos)
            chunks.append(
                Chunk(
                    text=text_chunk,
                    document_id=document_id,
                    filename=filename,
                    page_numbers=chunk_pages,
                    chunk_index=len(chunks),
                    metadata={"strategy": "recursive"},
                )
            )

        return chunks
