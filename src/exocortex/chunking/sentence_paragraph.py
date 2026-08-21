"""Sentence and paragraph-aware chunking strategy."""

from __future__ import annotations

import re

from exocortex.chunking.base import BaseChunker
from exocortex.ingestion import Chunk, PageContent


class SentenceParagraphChunker(BaseChunker):
    """Splits text by natural sentence and paragraph boundaries, maintaining sentence overlap."""

    def __init__(
        self,
        chunk_size: int = 512,
        sentence_overlap: int = 1,
    ) -> None:
        """Initialize SentenceParagraphChunker.

        Args:
            chunk_size: Target maximum characters per chunk (must be positive).
            sentence_overlap: Number of sentences to overlap between consecutive chunks (must be >= 0).

        Raises:
            ValueError: If chunk_size <= 0 or sentence_overlap < 0.
        """
        if chunk_size <= 0:
            raise ValueError("chunk_size must be positive")
        if sentence_overlap < 0:
            raise ValueError("sentence_overlap must be non-negative")
        self.chunk_size = chunk_size
        self.sentence_overlap = sentence_overlap

    @staticmethod
    def _split_into_sentences(text: str) -> list[str]:
        """Split text into sentences using regex boundary matching.

        Args:
            text: Text to split into sentences.

        Returns:
            List of non-empty sentence strings.
        """
        sentence_end = re.compile(r'(?<=[.!?])\s+(?=[A-Z0-9"\'])')
        sentences = sentence_end.split(text)
        return [s.strip() for s in sentences if s.strip()]

    def chunk(
        self,
        pages: list[PageContent],
        document_id: str,
        filename: str,
    ) -> list[Chunk]:
        """Split page contents into sentence and paragraph aware chunks.

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
        paragraphs = full_text.split("\n\n")

        all_sentences: list[str] = []
        for p in paragraphs:
            p = p.strip()
            if not p:
                continue
            all_sentences.extend(self._split_into_sentences(p))

        if not all_sentences:
            return []

        chunks: list[Chunk] = []
        current_sentences: list[str] = []
        current_len = 0
        chunk_index = 0
        search_start = 0

        i = 0
        while i < len(all_sentences):
            sent = all_sentences[i]
            sent_len = len(sent) + 1

            if current_len + sent_len > self.chunk_size and current_sentences:
                chunk_text = " ".join(current_sentences).strip()
                start_pos = full_text.find(chunk_text[:min(50, len(chunk_text))], search_start)
                if start_pos == -1:
                    start_pos = search_start
                end_pos = start_pos + len(chunk_text)
                search_start = max(0, start_pos)

                chunk_pages = self.find_pages_for_range(segments, start_pos, end_pos)
                chunks.append(
                    Chunk(
                        text=chunk_text,
                        document_id=document_id,
                        filename=filename,
                        page_numbers=chunk_pages,
                        chunk_index=chunk_index,
                        metadata={"strategy": "sentence_paragraph"},
                    )
                )
                chunk_index += 1

                # Overlap step: slide back by sentence_overlap
                overlap_count = min(self.sentence_overlap, len(current_sentences))
                if overlap_count > 0:
                    current_sentences = current_sentences[-overlap_count:]
                    current_len = sum(len(s) + 1 for s in current_sentences)
                else:
                    current_sentences = []
                    current_len = 0

            current_sentences.append(sent)
            current_len += sent_len
            i += 1

        if current_sentences:
            chunk_text = " ".join(current_sentences).strip()
            start_pos = full_text.find(chunk_text[:min(50, len(chunk_text))], search_start)
            if start_pos == -1:
                start_pos = search_start
            end_pos = start_pos + len(chunk_text)
            chunk_pages = self.find_pages_for_range(segments, start_pos, end_pos)
            chunks.append(
                Chunk(
                    text=chunk_text,
                    document_id=document_id,
                    filename=filename,
                    page_numbers=chunk_pages,
                    chunk_index=chunk_index,
                    metadata={"strategy": "sentence_paragraph"},
                )
            )

        return chunks
