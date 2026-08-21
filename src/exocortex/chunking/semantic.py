"""Semantic chunking strategy based on embedding cosine similarity."""

from __future__ import annotations

import math
import re
from typing import TYPE_CHECKING

from exocortex.chunking.base import BaseChunker
from exocortex.ingestion import Chunk, PageContent

if TYPE_CHECKING:
    from exocortex.embedding import EmbeddingClient


class SemanticChunker(BaseChunker):
    """Splits text dynamically based on cosine distance between consecutive sentence embeddings."""

    def __init__(
        self,
        embedding_client: EmbeddingClient | None = None,
        distance_threshold_percentile: float = 85.0,
        min_chunk_size: int = 50,
        max_chunk_size: int = 1000,
    ) -> None:
        """Initialize SemanticChunker.

        Args:
            embedding_client: EmbeddingClient instance used to generate sentence embeddings.
                If None, lazily initialized on first use.
            distance_threshold_percentile: Percentile threshold (0.0 - 100.0) above which
                cosine distance triggers a semantic breakpoint.
            min_chunk_size: Minimum characters before a split can occur.
            max_chunk_size: Maximum characters per chunk before forcing a split.

        Raises:
            ValueError: If distance_threshold_percentile is not in [0.0, 100.0],
                min_chunk_size <= 0, max_chunk_size <= 0, or min_chunk_size > max_chunk_size.
        """
        if not (0.0 <= distance_threshold_percentile <= 100.0):
            raise ValueError("distance_threshold_percentile must be between 0.0 and 100.0")
        if min_chunk_size <= 0:
            raise ValueError("min_chunk_size must be positive")
        if max_chunk_size <= 0:
            raise ValueError("max_chunk_size must be positive")
        if min_chunk_size > max_chunk_size:
            raise ValueError("min_chunk_size cannot exceed max_chunk_size")

        self.embedding_client = embedding_client
        self.distance_threshold_percentile = distance_threshold_percentile
        self.min_chunk_size = min_chunk_size
        self.max_chunk_size = max_chunk_size

    @staticmethod
    def _cosine_distance(vec1: list[float], vec2: list[float]) -> float:
        """Compute cosine distance (1 - cosine_similarity).

        Args:
            vec1: First vector.
            vec2: Second vector.

        Returns:
            Cosine distance value in [0.0, 2.0].
        """
        dot = sum(a * b for a, b in zip(vec1, vec2))
        norm1 = math.sqrt(sum(a * a for a in vec1))
        norm2 = math.sqrt(sum(b * b for b in vec2))
        if norm1 == 0.0 or norm2 == 0.0:
            return 1.0
        sim = dot / (norm1 * norm2)
        sim = max(-1.0, min(1.0, sim))
        return max(0.0, 1.0 - sim)

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
        """Split page contents into semantic chunks based on sentence embedding distance.

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
        if not full_text.strip():
            return []

        sentences = self._split_into_sentences(full_text)
        if not sentences:
            return []

        if len(sentences) == 1:
            return [
                Chunk(
                    text=sentences[0],
                    document_id=document_id,
                    filename=filename,
                    page_numbers=self.find_pages_for_range(segments, 0, len(sentences[0])),
                    chunk_index=0,
                    metadata={"strategy": "semantic"},
                )
            ]

        # Embed sentences
        if self.embedding_client is None:
            from exocortex.config import get_settings
            from exocortex.embedding import EmbeddingClient

            self.embedding_client = EmbeddingClient(get_settings())

        embeddings = self.embedding_client.embed_documents(sentences)

        # Compute distances between consecutive sentences
        distances: list[float] = []
        for i in range(len(embeddings) - 1):
            dist = self._cosine_distance(embeddings[i], embeddings[i + 1])
            distances.append(dist)

        # Determine threshold
        sorted_dists = sorted(distances)
        idx = int(len(sorted_dists) * (self.distance_threshold_percentile / 100.0))
        idx = min(idx, len(sorted_dists) - 1)
        threshold = sorted_dists[idx]

        # Split into groups
        chunks: list[Chunk] = []
        current_group: list[str] = [sentences[0]]
        current_len = len(sentences[0])
        chunk_index = 0
        search_start = 0

        for i, dist in enumerate(distances):
            next_sent = sentences[i + 1]
            should_split = (dist >= threshold and current_len >= self.min_chunk_size) or (
                current_len + len(next_sent) > self.max_chunk_size
            )

            if should_split:
                chunk_text = " ".join(current_group).strip()
                start_pos = full_text.find(chunk_text[: min(50, len(chunk_text))], search_start)
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
                        metadata={"strategy": "semantic"},
                    )
                )
                chunk_index += 1
                current_group = [next_sent]
                current_len = len(next_sent)
            else:
                current_group.append(next_sent)
                current_len += len(next_sent) + 1

        if current_group:
            chunk_text = " ".join(current_group).strip()
            start_pos = full_text.find(chunk_text[: min(50, len(chunk_text))], search_start)
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
                    metadata={"strategy": "semantic"},
                )
            )

        return chunks
