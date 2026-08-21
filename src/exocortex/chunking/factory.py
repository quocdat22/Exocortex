"""Chunking strategy factory."""

from __future__ import annotations

from typing import Any

from exocortex.chunking.base import BaseChunker
from exocortex.chunking.fixed_size import FixedSizeChunker
from exocortex.chunking.recursive import RecursiveCharacterChunker
from exocortex.chunking.semantic import SemanticChunker
from exocortex.chunking.sentence_paragraph import SentenceParagraphChunker


def get_chunker(
    strategy: str = "fixed",
    **kwargs: Any,
) -> BaseChunker:
    """Factory to retrieve a chunker instance by strategy name.

    Supported strategies:
    - "fixed" / "fixed_size": FixedSizeChunker
    - "recursive" / "recursive_character": RecursiveCharacterChunker
    - "sentence_paragraph" / "sentence" / "paragraph": SentenceParagraphChunker
    - "semantic": SemanticChunker

    Args:
        strategy: Strategy name or alias (case-insensitive).
        **kwargs: Strategy-specific configuration parameters:
            - chunk_size: int (fixed, recursive, sentence_paragraph)
            - chunk_overlap: int (fixed, recursive)
            - sentence_overlap: int (sentence_paragraph)
            - separators: list[str] | None (recursive)
            - embedding_client: EmbeddingClient | None (semantic)
            - distance_threshold_percentile: float (semantic)
            - min_chunk_size: int (semantic)
            - max_chunk_size: int (semantic)

    Returns:
        Configured BaseChunker instance.

    Raises:
        ValueError: If the strategy name is not recognized.
    """
    s = strategy.lower().strip()
    if s in ("fixed", "fixed_size"):
        chunk_size = kwargs.get("chunk_size", 512)
        chunk_overlap = kwargs.get("chunk_overlap", 50)
        return FixedSizeChunker(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    elif s in ("recursive", "recursive_character"):
        chunk_size = kwargs.get("chunk_size", 512)
        chunk_overlap = kwargs.get("chunk_overlap", 50)
        separators = kwargs.get("separators")
        return RecursiveCharacterChunker(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=separators,
        )
    elif s in ("sentence_paragraph", "sentence", "paragraph"):
        chunk_size = kwargs.get("chunk_size", 512)
        sentence_overlap = kwargs.get("sentence_overlap", 1)
        return SentenceParagraphChunker(
            chunk_size=chunk_size,
            sentence_overlap=sentence_overlap,
        )
    elif s == "semantic":
        return SemanticChunker(
            embedding_client=kwargs.get("embedding_client"),
            distance_threshold_percentile=kwargs.get(
                "distance_threshold_percentile", 85.0
            ),
            min_chunk_size=kwargs.get("min_chunk_size", 100),
            max_chunk_size=kwargs.get("max_chunk_size", 1000),
        )
    else:
        raise ValueError(
            f"Unknown chunking strategy: '{strategy}'. Supported: fixed, recursive, sentence_paragraph, semantic"
        )
