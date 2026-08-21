"""Chunking strategies and factory for Exocortex."""

from exocortex.chunking.base import BaseChunker
from exocortex.chunking.factory import get_chunker
from exocortex.chunking.fixed_size import FixedSizeChunker
from exocortex.chunking.recursive import RecursiveCharacterChunker
from exocortex.chunking.semantic import SemanticChunker
from exocortex.chunking.sentence_paragraph import SentenceParagraphChunker

__all__ = [
    "BaseChunker",
    "FixedSizeChunker",
    "RecursiveCharacterChunker",
    "SemanticChunker",
    "SentenceParagraphChunker",
    "get_chunker",
]
