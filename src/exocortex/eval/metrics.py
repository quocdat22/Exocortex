"""Traditional retrieval and operational evaluation metrics."""

from __future__ import annotations

import statistics
from typing import Any

from exocortex.ingestion import Chunk
from exocortex.vectorstore import SearchResult


def parse_chunk_page_numbers(metadata: dict) -> list[int]:
    """Parse comma-separated page numbers string or list from chunk metadata."""
    raw = metadata.get("page_numbers", "")
    if isinstance(raw, list):
        return [int(x) for x in raw]
    if not raw or raw == "?":
        return []
    try:
        return [int(p.strip()) for p in str(raw).split(",") if p.strip().isdigit()]
    except (ValueError, TypeError, AttributeError):
        return []


def calculate_hit_rate_and_mrr(
    search_results: list[SearchResult],
    reference_pages: list[int],
    top_k: int = 5,
) -> tuple[float, float]:
    """Calculate Hit@K (0 or 1) and Reciprocal Rank (1/rank) for a single query."""
    if not search_results or not reference_pages:
        return 0.0, 0.0

    ref_set = set(reference_pages)
    hit = 0.0
    reciprocal_rank = 0.0

    for rank, res in enumerate(search_results[:top_k], start=1):
        chunk_pages = parse_chunk_page_numbers(res.metadata)
        if any(p in ref_set for p in chunk_pages):
            hit = 1.0
            reciprocal_rank = 1.0 / rank
            break

    return hit, reciprocal_rank


def compute_chunk_statistics(chunks: list[Chunk]) -> dict[str, Any]:
    """Compute character length and count statistics across chunks."""
    if not chunks:
        return {
            "chunk_count": 0,
            "mean_chars": 0.0,
            "median_chars": 0.0,
            "min_chars": 0,
            "max_chars": 0,
        }

    lengths = [len(c.text) for c in chunks]
    return {
        "chunk_count": len(chunks),
        "mean_chars": round(statistics.mean(lengths), 2),
        "median_chars": round(statistics.median(lengths), 2),
        "min_chars": min(lengths),
        "max_chars": max(lengths),
    }
