"""Evaluation and benchmarking module for Exocortex."""

from exocortex.eval.dataset import (
    GoldenSample,
    load_golden_dataset,
    parse_pages_from_reference,
)
from exocortex.eval.metrics import (
    calculate_hit_rate_and_mrr,
    compute_chunk_statistics,
    parse_chunk_page_numbers,
)

__all__ = [
    "GoldenSample",
    "calculate_hit_rate_and_mrr",
    "compute_chunk_statistics",
    "load_golden_dataset",
    "parse_chunk_page_numbers",
    "parse_pages_from_reference",
]
