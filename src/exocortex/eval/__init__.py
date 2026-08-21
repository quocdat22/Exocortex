"""Evaluation and benchmarking module for Exocortex."""

from exocortex.eval.dataset import (
    GoldenSample,
    load_golden_dataset,
    parse_pages_from_reference,
)

__all__ = [
    "GoldenSample",
    "load_golden_dataset",
    "parse_pages_from_reference",
]
