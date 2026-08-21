"""Evaluation and benchmarking module for Exocortex."""

from exocortex.eval.benchmark import (
    format_markdown_report,
    run_benchmark,
)
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
from exocortex.eval.ragas_evaluator import (
    RagasEvaluator,
    evaluate_ragas_dataset,
)

__all__ = [
    "GoldenSample",
    "RagasEvaluator",
    "calculate_hit_rate_and_mrr",
    "compute_chunk_statistics",
    "evaluate_ragas_dataset",
    "format_markdown_report",
    "load_golden_dataset",
    "parse_chunk_page_numbers",
    "parse_pages_from_reference",
    "run_benchmark",
]
