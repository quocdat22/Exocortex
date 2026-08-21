"""Tests for the benchmark runner and report generator."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

from exocortex.eval.benchmark import format_markdown_report, run_benchmark
from exocortex.eval.dataset import GoldenSample
from exocortex.ingestion import PageContent
from exocortex.llm import LLMResponse
from exocortex.vectorstore import SearchResult


def test_format_markdown_report() -> None:
    """Test formatting of benchmark results into a Markdown report."""
    results = {
        "fixed": {
            "strategy": "fixed",
            "chunk_count": 50,
            "mean_chars": 500.0,
            "hit_rate_top_k": 0.85,
            "mrr": 0.75,
            "context_precision": 0.80,
            "context_recall": 0.82,
            "faithfulness": 0.90,
            "answer_relevancy": 0.88,
            "chunking_time_s": 0.1,
            "avg_retrieval_ms": 15.2,
        },
        "recursive": {
            "strategy": "recursive",
            "chunk_count": 45,
            "mean_chars": 520.0,
            "hit_rate_top_k": 0.90,
            "mrr": 0.80,
            "context_precision": 0.88,
            "context_recall": 0.89,
            "faithfulness": 0.92,
            "answer_relevancy": 0.91,
            "chunking_time_s": 0.12,
            "avg_retrieval_ms": 14.8,
        },
    }
    report = format_markdown_report(results)
    assert "# Chunking Strategies Benchmark Report" in report
    assert "fixed" in report
    assert "recursive" in report
    assert "0.850" in report
    assert "0.900" in report
    assert "Optimal Strategy:" in report
    assert "`recursive`" in report


def test_format_markdown_report_empty() -> None:
    """Test format_markdown_report with empty results dictionary."""
    report = format_markdown_report({})
    assert "# Chunking Strategies Benchmark Report" in report
    assert "No benchmark results available" in report


def test_format_markdown_report_best_strategy_scoring() -> None:
    """Verify that best_strategy accurately selects the strategy with the highest weighted score."""
    results = {
        "strat_a": {
            "strategy": "strat_a",
            "chunk_count": 10,
            "mean_chars": 100.0,
            "hit_rate_top_k": 0.5,
            "mrr": 0.5,
            "context_precision": 0.5,
            "context_recall": 0.5,
            "faithfulness": 0.5,
            "answer_relevancy": 0.5,
            "chunking_time_s": 0.1,
            "avg_retrieval_ms": 10.0,
        },
        "strat_b": {
            "strategy": "strat_b",
            "chunk_count": 10,
            "mean_chars": 100.0,
            "hit_rate_top_k": 0.95,
            "mrr": 0.9,
            "context_precision": 0.95,
            "context_recall": 0.95,
            "faithfulness": 0.95,
            "answer_relevancy": 0.95,
            "chunking_time_s": 0.1,
            "avg_retrieval_ms": 10.0,
        },
    }
    report = format_markdown_report(results)
    assert "`strat_b`" in report


@patch("exocortex.eval.benchmark.extract_text_from_pdf")
@patch("exocortex.eval.benchmark.load_golden_dataset")
@patch("exocortex.eval.benchmark.EmbeddingClient")
@patch("exocortex.eval.benchmark.LLMClient")
@patch("exocortex.eval.benchmark.RagasEvaluator")
@patch("exocortex.eval.benchmark.VectorStore")
def test_run_benchmark_mocked(
    mock_vectorstore_cls: MagicMock,
    mock_ragas_cls: MagicMock,
    mock_llm_cls: MagicMock,
    mock_emb_cls: MagicMock,
    mock_load_golden: MagicMock,
    mock_extract_pdf: MagicMock,
    tmp_path: Path,
) -> None:
    """Test full run_benchmark execution with mocked external dependencies."""
    # Setup sample PDF pages
    mock_extract_pdf.return_value = [
        PageContent(page_number=1, text="Machine learning system design introduction."),
        PageContent(page_number=2, text="Data engineering and feature pipelines in production."),
    ]

    # Setup sample Golden dataset
    mock_load_golden.return_value = [
        GoldenSample(
            entry_id=1,
            question="What is machine learning system design?",
            ground_truth="An iterative process for production-ready systems.",
            reference_pages=[1],
            reference_location="Page 1",
            excerpt_context="Machine learning system design introduction.",
        )
    ]

    # Setup mock embedding client
    mock_emb_instance = MagicMock()
    mock_emb_instance.embed_documents.return_value = [[0.1] * 1024]
    mock_emb_instance.embed_query.return_value = [0.1] * 1024
    mock_emb_cls.return_value = mock_emb_instance

    # Setup mock vectorstore
    mock_vs_instance = MagicMock()
    mock_vs_instance.query.return_value = [
        SearchResult(
            text="Machine learning system design introduction.",
            metadata={"page_numbers": "1", "filename": "test.pdf"},
            distance=0.05,
            chunk_id="chunk_1",
        )
    ]
    mock_vectorstore_cls.return_value = mock_vs_instance

    # Setup mock LLM client
    mock_llm_instance = MagicMock()
    mock_llm_instance.generate.return_value = LLMResponse(
        answer="Machine learning system design is an iterative process.",
        sources=[{"filename": "test.pdf", "page_numbers": "1", "text_preview": "..."}],
        model="deepseek-v4-flash",
    )
    mock_llm_cls.return_value = mock_llm_instance

    # Setup mock Ragas evaluator
    mock_ragas_instance = MagicMock()
    mock_ragas_instance.evaluate_records.return_value = {
        "context_precision": 0.88,
        "context_recall": 0.90,
        "faithfulness": 0.95,
        "answer_relevancy": 0.92,
    }
    mock_ragas_cls.return_value = mock_ragas_instance

    output_md = tmp_path / "benchmark_report.md"
    output_csv = tmp_path / "benchmark_details.csv"

    results = run_benchmark(
        pdf_path=Path("dummy.pdf"),
        golden_path=Path("dummy_golden.md"),
        output_md=output_md,
        output_csv=output_csv,
        strategies=["fixed"],
    )

    assert "fixed" in results
    fixed_res = results["fixed"]
    assert fixed_res["hit_rate_top_k"] == 1.0
    assert fixed_res["mrr"] == 1.0
    assert fixed_res["context_precision"] == 0.88
    assert fixed_res["context_recall"] == 0.90
    assert fixed_res["faithfulness"] == 0.95
    assert fixed_res["answer_relevancy"] == 0.92

    # Check files created
    assert output_md.exists()
    assert output_csv.exists()
    md_content = output_md.read_text(encoding="utf-8")
    assert "# Chunking Strategies Benchmark Report" in md_content
    assert "fixed" in md_content

    csv_content = output_csv.read_text(encoding="utf-8")
    assert "strategy,entry_id,question,hit,mrr,retrieval_ms" in csv_content
    assert "fixed,1," in csv_content


@patch("exocortex.eval.benchmark.run_benchmark")
def test_benchmark_main_cli(mock_run_benchmark: MagicMock) -> None:
    """Test CLI entry point with arguments."""
    from exocortex.eval.benchmark import main

    test_args = [
        "benchmark.py",
        "--pdf",
        "custom_pdf.pdf",
        "--golden",
        "custom_golden.md",
        "--output-md",
        "custom_report.md",
        "--output-csv",
        "custom_details.csv",
        "--strategies",
        "fixed",
        "recursive",
    ]
    with patch("sys.argv", test_args):
        main()

    mock_run_benchmark.assert_called_once_with(
        pdf_path=Path("custom_pdf.pdf"),
        golden_path=Path("custom_golden.md"),
        output_md=Path("custom_report.md"),
        output_csv=Path("custom_details.csv"),
        strategies=["fixed", "recursive"],
    )

