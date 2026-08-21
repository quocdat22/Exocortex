"""Benchmark runner and Markdown/CSV report generator."""

from __future__ import annotations

import argparse
import logging
import time
from pathlib import Path
from typing import Any

import pandas as pd
from tabulate import tabulate

from exocortex.chunking import get_chunker
from exocortex.config import get_settings
from exocortex.embedding import EmbeddingClient
from exocortex.eval.dataset import load_golden_dataset
from exocortex.eval.metrics import calculate_hit_rate_and_mrr, compute_chunk_statistics
from exocortex.eval.ragas_evaluator import RagasEvaluator
from exocortex.ingestion import extract_text_from_pdf, generate_document_id
from exocortex.llm import LLMClient
from exocortex.vectorstore import VectorStore

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

STRATEGIES = ["fixed", "recursive", "sentence_paragraph", "semantic"]


def format_markdown_report(benchmark_results: dict[str, dict[str, Any]]) -> str:
    """Format benchmark results dict into a comprehensive Markdown report."""
    if not benchmark_results:
        return (
            "# Chunking Strategies Benchmark Report\n\n"
            "## 1. Summary of Benchmark Results\n\n"
            "No benchmark results available.\n\n"
            "## 2. Recommendation\n\n"
            "No benchmark results available.\n\n"
            f"Generated at: {time.strftime('%Y-%m-%d %H:%M:%S')}\n"
        )

    rows = []
    for strat, data in benchmark_results.items():
        rows.append([
            strat,
            data.get("chunk_count", 0),
            f"{data.get('mean_chars', 0.0):.0f}",
            f"{data.get('hit_rate_top_k', 0.0):.3f}",
            f"{data.get('mrr', 0.0):.3f}",
            f"{data.get('context_precision', 0.0):.3f}",
            f"{data.get('context_recall', 0.0):.3f}",
            f"{data.get('faithfulness', 0.0):.3f}",
            f"{data.get('answer_relevancy', 0.0):.3f}",
            f"{data.get('avg_retrieval_ms', 0.0):.1f} ms",
        ])

    headers = [
        "Strategy",
        "Chunks",
        "Avg Len",
        "HitRate@K",
        "MRR",
        "Context Precision",
        "Context Recall",
        "Faithfulness",
        "Answer Relevancy",
        "Latency",
    ]
    table_md = tabulate(rows, headers=headers, tablefmt="github", floatfmt=".3f")

    # Find winning strategy based on combined Ragas + Retrieval score
    best_strategy = max(
        benchmark_results.keys(),
        key=lambda s: (
            benchmark_results[s].get("context_recall", 0.0) * 0.3
            + benchmark_results[s].get("context_precision", 0.0) * 0.3
            + benchmark_results[s].get("faithfulness", 0.0) * 0.2
            + benchmark_results[s].get("hit_rate_top_k", 0.0) * 0.2
        ),
    )

    report = f"""# Chunking Strategies Benchmark Report

## 1. Summary of Benchmark Results

{table_md}

## 2. Recommendation

Based on the multi-dimensional evaluation (Ragas Context Precision/Recall, Faithfulness, Hit Rate@K, and MRR):
- **Optimal Strategy:** `{best_strategy}`
- **Recommendation:** Set `CHUNKING_STRATEGY={best_strategy}` in `src/exocortex/config.py`.

Generated at: {time.strftime('%Y-%m-%d %H:%M:%S')}
"""
    return report


def run_benchmark(
    pdf_path: Path
    | str = Path(
        "data/ebooks/Designing Machine Learning Systems An Iterative Process for Production-Ready (Chip Huyen)[21-43].pdf"
    ),
    golden_path: Path | str = Path("data/GoldenDatset.md"),
    output_md: Path | str | None = Path("docs/benchmark_results.md"),
    output_csv: Path | str | None = Path("data/benchmark_results.csv"),
    strategies: list[str] | None = None,
) -> dict[str, Any]:
    """Execute end-to-end benchmark across all chunking strategies."""
    strategies = strategies or STRATEGIES
    settings = get_settings()
    pdf_path = Path(pdf_path)
    golden_path = Path(golden_path)
    output_md = Path(output_md) if output_md else None
    output_csv = Path(output_csv) if output_csv else None

    golden_samples = load_golden_dataset(golden_path)
    pages = extract_text_from_pdf(pdf_path)
    document_id = generate_document_id(pdf_path.name)

    embedding_client = EmbeddingClient(settings)
    llm_client = LLMClient(settings)
    ragas_eval = RagasEvaluator(settings)

    all_results = {}
    detailed_rows = []

    for strat in strategies:
        logger.info(
            f"\n==================== Benchmarking Strategy: {strat} ===================="
        )
        chunker = get_chunker(strat, embedding_client=embedding_client)

        t0 = time.perf_counter()
        chunks = chunker.chunk(pages, document_id=document_id, filename=pdf_path.name)
        chunking_time = time.perf_counter() - t0

        stats = compute_chunk_statistics(chunks)
        logger.info(
            f"Created {len(chunks)} chunks in {chunking_time:.2f}s (mean length: {stats['mean_chars']} chars)"
        )

        # Create isolated collection for this strategy
        coll_name = f"benchmark_{strat}_{int(time.time() * 1000)}"
        settings_copy = settings.model_copy(
            update={"chroma_collection_name": coll_name}
        )
        vector_store = VectorStore(settings_copy)

        texts = [c.text for c in chunks]
        embeddings = embedding_client.embed_documents(texts)
        vector_store.add_chunks(chunks, embeddings)

        hits, mrrs, retrieval_times = [], [], []
        ragas_records = []

        for sample in golden_samples:
            # Step 1: Embed query & retrieve
            t_ret_0 = time.perf_counter()
            q_emb = embedding_client.embed_query(sample.question)
            results = vector_store.query(q_emb, top_k=settings.top_k)
            ret_duration = (time.perf_counter() - t_ret_0) * 1000
            retrieval_times.append(ret_duration)

            hit, mrr = calculate_hit_rate_and_mrr(
                results, sample.reference_pages, top_k=settings.top_k
            )
            hits.append(hit)
            mrrs.append(mrr)

            # Step 2: Generate answer with LLM
            llm_resp = llm_client.generate(
                query=sample.question, search_results=results
            )

            ragas_records.append(
                {
                    "question": sample.question,
                    "answer": llm_resp.answer,
                    "contexts": [r.text for r in results],
                    "ground_truth": sample.ground_truth,
                }
            )

            detailed_rows.append(
                {
                    "strategy": strat,
                    "entry_id": sample.entry_id,
                    "question": sample.question,
                    "hit": hit,
                    "mrr": mrr,
                    "retrieval_ms": ret_duration,
                }
            )

        # Run Ragas evaluation
        ragas_scores = ragas_eval.evaluate_records(ragas_records)

        all_results[strat] = {
            "strategy": strat,
            "chunk_count": stats["chunk_count"],
            "mean_chars": stats["mean_chars"],
            "hit_rate_top_k": sum(hits) / len(hits) if hits else 0.0,
            "mrr": sum(mrrs) / len(mrrs) if mrrs else 0.0,
            "context_precision": ragas_scores.get("context_precision", 0.0),
            "context_recall": ragas_scores.get("context_recall", 0.0),
            "faithfulness": ragas_scores.get("faithfulness", 0.0),
            "answer_relevancy": ragas_scores.get("answer_relevancy", 0.0),
            "chunking_time_s": chunking_time,
            "avg_retrieval_ms": sum(retrieval_times) / len(retrieval_times)
            if retrieval_times
            else 0.0,
        }

        try:
            vector_store.client.delete_collection(coll_name)
        except Exception as e:  # noqa: BLE001
            logger.debug("Failed to delete benchmark collection %s: %s", coll_name, e)

    # Export outputs
    report_md = format_markdown_report(all_results)
    if output_md:
        output_md.parent.mkdir(parents=True, exist_ok=True)
        output_md.write_text(report_md, encoding="utf-8")

    if output_csv:
        df_details = pd.DataFrame(detailed_rows)
        output_csv.parent.mkdir(parents=True, exist_ok=True)
        df_details.to_csv(output_csv, index=False)

    print("\n" + report_md)
    return all_results


def main() -> None:
    """CLI entry point for running benchmarks."""
    parser = argparse.ArgumentParser(description="Run RAG Chunking Benchmark")
    parser.add_argument(
        "--pdf",
        type=Path,
        default=Path(
            "data/ebooks/Designing Machine Learning Systems An Iterative Process for Production-Ready (Chip Huyen)[21-43].pdf"
        ),
        help="Path to evaluation PDF",
    )
    parser.add_argument(
        "--golden",
        type=Path,
        default=Path("data/GoldenDatset.md"),
        help="Path to Golden Dataset markdown file",
    )
    parser.add_argument(
        "--output-md",
        type=Path,
        default=Path("docs/benchmark_results.md"),
        help="Output markdown report path",
    )
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=Path("data/benchmark_results.csv"),
        help="Output CSV details path",
    )
    parser.add_argument(
        "--strategies",
        nargs="+",
        default=None,
        help="List of chunking strategies to evaluate (e.g. fixed recursive semantic)",
    )
    args = parser.parse_args()
    run_benchmark(
        pdf_path=args.pdf,
        golden_path=args.golden,
        output_md=args.output_md,
        output_csv=args.output_csv,
        strategies=args.strategies,
    )


if __name__ == "__main__":
    main()
