"""Tests for Ragas evaluation module."""

from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from exocortex.config import Settings
from exocortex.eval.ragas_evaluator import RagasEvaluator, evaluate_ragas_dataset


def test_ragas_evaluator_init_default():
    settings = Settings(
        deepseek_api_key="test_deepseek_key",
        deepseek_base_url="https://api.deepseek.com",
        deepseek_model="deepseek-v4-flash",
        ollama_base_url="http://localhost:11434",
        embedding_model="qwen3-embedding:0.6b",
    )
    evaluator = RagasEvaluator(settings=settings)
    assert evaluator.llm is not None
    assert evaluator.embeddings is not None
    assert len(evaluator.metrics) == 4
    assert evaluator.settings.deepseek_model == "deepseek-v4-flash"
    assert evaluator.settings.embedding_model == "qwen3-embedding:0.6b"


def test_ragas_evaluator_init_without_explicit_settings():
    evaluator = RagasEvaluator()
    assert evaluator.settings is not None
    assert evaluator.llm is not None
    assert evaluator.embeddings is not None
    assert len(evaluator.metrics) == 4


def test_ragas_evaluator_evaluate_records_empty():
    evaluator = RagasEvaluator(settings=Settings(deepseek_api_key="test_key"))
    res = evaluator.evaluate_records([])
    assert res["context_precision"] == 0.0
    assert res["context_recall"] == 0.0
    assert res["faithfulness"] == 0.0
    assert res["answer_relevancy"] == 0.0
    assert isinstance(res["df"], pd.DataFrame)
    assert res["df"].empty


@patch("exocortex.eval.ragas_evaluator.evaluate")
def test_ragas_evaluator_evaluate_records_mocked(mock_evaluate):
    mock_df = pd.DataFrame(
        {
            "context_precision": [0.8, 0.9],
            "context_recall": [0.7, 0.9],
            "faithfulness": [0.85, 0.95],
            "answer_relevancy": [0.9, 0.9],
        }
    )
    mock_results = MagicMock()
    mock_results.to_pandas.return_value = mock_df
    mock_evaluate.return_value = mock_results

    settings = Settings(deepseek_api_key="test_key")
    evaluator = RagasEvaluator(settings=settings)

    records = [
        {
            "question": "What is ML?",
            "answer": "ML is Machine Learning.",
            "contexts": ["Machine learning is a field of AI."],
            "ground_truth": "Machine learning is a subfield of AI.",
        },
        {
            "question": "What is RAG?",
            "answer": "RAG is Retrieval-Augmented Generation.",
            "contexts": ["Retrieval-Augmented Generation enhances LLM responses."],
            "ground_truth": "Retrieval-Augmented Generation combines retrieval with generative models.",
        },
    ]

    res = evaluator.evaluate_records(records)

    assert mock_evaluate.called
    assert res["context_precision"] == pytest.approx(0.85)
    assert res["context_recall"] == pytest.approx(0.8)
    assert res["faithfulness"] == pytest.approx(0.9)
    assert res["answer_relevancy"] == pytest.approx(0.9)
    assert isinstance(res["df"], pd.DataFrame)
    assert len(res["df"]) == 2


@patch("exocortex.eval.ragas_evaluator.evaluate")
def test_ragas_evaluator_missing_metrics_columns(mock_evaluate):
    mock_df = pd.DataFrame(
        {
            "context_precision": [0.8, 0.9],
            # other metrics missing
        }
    )
    mock_results = MagicMock()
    mock_results.to_pandas.return_value = mock_df
    mock_evaluate.return_value = mock_results

    evaluator = RagasEvaluator(settings=Settings(deepseek_api_key="test_key"))
    records = [
        {
            "question": "Q1",
            "answer": "A1",
            "contexts": ["C1"],
            "ground_truth": "GT1",
        }
    ]
    res = evaluator.evaluate_records(records)
    assert res["context_precision"] == pytest.approx(0.85)
    assert res["context_recall"] == 0.0
    assert res["faithfulness"] == 0.0
    assert res["answer_relevancy"] == 0.0


@patch("exocortex.eval.ragas_evaluator.evaluate")
def test_evaluate_ragas_dataset_function(mock_evaluate):
    mock_df = pd.DataFrame(
        {
            "context_precision": [1.0],
            "context_recall": [1.0],
            "faithfulness": [1.0],
            "answer_relevancy": [1.0],
        }
    )
    mock_results = MagicMock()
    mock_results.to_pandas.return_value = mock_df
    mock_evaluate.return_value = mock_results

    settings = Settings(deepseek_api_key="test_key")
    records = [
        {
            "question": "Q",
            "answer": "A",
            "contexts": ["C"],
            "ground_truth": "GT",
        }
    ]
    res = evaluate_ragas_dataset(records, settings=settings)
    assert res["context_precision"] == 1.0
    assert res["faithfulness"] == 1.0
