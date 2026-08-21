"""Ragas evaluator module with DeepSeek LLM and Ollama Embeddings integration."""

from __future__ import annotations

import logging
from typing import Any

import pandas as pd
from datasets import Dataset
from langchain_community.embeddings import OllamaEmbeddings
from langchain_openai import ChatOpenAI
from ragas import evaluate
from ragas.metrics import (
    answer_relevancy,
    context_precision,
    context_recall,
    faithfulness,
)

from exocortex.config import Settings, get_settings

logger = logging.getLogger(__name__)


class RagasEvaluator:
    """Evaluates RAG outputs against Golden Dataset using Ragas metrics."""

    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or get_settings()

        # Configure DeepSeek as Ragas Evaluator LLM
        self.llm = ChatOpenAI(
            model=self.settings.deepseek_model,
            api_key=self.settings.deepseek_api_key,
            base_url=self.settings.deepseek_base_url,
            temperature=0.0,
        )

        # Configure Ollama as Ragas Evaluator Embeddings
        self.embeddings = OllamaEmbeddings(
            model=self.settings.embedding_model,
            base_url=self.settings.ollama_base_url,
        )

        self.metrics = [
            context_precision,
            context_recall,
            faithfulness,
            answer_relevancy,
        ]

    def evaluate_records(
        self,
        records: list[dict[str, Any]],
    ) -> dict[str, Any]:
        """Run Ragas evaluation on a list of evaluation records.

        Each record contains:
        - 'question': str
        - 'answer': str
        - 'contexts': list[str]
        - 'ground_truth': str
        """
        if not records:
            return {
                "context_precision": 0.0,
                "context_recall": 0.0,
                "faithfulness": 0.0,
                "answer_relevancy": 0.0,
                "df": pd.DataFrame(),
            }

        df_input = pd.DataFrame(records)
        dataset = Dataset.from_pandas(df_input)

        logger.info(f"Running Ragas evaluation on {len(records)} records...")
        results = evaluate(
            dataset=dataset,
            metrics=self.metrics,
            llm=self.llm,
            embeddings=self.embeddings,
        )

        df_result = results.to_pandas()
        return {
            "context_precision": float(df_result["context_precision"].mean())
            if "context_precision" in df_result
            and not pd.isna(df_result["context_precision"].mean())
            else 0.0,
            "context_recall": float(df_result["context_recall"].mean())
            if "context_recall" in df_result
            and not pd.isna(df_result["context_recall"].mean())
            else 0.0,
            "faithfulness": float(df_result["faithfulness"].mean())
            if "faithfulness" in df_result
            and not pd.isna(df_result["faithfulness"].mean())
            else 0.0,
            "answer_relevancy": float(df_result["answer_relevancy"].mean())
            if "answer_relevancy" in df_result
            and not pd.isna(df_result["answer_relevancy"].mean())
            else 0.0,
            "df": df_result,
        }


def evaluate_ragas_dataset(
    records: list[dict[str, Any]],
    settings: Settings | None = None,
) -> dict[str, Any]:
    """Evaluate a dataset using RagasEvaluator."""
    evaluator = RagasEvaluator(settings=settings)
    return evaluator.evaluate_records(records)
