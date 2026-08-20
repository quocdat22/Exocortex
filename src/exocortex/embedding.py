"""Embedding client using Ollama API.

Connects to a local Ollama instance to generate embeddings using the
qwen3-embedding:0.6b model. Supports both single and batch embedding.
"""

from __future__ import annotations

import logging

import httpx

from exocortex.config import Settings

logger = logging.getLogger(__name__)

# Instruction prefix for query embedding (improves retrieval accuracy)
QUERY_INSTRUCTION = (
    "Instruct: Given a web search query, retrieve relevant passages "
    "that answer the query\nQuery: "
)


class EmbeddingClient:
    """Client for generating embeddings via Ollama API.

    Args:
        settings: Application settings with Ollama configuration.
    """

    def __init__(self, settings: Settings) -> None:
        self.base_url = settings.ollama_base_url
        self.model = settings.embedding_model
        self.num_ctx = settings.embedding_num_ctx
        self.expected_dim = settings.embedding_dim

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """Embed a list of document texts (no instruction prefix).

        Args:
            texts: List of document/chunk texts to embed.

        Returns:
            List of embedding vectors (each is a list of floats).

        Raises:
            ConnectionError: If Ollama is not reachable.
            RuntimeError: If the embedding request fails.
        """
        if not texts:
            return []

        return self._call_embed_api(texts)

    def embed_query(self, query: str) -> list[float]:
        """Embed a single query text (with instruction prefix).

        Prepends the instruction prefix for better retrieval accuracy
        with qwen3-embedding's instruction-aware architecture.

        Args:
            query: The user's query text.

        Returns:
            A single embedding vector (list of floats).

        Raises:
            ConnectionError: If Ollama is not reachable.
            RuntimeError: If the embedding request fails.
        """
        prefixed_query = f"{QUERY_INSTRUCTION}{query}"
        embeddings = self._call_embed_api([prefixed_query])
        return embeddings[0]

    def _call_embed_api(self, texts: list[str]) -> list[list[float]]:
        """Call the Ollama /api/embed endpoint.

        Args:
            texts: List of texts to embed.

        Returns:
            List of embedding vectors.

        Raises:
            ConnectionError: If Ollama server is unreachable.
            RuntimeError: If the API returns an error.
        """
        url = f"{self.base_url}/api/embed"
        payload = {
            "model": self.model,
            "input": texts,
            "options": {
                "num_ctx": self.num_ctx,
            },
        }

        try:
            with httpx.Client(timeout=120.0) as client:
                response = client.post(url, json=payload)
        except httpx.ConnectError as e:
            raise ConnectionError(
                f"Cannot connect to Ollama at {self.base_url}. "
                f"Is Ollama running? Error: {e}"
            ) from e

        if response.status_code != 200:
            raise RuntimeError(
                f"Ollama embedding failed (HTTP {response.status_code}): "
                f"{response.text}"
            )

        data = response.json()
        embeddings = data.get("embeddings", [])

        if len(embeddings) != len(texts):
            raise RuntimeError(
                f"Expected {len(texts)} embeddings, got {len(embeddings)}"
            )

        # Validate dimension
        if embeddings and len(embeddings[0]) != self.expected_dim:
            logger.warning(
                f"Embedding dimension mismatch: expected {self.expected_dim}, "
                f"got {len(embeddings[0])}"
            )

        return embeddings

    def health_check(self) -> bool:
        """Check if Ollama is reachable and the embedding model is available.

        Returns:
            True if Ollama is healthy and model is available, False otherwise.
        """
        try:
            with httpx.Client(timeout=10.0) as client:
                # Check Ollama is running
                response = client.get(f"{self.base_url}/api/tags")
                if response.status_code != 200:
                    return False

                # Check model is available
                models = response.json().get("models", [])
                model_names = [m.get("name", "") for m in models]
                # Match with or without tag
                model_base = self.model.split(":")[0]
                return any(model_base in name for name in model_names)

        except (httpx.ConnectError, httpx.TimeoutException):
            return False
