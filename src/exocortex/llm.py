"""LLM client for answer generation using DeepSeek API.

Uses the OpenAI-compatible API to send context-augmented prompts
to DeepSeek-v4-flash and generate grounded answers.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

from openai import OpenAI

from exocortex.config import Settings
from exocortex.vectorstore import SearchResult

logger = logging.getLogger(__name__)

# System prompt that instructs the LLM to stay grounded in context
SYSTEM_PROMPT = """You are a helpful assistant that answers questions based ONLY on the provided context from ebooks.

Rules:
1. Answer the question using ONLY the information in the context below.
2. If the context does not contain enough information to answer, say: "I don't have enough information in the provided documents to answer this question."
3. Cite your sources by mentioning the filename and page number(s) when possible.
4. Be concise and accurate. Do not add information beyond what is in the context.
5. If the question is ambiguous, ask for clarification.

Context from documents:
{context}
"""


def _format_context(results: list[SearchResult]) -> str:
    """Format search results into a context string for the LLM prompt.

    Args:
        results: List of SearchResult objects from vector search.

    Returns:
        Formatted context string with source attribution.
    """
    if not results:
        return "(No relevant documents found)"

    context_parts: list[str] = []
    for i, result in enumerate(results, 1):
        filename = result.metadata.get("filename", "unknown")
        pages = result.metadata.get("page_numbers", "?")
        context_parts.append(
            f"[Source {i}: {filename}, page(s) {pages}]\n{result.text}"
        )

    return "\n\n---\n\n".join(context_parts)


@dataclass
class LLMResponse:
    """Response from the LLM with answer and source information."""

    answer: str  # The generated answer text
    sources: list[dict]  # Source chunks used [{filename, page_numbers, text_preview}]
    model: str  # Model name used
    usage: dict | None = None  # Token usage stats if available


class LLMClient:
    """Client for generating answers using DeepSeek LLM.

    Uses the OpenAI Python client to connect to DeepSeek's
    OpenAI-compatible API endpoint.

    Args:
        settings: Application settings with DeepSeek configuration.
    """

    def __init__(self, settings: Settings) -> None:
        self.model = settings.deepseek_model
        self.temperature = settings.llm_temperature
        self.max_tokens = settings.llm_max_tokens

        self.client = OpenAI(
            api_key=settings.deepseek_api_key,
            base_url=settings.deepseek_base_url,
        )

    def generate(
        self,
        query: str,
        search_results: list[SearchResult],
    ) -> LLMResponse:
        """Generate an answer based on query and retrieved context.

        Args:
            query: The user's question.
            search_results: Retrieved chunks from the vector store.

        Returns:
            LLMResponse with the generated answer and source info.

        Raises:
            RuntimeError: If the LLM API call fails.
        """
        context = _format_context(search_results)
        system_message = SYSTEM_PROMPT.format(context=context)

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_message},
                    {"role": "user", "content": query},
                ],
                temperature=self.temperature,
                max_tokens=self.max_tokens,
            )
        except Exception as e:
            raise RuntimeError(f"LLM API call failed: {e}") from e

        answer = response.choices[0].message.content or ""

        # Extract source info from search results
        sources = []
        for result in search_results:
            sources.append(
                {
                    "filename": result.metadata.get("filename", "unknown"),
                    "page_numbers": result.metadata.get("page_numbers", "?"),
                    "text_preview": result.text[:200] + "..."
                    if len(result.text) > 200
                    else result.text,
                }
            )

        usage = None
        if response.usage:
            usage = {
                "prompt_tokens": response.usage.prompt_tokens,
                "completion_tokens": response.usage.completion_tokens,
                "total_tokens": response.usage.total_tokens,
            }

        logger.info(
            f"LLM generated answer: {len(answer)} chars, "
            f"sources: {len(sources)}, usage: {usage}"
        )

        return LLMResponse(
            answer=answer,
            sources=sources,
            model=self.model,
            usage=usage,
        )

    def health_check(self) -> bool:
        """Check if the DeepSeek API is reachable.

        Attempts a minimal API call to verify connectivity.

        Returns:
            True if API is reachable and responds, False otherwise.
        """
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": "ping"}],
                max_tokens=5,
                temperature=0,
            )
            return bool(response.choices)
        except Exception as e:
            logger.warning(f"LLM health check failed: {e}")
            return False
