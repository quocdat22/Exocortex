"""LLM client for answer generation using DeepSeek API.

Uses the OpenAI-compatible API to send context-augmented prompts
to DeepSeek-v4-flash and generate grounded answers.
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass

from openai import OpenAI

from exocortex.config import Settings
from exocortex.session import Message
from exocortex.vectorstore import SearchResult

logger = logging.getLogger(__name__)

# System prompt used when retrieval produces context chunks
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

# Conversational system prompt used when retrieval is bypassed or search results are empty
CONVERSATIONAL_SYSTEM_PROMPT = """You are a helpful assistant engaging in a conversation with the user.
Answer the user's question, request, or follow-up politely and accurately based on the ongoing conversation history.
If the user asks for new factual information or specific document excerpts that were not retrieved, politely state that no matching document sections were found."""

REWRITE_ROUTER_PROMPT = """You are an expert query analyzer and router for a conversational Retrieval-Augmented Generation (RAG) system answering questions from ebooks.

Given the Chat History and a new Follow-up Question from the user:

1. **Routing Decision (`needs_retrieval` - boolean)**:
   - Set `needs_retrieval: true` if the question requires looking up facts, definitions, book content, explanations, or deeper details from the ebook collection. This includes follow-up questions that ask for more details or elaboration on topics mentioned earlier (e.g., 'How is it different?', 'Tell me more about X', 'Why is that so?', 'Give examples', 'What is static data?').
   - Set `needs_retrieval: false` ONLY if the user question can be fully and accurately answered using the existing chat history alone, without searching the book. This includes:
     * Greetings / social pleasantries ('Hello', 'Thanks for the explanation', 'Good morning').
     * Requests to format, translate, simplify, or summarize what the assistant ALREADY stated in the conversation history ('Summarize your previous response into 2 bullet points', 'Translate what you just said to Vietnamese', 'Explain your last point in simpler terms').
     * Clarifications about the assistant's wording or meta-conversation ('Why did you say that?', 'Which of the points you mentioned was the first one?').

2. **Query Reformulation (`standalone_query` - string)**:
   - Rewrite the user's follow-up question into an explicit, search-optimized standalone question in English.
   - De-reference all pronouns ('it', 'they', 'this', 'that', 'its', 'the second one', 'these methods') and implicit topics by replacing them with the exact nouns, entities, and subjects from the chat history.
   - Example: If the chat discussed data in ML research vs production and the user asks 'How is it different?', rewrite to 'How does data differ between machine learning in research and production roles?'
   - If `needs_retrieval` is false, provide a clean standalone phrasing of the user's request.

Respond ONLY with a JSON object in this exact schema:
{
  "needs_retrieval": true,
  "standalone_query": "Explicit search query with all pronouns resolved"
}
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
        meta = result.metadata or {}
        filename = meta.get("filename", "unknown")
        pages = meta.get("page_numbers", "?")
        text = result.text or ""
        context_parts.append(
            f"[Source {i}: {filename}, page(s) {pages}]\n{text}"
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
            meta = result.metadata or {}
            text = result.text or ""
            sources.append(
                {
                    "filename": meta.get("filename", "unknown"),
                    "page_numbers": meta.get("page_numbers", "?"),
                    "text": text,
                    "text_preview": text[:200] + "..." if len(text) > 200 else text,
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

    def rewrite_and_route(
        self,
        history: list[Message],
        question: str,
    ) -> tuple[str, bool]:
        """Analyze history and rewrite follow-up question into a standalone query.

        Args:
            history: List of recent Message objects (chronological order).
            question: The user's newest follow-up question.

        Returns:
            Tuple of (standalone_query, needs_retrieval).
        """
        if not history:
            return question, True

        history_lines: list[str] = []
        for msg in history:
            history_lines.append(f"{msg.role.upper()}: {msg.content}")
        history_text = "\n".join(history_lines)

        user_content = f"Chat History:\n{history_text}\n\nNew Follow-up Question:\n{question}"

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": REWRITE_ROUTER_PROMPT},
                    {"role": "user", "content": user_content},
                ],
                response_format={"type": "json_object"},
                temperature=0.0,
                max_tokens=512,
            )
            raw_content = response.choices[0].message.content or "{}"
            # Clean markdown code block wraps if present
            cleaned = re.sub(r"^```(?:json)?\s*", "", raw_content.strip(), flags=re.IGNORECASE)
            cleaned = re.sub(r"\s*```$", "", cleaned)
            data = json.loads(cleaned)

            standalone = data.get("standalone_query", question).strip() or question
            needs_retrieval = bool(data.get("needs_retrieval", True))
            return standalone, needs_retrieval
        except Exception as e:
            logger.warning(f"Query rewrite and router failed ({e}), falling back to raw question: {question}")
            return question, True

    def generate_with_history(
        self,
        messages_history: list[Message],
        query: str,
        search_results: list[SearchResult],
    ) -> LLMResponse:
        """Generate an answer given conversation history and retrieved context chunks.

        Args:
            messages_history: Recent conversation Message items.
            query: The user's query.
            search_results: Retrieved chunks from the vector store.

        Returns:
            LLMResponse with answer, source info, and token usage.
        """
        if search_results:
            context = _format_context(search_results)
            system_message = SYSTEM_PROMPT.format(context=context)
        else:
            system_message = CONVERSATIONAL_SYSTEM_PROMPT

        messages_payload: list[dict[str, str]] = [
            {"role": "system", "content": system_message}
        ]

        for msg in messages_history:
            if msg.role in ("user", "assistant"):
                messages_payload.append({"role": msg.role, "content": msg.content})

        messages_payload.append({"role": "user", "content": query})

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages_payload,
                temperature=self.temperature,
                max_tokens=self.max_tokens,
            )
        except Exception as e:
            raise RuntimeError(f"LLM API call failed: {e}") from e

        answer = response.choices[0].message.content or ""

        sources = []
        for result in search_results:
            meta = result.metadata or {}
            text = result.text or ""
            sources.append(
                {
                    "filename": meta.get("filename", "unknown"),
                    "page_numbers": meta.get("page_numbers", "?"),
                    "text": text,
                    "text_preview": text[:200] + "..." if len(text) > 200 else text,
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
            f"LLM generated chat answer: {len(answer)} chars, sources: {len(sources)}, usage: {usage}"
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
