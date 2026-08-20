# Phase 4: Retrieval + LLM Generation

## Objective

Build the retrieval engine that takes a user query, finds the most relevant chunks from
ChromaDB, and sends them as context to DeepSeek LLM to generate a grounded answer with
source citations.

## Prerequisites

- Phase 3 completed (embedding client and vector store working)
- DeepSeek API key configured in `.env`
- At least one document indexed in ChromaDB (from Phase 2+3 pipeline)

---

## Concepts

### RAG Query Flow

```
User Query
    │
    ▼
┌─────────────────┐
│ Embed Query      │  ← Ollama (with instruction prefix)
│ (qwen3-embedding)│
└────────┬────────┘
         │ query vector
         ▼
┌─────────────────┐
│ Search ChromaDB  │  ← cosine similarity, top_k=5
│ (vector search)  │
└────────┬────────┘
         │ top-K chunks + metadata
         ▼
┌─────────────────┐
│ Build Prompt     │  ← system prompt + context chunks + user query
│ (prompt template)│
└────────┬────────┘
         │ messages
         ▼
┌─────────────────┐
│ Call DeepSeek    │  ← OpenAI-compatible API, temp=0.1
│ LLM (API)       │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ Format Response  │  ← answer + source chunks
└─────────────────┘
```

### Prompt Engineering

The system prompt instructs the LLM to:
1. Answer ONLY based on the provided context
2. Cite which source (filename, page) the answer comes from
3. Say "I don't have enough information" if context doesn't contain the answer
4. Use low temperature (0.1) for factual, deterministic responses

---

## Implementation

### File: `src/exocortex/llm.py`

```python
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
```

### File: `src/exocortex/retrieval.py`

```python
"""Retrieval engine — orchestrates query → embed → search → LLM answer.

This is the main entry point for the RAG pipeline. It ties together
the embedding client, vector store, and LLM client to answer user queries.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

from exocortex.config import Settings
from exocortex.embedding import EmbeddingClient
from exocortex.llm import LLMClient, LLMResponse
from exocortex.vectorstore import SearchResult, VectorStore

logger = logging.getLogger(__name__)


@dataclass
class QueryResponse:
    """Complete response to a user query."""

    answer: str  # LLM-generated answer
    sources: list[dict]  # Source documents with metadata
    query: str  # Original query
    num_chunks_retrieved: int  # How many chunks were retrieved
    model: str  # LLM model used
    usage: dict | None = None  # Token usage stats


class RAGEngine:
    """Retrieval-Augmented Generation engine.

    Orchestrates the full RAG pipeline:
    1. Embed the user query (via Ollama)
    2. Search ChromaDB for relevant chunks
    3. Send context + query to DeepSeek LLM
    4. Return structured response with sources

    Args:
        settings: Application settings.
        embedding_client: Pre-initialized embedding client (optional).
        vector_store: Pre-initialized vector store (optional).
        llm_client: Pre-initialized LLM client (optional).
    """

    def __init__(
        self,
        settings: Settings,
        embedding_client: EmbeddingClient | None = None,
        vector_store: VectorStore | None = None,
        llm_client: LLMClient | None = None,
    ) -> None:
        self.settings = settings
        self.embedding_client = embedding_client or EmbeddingClient(settings)
        self.vector_store = vector_store or VectorStore(settings)
        self.llm_client = llm_client or LLMClient(settings)

    def query(self, question: str) -> QueryResponse:
        """Process a user question through the full RAG pipeline.

        Args:
            question: The user's question in English.

        Returns:
            QueryResponse with the answer and source information.

        Raises:
            ConnectionError: If Ollama is not reachable.
            RuntimeError: If any pipeline step fails.
        """
        logger.info(f"Processing query: {question[:100]}...")

        # Step 1: Embed the query
        query_embedding = self.embedding_client.embed_query(question)
        logger.info(f"Query embedded (dim={len(query_embedding)})")

        # Step 2: Search vector store
        search_results = self.vector_store.query(
            query_embedding=query_embedding,
            top_k=self.settings.top_k,
        )
        logger.info(f"Retrieved {len(search_results)} chunks")

        # Step 3: Generate answer with LLM
        llm_response = self.llm_client.generate(
            query=question,
            search_results=search_results,
        )
        logger.info(f"LLM generated answer ({len(llm_response.answer)} chars)")

        return QueryResponse(
            answer=llm_response.answer,
            sources=llm_response.sources,
            query=question,
            num_chunks_retrieved=len(search_results),
            model=llm_response.model,
            usage=llm_response.usage,
        )

    def ingest_and_index(self, pdf_path: str | object) -> dict:
        """Convenience method: ingest a PDF and index it in the vector store.

        Combines Phase 2 (ingestion) and Phase 3 (embedding + storage).

        Args:
            pdf_path: Path to the PDF file.

        Returns:
            Dict with ingestion results: {filename, document_id, chunk_count}.
        """
        from pathlib import Path

        from exocortex.ingestion import ingest_pdf

        path = Path(pdf_path)
        chunks = ingest_pdf(
            pdf_path=path,
            chunk_size=self.settings.chunk_size,
            chunk_overlap=self.settings.chunk_overlap,
        )

        if not chunks:
            return {
                "filename": path.name,
                "document_id": "",
                "chunk_count": 0,
            }

        # Embed all chunks
        texts = [c.text for c in chunks]
        embeddings = self.embedding_client.embed_documents(texts)

        # Store in vector store
        self.vector_store.add_chunks(chunks, embeddings)

        logger.info(f"Ingested {path.name}: {len(chunks)} chunks embedded and stored")

        return {
            "filename": path.name,
            "document_id": chunks[0].document_id,
            "chunk_count": len(chunks),
        }
```

---

## Tests

### File: `tests/test_llm.py`

```python
"""Tests for Phase 4: LLM client."""

import logging

import pytest

from exocortex.config import Settings
from exocortex.llm import LLMClient, LLMResponse, _format_context, SYSTEM_PROMPT
from exocortex.vectorstore import SearchResult

logger = logging.getLogger(__name__)


@pytest.fixture
def settings() -> Settings:
    return Settings(deepseek_api_key="test-key")


@pytest.fixture
def sample_results() -> list[SearchResult]:
    return [
        SearchResult(
            text="Machine learning is a subset of artificial intelligence.",
            metadata={"filename": "ml_book.pdf", "page_numbers": "1,2"},
            distance=0.1,
            chunk_id="doc1_0",
        ),
        SearchResult(
            text="Deep learning uses neural networks.",
            metadata={"filename": "ml_book.pdf", "page_numbers": "5"},
            distance=0.2,
            chunk_id="doc1_1",
        ),
    ]


def test_format_context(sample_results):
    """Context should include source attribution."""
    context = _format_context(sample_results)

    assert "Source 1: ml_book.pdf" in context
    assert "Source 2: ml_book.pdf" in context
    assert "Machine learning" in context
    assert "Deep learning" in context


def test_format_context_empty():
    """Empty results should produce a 'no documents' message."""
    context = _format_context([])
    assert "No relevant documents found" in context


def test_system_prompt_has_placeholder():
    """System prompt should have {context} placeholder."""
    assert "{context}" in SYSTEM_PROMPT


def test_llm_response_dataclass():
    """LLMResponse should hold all fields."""
    response = LLMResponse(
        answer="Test answer",
        sources=[{"filename": "test.pdf", "page_numbers": "1"}],
        model="deepseek-v4-flash",
        usage={"total_tokens": 100},
    )
    assert response.answer == "Test answer"
    assert len(response.sources) == 1


# --- Integration tests (require DeepSeek API key) ---


@pytest.fixture
def llm_client(settings) -> LLMClient:
    """Create LLM client — skips if no real API key."""
    if not settings.deepseek_api_key or settings.deepseek_api_key == "test-key":
        pytest.skip("No DeepSeek API key configured")
    return LLMClient(settings)


def test_llm_generate_integration(llm_client, sample_results):
    """Should generate an answer based on context."""
    response = llm_client.generate(
        query="What is machine learning?",
        search_results=sample_results,
    )

    assert isinstance(response, LLMResponse)
    assert len(response.answer) > 0
    assert len(response.sources) == 2
    logger.info(f"LLM answer: {response.answer[:200]}")
    logger.info(f"Token usage: {response.usage}")


def test_llm_health_check_integration(llm_client):
    """Health check should return True with valid API key."""
    result = llm_client.health_check()
    assert result is True
    logger.info("LLM health check: OK")
```

### File: `tests/test_retrieval.py`

```python
"""Tests for Phase 4: RAG retrieval engine."""

import logging

import pytest

from exocortex.config import Settings
from exocortex.retrieval import RAGEngine, QueryResponse

logger = logging.getLogger(__name__)


@pytest.fixture
def settings() -> Settings:
    """Settings — tests requiring real services will check availability."""
    return Settings(deepseek_api_key="test-key")


def test_query_response_dataclass():
    """QueryResponse should hold all required fields."""
    response = QueryResponse(
        answer="Test answer",
        sources=[],
        query="test question",
        num_chunks_retrieved=0,
        model="deepseek-v4-flash",
    )
    assert response.answer == "Test answer"
    assert response.query == "test question"


def test_rag_engine_initialization(settings):
    """RAGEngine should initialize without errors."""
    # Note: this will try to init ChromaDB and other clients
    # but won't make API calls
    engine = RAGEngine(settings)
    assert engine.settings == settings


# --- Full Integration Test ---
# This test requires:
# 1. Ollama running with qwen3-embedding:0.6b
# 2. DeepSeek API key configured
# 3. At least one document indexed in ChromaDB


@pytest.fixture
def full_engine() -> RAGEngine:
    """Create a fully configured RAG engine."""
    settings = Settings()  # Load from .env

    if not settings.deepseek_api_key or settings.deepseek_api_key == "test-key":
        pytest.skip("No DeepSeek API key configured")

    engine = RAGEngine(settings)

    if not engine.embedding_client.health_check():
        pytest.skip("Ollama not available")

    if engine.vector_store.count() == 0:
        pytest.skip("No documents indexed — run ingestion first")

    return engine


def test_rag_query_integration(full_engine):
    """Full RAG pipeline: query → embed → search → LLM → answer."""
    response = full_engine.query("What is this book about?")

    assert isinstance(response, QueryResponse)
    assert len(response.answer) > 0
    assert response.num_chunks_retrieved > 0
    assert len(response.sources) > 0

    logger.info(f"Query: {response.query}")
    logger.info(f"Chunks retrieved: {response.num_chunks_retrieved}")
    logger.info(f"Answer: {response.answer[:300]}")
    logger.info(f"Sources: {response.sources}")
```

---

## Success Criteria

Phase 4 is complete when ALL of the following are true:

### Automated Tests
```bash
# Unit tests (no external services required)
uv run pytest tests/test_llm.py tests/test_retrieval.py -v -k "not integration" --log-cli-level=INFO

# Integration tests (require Ollama + DeepSeek API key + indexed docs)
uv run pytest tests/test_llm.py tests/test_retrieval.py -v -k "integration" --log-cli-level=INFO
```

**Expected:** Unit tests PASSED. Integration tests PASSED if services available, SKIPPED otherwise.

### Manual Verification (End-to-End)
```bash
# Step 1: Ingest a PDF
uv run python -c "
from exocortex.config import get_settings
from exocortex.retrieval import RAGEngine

engine = RAGEngine(get_settings())
result = engine.ingest_and_index('data/ebooks/YOUR_BOOK.pdf')
print(f'Ingested: {result}')
"

# Step 2: Query the ingested content
uv run python -c "
from exocortex.config import get_settings
from exocortex.retrieval import RAGEngine

engine = RAGEngine(get_settings())
response = engine.query('What is the main topic of this book?')
print(f'Answer: {response.answer}')
print(f'Sources: {response.sources}')
print(f'Chunks used: {response.num_chunks_retrieved}')
"
```

**Expected:** The LLM returns an answer that is clearly derived from the ebook content, with source citations.

### Checklist
- [x] `src/exocortex/llm.py` exists with `LLMClient` class
- [x] `src/exocortex/retrieval.py` exists with `RAGEngine` class
- [x] System prompt enforces context-only answers
- [x] Query embedding uses instruction prefix
- [x] LLM response includes source attribution
- [x] `RAGEngine.query()` returns `QueryResponse` with answer + sources
- [x] `RAGEngine.ingest_and_index()` combines ingestion + embedding + storage
- [x] All unit tests pass
- [x] Full pipeline works end-to-end (manual test)
