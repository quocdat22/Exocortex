# Chunking Strategies & Ragas Benchmarking Framework Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement 4 modular chunking strategies (Fixed-size, Recursive Character, Sentence/Paragraph-aware, Semantic), build a Ragas + traditional retrieval evaluation framework on a Golden Dataset, and benchmark all strategies to determine the optimal chunker for Exocortex.

**Architecture:** Refactor chunking logic into `src/exocortex/chunking/` following the Strategy Pattern with `BaseChunker` and `get_chunker` factory. Build an evaluation suite under `src/exocortex/eval/` that parses `data/GoldenDatset.md`, evaluates retrieval and generation with Ragas and custom metrics (Hit Rate@K, MRR, latency, token distributions), and generates structured benchmark reports.

**Tech Stack:** Python 3.12, PyMuPDF (fitz), ChromaDB, Ollama (`qwen3-embedding:0.6b`), DeepSeek API (`deepseek-v4-flash`), Ragas, LangChain, Pandas, Tabulate, Pytest.

**Spec:** [`docs/superpowers/specs/2026-08-21-chunking-benchmark-design.md`](file:///home/rookie/projects/Exocortex/docs/superpowers/specs/2026-08-21-chunking-benchmark-design.md)

## Global Constraints
- All chunkers must produce `list[Chunk]` with accurate 1-indexed `page_numbers`, deterministic `document_id`, and incremental `chunk_index`.
- Backward compatibility: `ingest_pdf()` in `src/exocortex/ingestion.py` must retain its default interface while supporting strategy selection.
- All dependencies must be managed via `pyproject.toml` and synchronized using `uv`.
- All tests must pass with `uv run pytest`.

---

### Task 1: Dependencies Setup for Evaluation & Benchmark

**Files:**
- Modify: `pyproject.toml:7-31`
- Test: `tests/test_deps.py`

**Interfaces:**
- Consumes: `pyproject.toml`
- Produces: Installed dependencies (`ragas`, `langchain`, `langchain-openai`, `langchain-community`, `pandas`, `tabulate`)

- [ ] **Step 1: Write test to verify dependencies can be imported**

```python
# tests/test_deps.py
def test_evaluation_dependencies_importable():
    import langchain
    import langchain_community
    import langchain_openai
    import pandas
    import ragas
    import tabulate

    assert ragas is not None
    assert langchain is not None
    assert pandas is not None
```

- [ ] **Step 2: Run test to verify it fails (missing packages)**

Run: `uv run pytest tests/test_deps.py -v`  
Expected: FAIL with `ModuleNotFoundError`

- [ ] **Step 3: Update `pyproject.toml` with dependencies and sync**

Update `pyproject.toml`:
```toml
dependencies = [
    "chromadb>=1.0.0",
    "fastapi>=0.115.0",
    "httpx>=0.27.0",
    "langchain>=0.3.0",
    "langchain-community>=0.3.0",
    "langchain-openai>=0.2.0",
    "openai>=1.40.0",
    "pandas>=2.2.0",
    "pydantic>=2.9.0",
    "pydantic-settings>=2.5.0",
    "pymupdf>=1.24.0",
    "python-dotenv>=1.0.0",
    "python-multipart>=0.0.9",
    "ragas>=0.2.0",
    "streamlit>=1.38.0",
    "tabulate>=0.9.0",
    "uvicorn[standard]>=0.30.0",
]
```
Run `uv sync` to install new packages.

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_deps.py -v`  
Expected: PASS

- [ ] **Step 5: Commit changes**

```bash
git add pyproject.toml uv.lock tests/test_deps.py
git commit -m "build: add ragas and langchain dependencies for chunking evaluation"
```

---

### Task 2: Chunking Base Interface & Fixed-Size Chunker Strategy

**Files:**
- Create: `src/exocortex/chunking/base.py`
- Create: `src/exocortex/chunking/fixed_size.py`
- Test: `tests/test_chunking_fixed.py`

**Interfaces:**
- Consumes: `PageContent`, `Chunk` from `src/exocortex/ingestion.py`
- Produces: `BaseChunker`, `FixedSizeChunker` in `src/exocortex/chunking/`

- [ ] **Step 1: Write unit tests for BaseChunker and FixedSizeChunker**

```python
# tests/test_chunking_fixed.py
import pytest
from exocortex.ingestion import PageContent, Chunk
from exocortex.chunking.fixed_size import FixedSizeChunker

def test_fixed_size_chunker_basic():
    chunker = FixedSizeChunker(chunk_size=100, chunk_overlap=20)
    pages = [
        PageContent(page_number=1, text="This is page 1 content with some text to be split."),
        PageContent(page_number=2, text="This is page 2 content which continues the discussion."),
    ]
    chunks = chunker.chunk(pages, document_id="doc123", filename="test.pdf")
    assert len(chunks) > 0
    assert all(isinstance(c, Chunk) for c in chunks)
    assert chunks[0].document_id == "doc123"
    assert chunks[0].filename == "test.pdf"
    assert chunks[0].chunk_index == 0
    assert 1 in chunks[0].page_numbers

def test_fixed_size_chunker_validation():
    with pytest.raises(ValueError):
        FixedSizeChunker(chunk_size=50, chunk_overlap=50)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_chunking_fixed.py -v`  
Expected: FAIL with `ModuleNotFoundError`

- [ ] **Step 3: Implement `BaseChunker` and `FixedSizeChunker`**

`src/exocortex/chunking/base.py`:
```python
from __future__ import annotations
from abc import ABC, abstractmethod
from exocortex.ingestion import Chunk, PageContent

class BaseChunker(ABC):
    """Abstract base class for all chunking strategies."""

    @abstractmethod
    def chunk(
        self,
        pages: list[PageContent],
        document_id: str,
        filename: str,
    ) -> list[Chunk]:
        """Split page contents into a list of Chunks."""
        pass

    @staticmethod
    def build_page_segments(pages: list[PageContent]) -> tuple[str, list[tuple[int, int, str]]]:
        """Build concatenated text and offset mappings for page tracking."""
        segments: list[tuple[int, int, str]] = []
        current_offset = 0
        for page in pages:
            segments.append((current_offset, page.page_number, page.text))
            current_offset += len(page.text) + 1
        full_text = " ".join(page.text for page in pages)
        return full_text, segments

    @staticmethod
    def find_pages_for_range(
        segments: list[tuple[int, int, str]],
        start: int,
        end: int,
    ) -> list[int]:
        """Find which page numbers a character range spans."""
        pages = set()
        for seg_start, page_num, text in segments:
            seg_end = seg_start + len(text)
            if start < seg_end and end > seg_start:
                pages.add(page_num)
        return sorted(pages)
```

`src/exocortex/chunking/fixed_size.py`:
```python
from __future__ import annotations
from exocortex.chunking.base import BaseChunker
from exocortex.ingestion import Chunk, PageContent

class FixedSizeChunker(BaseChunker):
    """Fixed character window chunking with sliding overlap."""

    def __init__(self, chunk_size: int = 512, chunk_overlap: int = 50) -> None:
        if chunk_size <= 0:
            raise ValueError("chunk_size must be positive")
        if chunk_overlap < 0:
            raise ValueError("chunk_overlap must be non-negative")
        if chunk_overlap >= chunk_size:
            raise ValueError("chunk_overlap must be less than chunk_size")
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def chunk(
        self,
        pages: list[PageContent],
        document_id: str,
        filename: str,
    ) -> list[Chunk]:
        if not pages:
            return []

        full_text, segments = self.build_page_segments(pages)
        chunks: list[Chunk] = []
        step = self.chunk_size - self.chunk_overlap
        chunk_index = 0

        for start in range(0, len(full_text), step):
            end = min(start + self.chunk_size, len(full_text))
            chunk_text_content = full_text[start:end].strip()

            if not chunk_text_content:
                continue

            chunk_pages = self.find_pages_for_range(segments, start, end)
            chunks.append(
                Chunk(
                    text=chunk_text_content,
                    document_id=document_id,
                    filename=filename,
                    page_numbers=chunk_pages,
                    chunk_index=chunk_index,
                    metadata={"strategy": "fixed_size"},
                )
            )
            chunk_index += 1
            if end >= len(full_text):
                break

        return chunks
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_chunking_fixed.py -v`  
Expected: PASS

- [ ] **Step 5: Commit changes**

```bash
git add src/exocortex/chunking/base.py src/exocortex/chunking/fixed_size.py tests/test_chunking_fixed.py
git commit -m "feat(chunking): add BaseChunker interface and FixedSizeChunker strategy"
```

---

### Task 3: Recursive Character Chunker Strategy

**Files:**
- Create: `src/exocortex/chunking/recursive.py`
- Test: `tests/test_chunking_recursive.py`

**Interfaces:**
- Consumes: `BaseChunker`, `PageContent`, `Chunk`
- Produces: `RecursiveCharacterChunker` in `src/exocortex/chunking/recursive.py`

- [ ] **Step 1: Write unit tests for RecursiveCharacterChunker**

```python
# tests/test_chunking_recursive.py
from exocortex.ingestion import PageContent, Chunk
from exocortex.chunking.recursive import RecursiveCharacterChunker

def test_recursive_character_chunker_splits_paragraphs_and_sentences():
    chunker = RecursiveCharacterChunker(chunk_size=120, chunk_overlap=20)
    text = (
        "Paragraph 1 is about ML systems.\n\n"
        "Paragraph 2 is about training and inference bottlenecks.\n\n"
        "Paragraph 3 discusses latency and throughput."
    )
    pages = [PageContent(page_number=1, text=text)]
    chunks = chunker.chunk(pages, document_id="doc1", filename="doc1.pdf")
    assert len(chunks) >= 2
    assert all(len(c.text) <= 150 for c in chunks)
    assert all(c.page_numbers == [1] for c in chunks)
    assert chunks[0].chunk_index == 0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_chunking_recursive.py -v`  
Expected: FAIL with `ModuleNotFoundError`

- [ ] **Step 3: Implement `RecursiveCharacterChunker`**

`src/exocortex/chunking/recursive.py`:
```python
from __future__ import annotations
from exocortex.chunking.base import BaseChunker
from exocortex.ingestion import Chunk, PageContent

class RecursiveCharacterChunker(BaseChunker):
    """Recursively splits text on hierarchical separators (paragraphs -> sentences -> words)."""

    def __init__(
        self,
        chunk_size: int = 512,
        chunk_overlap: int = 50,
        separators: list[str] | None = None,
    ) -> None:
        if chunk_size <= 0:
            raise ValueError("chunk_size must be positive")
        if chunk_overlap < 0:
            raise ValueError("chunk_overlap must be non-negative")
        if chunk_overlap >= chunk_size:
            raise ValueError("chunk_overlap must be less than chunk_size")
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.separators = separators or ["\n\n", "\n", ". ", " ", ""]

    def _split_text(self, text: str, separators: list[str]) -> list[str]:
        """Split text recursively using the given separators."""
        final_chunks: list[str] = []
        separator = separators[-1]
        new_separators = []
        for i, sep in enumerate(separators):
            if sep == "":
                separator = ""
                break
            if sep in text:
                separator = sep
                new_separators = separators[i + 1:]
                break

        splits = text.split(separator) if separator else list(text)
        good_splits: list[str] = []
        _separator = "" if separator == "" else separator

        for s in splits:
            if len(s) < self.chunk_size:
                good_splits.append(s)
            else:
                if good_splits:
                    merged = self._merge_splits(good_splits, _separator)
                    final_chunks.extend(merged)
                    good_splits = []
                if not new_separators:
                    final_chunks.append(s)
                else:
                    other_info = self._split_text(s, new_separators)
                    final_chunks.extend(other_info)

        if good_splits:
            merged = self._merge_splits(good_splits, _separator)
            final_chunks.extend(merged)

        return final_chunks

    def _merge_splits(self, splits: list[str], separator: str) -> list[str]:
        """Combine small splits into chunks up to chunk_size with overlap."""
        docs: list[str] = []
        current_doc: list[str] = []
        total = 0

        for d in splits:
            _len = len(d)
            if total + _len + (len(separator) if current_doc else 0) > self.chunk_size:
                if total > 0:
                    doc = separator.join(current_doc).strip()
                    if doc:
                        docs.append(doc)
                    # Keep overlap from the end of current_doc
                    while total > self.chunk_overlap and current_doc:
                        popped = current_doc.pop(0)
                        total -= len(popped) + len(separator)
                        if total < 0:
                            total = 0
                current_doc.append(d)
                total += _len + (len(separator) if len(current_doc) > 1 else 0)
            else:
                current_doc.append(d)
                total += _len + (len(separator) if len(current_doc) > 1 else 0)

        if current_doc:
            doc = separator.join(current_doc).strip()
            if doc:
                docs.append(doc)

        return docs

    def chunk(
        self,
        pages: list[PageContent],
        document_id: str,
        filename: str,
    ) -> list[Chunk]:
        if not pages:
            return []

        full_text, segments = self.build_page_segments(pages)
        raw_chunks = self._split_text(full_text, self.separators)

        chunks: list[Chunk] = []
        current_search_start = 0

        for i, text_chunk in enumerate(raw_chunks):
            text_chunk = text_chunk.strip()
            if not text_chunk:
                continue

            # Locate position in full_text to get page mapping
            start_pos = full_text.find(text_chunk[:min(50, len(text_chunk))], current_search_start)
            if start_pos == -1:
                start_pos = current_search_start
            end_pos = start_pos + len(text_chunk)
            current_search_start = max(0, start_pos + max(1, len(text_chunk) - self.chunk_overlap))

            chunk_pages = self.find_pages_for_range(segments, start_pos, end_pos)
            chunks.append(
                Chunk(
                    text=text_chunk,
                    document_id=document_id,
                    filename=filename,
                    page_numbers=chunk_pages,
                    chunk_index=i,
                    metadata={"strategy": "recursive"},
                )
            )

        return chunks
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_chunking_recursive.py -v`  
Expected: PASS

- [ ] **Step 5: Commit changes**

```bash
git add src/exocortex/chunking/recursive.py tests/test_chunking_recursive.py
git commit -m "feat(chunking): add RecursiveCharacterChunker strategy"
```

---

### Task 4: Sentence & Paragraph-Aware Chunker Strategy

**Files:**
- Create: `src/exocortex/chunking/sentence_paragraph.py`
- Test: `tests/test_chunking_sentence.py`

**Interfaces:**
- Consumes: `BaseChunker`, `PageContent`, `Chunk`
- Produces: `SentenceParagraphChunker` in `src/exocortex/chunking/sentence_paragraph.py`

- [ ] **Step 1: Write unit tests for SentenceParagraphChunker**

```python
# tests/test_chunking_sentence.py
from exocortex.ingestion import PageContent
from exocortex.chunking.sentence_paragraph.py import SentenceParagraphChunker

def test_sentence_paragraph_chunker():
    chunker = SentenceParagraphChunker(chunk_size=150, sentence_overlap=1)
    text = (
        "Sentence one is here. Sentence two follows it. Sentence three is right next. "
        "Sentence four continues the discussion. Sentence five wraps up."
    )
    pages = [PageContent(page_number=1, text=text)]
    chunks = chunker.chunk(pages, document_id="doc1", filename="doc1.pdf")
    assert len(chunks) >= 2
    assert "Sentence one is here." in chunks[0].text
    assert chunks[0].metadata["strategy"] == "sentence_paragraph"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_chunking_sentence.py -v`  
Expected: FAIL with `ModuleNotFoundError`

- [ ] **Step 3: Implement `SentenceParagraphChunker`**

`src/exocortex/chunking/sentence_paragraph.py`:
```python
from __future__ import annotations
import re
from exocortex.chunking.base import BaseChunker
from exocortex.ingestion import Chunk, PageContent

class SentenceParagraphChunker(BaseChunker):
    """Splits text by natural sentence and paragraph boundaries, maintaining sentence overlap."""

    def __init__(
        self,
        chunk_size: int = 512,
        sentence_overlap: int = 1,
    ) -> None:
        if chunk_size <= 0:
            raise ValueError("chunk_size must be positive")
        if sentence_overlap < 0:
            raise ValueError("sentence_overlap must be non-negative")
        self.chunk_size = chunk_size
        self.sentence_overlap = sentence_overlap

    @staticmethod
    def _split_into_sentences(text: str) -> list[str]:
        """Split text into sentences using regex boundary matching."""
        sentence_end = re.compile(r'(?<=[.!?])\s+(?=[A-Z0-9"\'])')
        sentences = sentence_end.split(text)
        return [s.strip() for s in sentences if s.strip()]

    def chunk(
        self,
        pages: list[PageContent],
        document_id: str,
        filename: str,
    ) -> list[Chunk]:
        if not pages:
            return []

        full_text, segments = self.build_page_segments(pages)
        paragraphs = full_text.split("\n\n")

        all_sentences: list[str] = []
        for p in paragraphs:
            p = p.strip()
            if not p:
                continue
            all_sentences.extend(self._split_into_sentences(p))

        if not all_sentences:
            return []

        chunks: list[Chunk] = []
        current_sentences: list[str] = []
        current_len = 0
        chunk_index = 0
        search_start = 0

        i = 0
        while i < len(all_sentences):
            sent = all_sentences[i]
            sent_len = len(sent) + 1

            if current_len + sent_len > self.chunk_size and current_sentences:
                chunk_text = " ".join(current_sentences).strip()
                start_pos = full_text.find(chunk_text[:min(50, len(chunk_text))], search_start)
                if start_pos == -1:
                    start_pos = search_start
                end_pos = start_pos + len(chunk_text)
                search_start = max(0, start_pos)

                chunk_pages = self.find_pages_for_range(segments, start_pos, end_pos)
                chunks.append(
                    Chunk(
                        text=chunk_text,
                        document_id=document_id,
                        filename=filename,
                        page_numbers=chunk_pages,
                        chunk_index=chunk_index,
                        metadata={"strategy": "sentence_paragraph"},
                    )
                )
                chunk_index += 1

                # Overlap step: slide back by sentence_overlap
                overlap_count = min(self.sentence_overlap, len(current_sentences))
                if overlap_count > 0:
                    current_sentences = current_sentences[-overlap_count:]
                    current_len = sum(len(s) + 1 for s in current_sentences)
                else:
                    current_sentences = []
                    current_len = 0

            current_sentences.append(sent)
            current_len += sent_len
            i += 1

        if current_sentences:
            chunk_text = " ".join(current_sentences).strip()
            start_pos = full_text.find(chunk_text[:min(50, len(chunk_text))], search_start)
            if start_pos == -1:
                start_pos = search_start
            end_pos = start_pos + len(chunk_text)
            chunk_pages = self.find_pages_for_range(segments, start_pos, end_pos)
            chunks.append(
                Chunk(
                    text=chunk_text,
                    document_id=document_id,
                    filename=filename,
                    page_numbers=chunk_pages,
                    chunk_index=chunk_index,
                    metadata={"strategy": "sentence_paragraph"},
                )
            )

        return chunks
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_chunking_sentence.py -v`  
Expected: PASS

- [ ] **Step 5: Commit changes**

```bash
git add src/exocortex/chunking/sentence_paragraph.py tests/test_chunking_sentence.py
git commit -m "feat(chunking): add SentenceParagraphChunker strategy"
```

---

### Task 5: Semantic Chunker Strategy (Embedding Similarity)

**Files:**
- Create: `src/exocortex/chunking/semantic.py`
- Test: `tests/test_chunking_semantic.py`

**Interfaces:**
- Consumes: `EmbeddingClient` from `src/exocortex/embedding.py`, `BaseChunker`, `PageContent`, `Chunk`
- Produces: `SemanticChunker` in `src/exocortex/chunking/semantic.py`

- [ ] **Step 1: Write unit tests for SemanticChunker**

```python
# tests/test_chunking_semantic.py
from unittest.mock import MagicMock
import numpy as np
from exocortex.ingestion import PageContent
from exocortex.chunking.semantic import SemanticChunker

def test_semantic_chunker_breakpoint_splitting():
    mock_embedding_client = MagicMock()
    # Mock embeddings such that sentence 1 & 2 are close, 3 & 4 are close, but 2 and 3 are far apart
    v1 = [1.0, 0.0]
    v2 = [0.95, 0.05]
    v3 = [0.0, 1.0]
    v4 = [0.05, 0.95]
    mock_embedding_client.embed_documents.return_value = [v1, v2, v3, v4]

    chunker = SemanticChunker(
        embedding_client=mock_embedding_client,
        distance_threshold_percentile=50.0,
        max_chunk_size=1000,
    )
    pages = [
        PageContent(
            page_number=1,
            text="Apples are delicious fruits. Oranges are also sweet citrus. Machine learning algorithms train models. Deep learning uses neural networks."
        )
    ]
    chunks = chunker.chunk(pages, document_id="doc1", filename="doc1.pdf")
    assert len(chunks) == 2
    assert "fruits" in chunks[0].text
    assert "Machine learning" in chunks[1].text
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_chunking_semantic.py -v`  
Expected: FAIL with `ModuleNotFoundError`

- [ ] **Step 3: Implement `SemanticChunker`**

`src/exocortex/chunking/semantic.py`:
```python
from __future__ import annotations
import math
import re
from typing import TYPE_CHECKING
from exocortex.chunking.base import BaseChunker
from exocortex.ingestion import Chunk, PageContent

if TYPE_CHECKING:
    from exocortex.embedding import EmbeddingClient

class SemanticChunker(BaseChunker):
    """Splits text dynamically based on cosine distance between consecutive sentence embeddings."""

    def __init__(
        self,
        embedding_client: EmbeddingClient | None = None,
        distance_threshold_percentile: float = 85.0,
        min_chunk_size: int = 100,
        max_chunk_size: int = 1000,
    ) -> None:
        self.embedding_client = embedding_client
        self.distance_threshold_percentile = distance_threshold_percentile
        self.min_chunk_size = min_chunk_size
        self.max_chunk_size = max_chunk_size

    @staticmethod
    def _cosine_distance(vec1: list[float], vec2: list[float]) -> float:
        """Compute cosine distance (1 - cosine_similarity)."""
        dot = sum(a * b for a, b in zip(vec1, vec2))
        norm1 = math.sqrt(sum(a * a for a in vec1))
        norm2 = math.sqrt(sum(b * b for b in vec2))
        if norm1 == 0 or norm2 == 0:
            return 1.0
        sim = dot / (norm1 * norm2)
        return max(0.0, 1.0 - sim)

    @staticmethod
    def _split_into_sentences(text: str) -> list[str]:
        sentence_end = re.compile(r'(?<=[.!?])\s+(?=[A-Z0-9"\'])')
        sentences = sentence_end.split(text)
        return [s.strip() for s in sentences if s.strip()]

    def chunk(
        self,
        pages: list[PageContent],
        document_id: str,
        filename: str,
    ) -> list[Chunk]:
        if not pages:
            return []

        full_text, segments = self.build_page_segments(pages)
        sentences = self._split_into_sentences(full_text)

        if not sentences:
            return []
        if len(sentences) == 1:
            return [
                Chunk(
                    text=sentences[0],
                    document_id=document_id,
                    filename=filename,
                    page_numbers=self.find_pages_for_range(segments, 0, len(sentences[0])),
                    chunk_index=0,
                    metadata={"strategy": "semantic"},
                )
            ]

        # Embed sentences
        if self.embedding_client is None:
            from exocortex.config import get_settings
            from exocortex.embedding import EmbeddingClient
            self.embedding_client = EmbeddingClient(get_settings())

        embeddings = self.embedding_client.embed_documents(sentences)

        # Compute distances between consecutive sentences
        distances: list[float] = []
        for i in range(len(embeddings) - 1):
            dist = self._cosine_distance(embeddings[i], embeddings[i + 1])
            distances.append(dist)

        # Determine threshold
        sorted_dists = sorted(distances)
        idx = int(len(sorted_dists) * (self.distance_threshold_percentile / 100.0))
        idx = min(idx, len(sorted_dists) - 1)
        threshold = sorted_dists[idx]

        # Split into groups
        chunks: list[Chunk] = []
        current_group: list[str] = [sentences[0]]
        current_len = len(sentences[0])
        chunk_index = 0
        search_start = 0

        for i, dist in enumerate(distances):
            next_sent = sentences[i + 1]
            should_split = (dist >= threshold and current_len >= self.min_chunk_size) or (
                current_len + len(next_sent) > self.max_chunk_size
            )

            if should_split:
                chunk_text = " ".join(current_group).strip()
                start_pos = full_text.find(chunk_text[:min(50, len(chunk_text))], search_start)
                if start_pos == -1:
                    start_pos = search_start
                end_pos = start_pos + len(chunk_text)
                search_start = max(0, start_pos)

                chunk_pages = self.find_pages_for_range(segments, start_pos, end_pos)
                chunks.append(
                    Chunk(
                        text=chunk_text,
                        document_id=document_id,
                        filename=filename,
                        page_numbers=chunk_pages,
                        chunk_index=chunk_index,
                        metadata={"strategy": "semantic"},
                    )
                )
                chunk_index += 1
                current_group = [next_sent]
                current_len = len(next_sent)
            else:
                current_group.append(next_sent)
                current_len += len(next_sent) + 1

        if current_group:
            chunk_text = " ".join(current_group).strip()
            start_pos = full_text.find(chunk_text[:min(50, len(chunk_text))], search_start)
            if start_pos == -1:
                start_pos = search_start
            end_pos = start_pos + len(chunk_text)
            chunk_pages = self.find_pages_for_range(segments, start_pos, end_pos)
            chunks.append(
                Chunk(
                    text=chunk_text,
                    document_id=document_id,
                    filename=filename,
                    page_numbers=chunk_pages,
                    chunk_index=chunk_index,
                    metadata={"strategy": "semantic"},
                )
            )

        return chunks
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_chunking_semantic.py -v`  
Expected: PASS

- [ ] **Step 5: Commit changes**

```bash
git add src/exocortex/chunking/semantic.py tests/test_chunking_semantic.py
git commit -m "feat(chunking): add SemanticChunker strategy with embedding similarity"
```

---

### Task 6: Factory, Config & Ingestion Pipeline Integration

**Files:**
- Create: `src/exocortex/chunking/factory.py`
- Create: `src/exocortex/chunking/__init__.py`
- Modify: `src/exocortex/config.py:36-39`
- Modify: `src/exocortex/ingestion.py:216-253`
- Test: `tests/test_chunking_factory.py`
- Test: `tests/test_ingestion.py`

**Interfaces:**
- Consumes: All chunker classes, `Settings`
- Produces: `get_chunker(strategy: str, ...)` and updated `ingest_pdf()`

- [ ] **Step 1: Write tests for factory and updated ingest_pdf**

```python
# tests/test_chunking_factory.py
import pytest
from exocortex.chunking import get_chunker, FixedSizeChunker, RecursiveCharacterChunker, SentenceParagraphChunker, SemanticChunker

def test_get_chunker_strategies():
    assert isinstance(get_chunker("fixed"), FixedSizeChunker)
    assert isinstance(get_chunker("recursive"), RecursiveCharacterChunker)
    assert isinstance(get_chunker("sentence_paragraph"), SentenceParagraphChunker)
    assert isinstance(get_chunker("semantic"), SemanticChunker)

def test_get_chunker_invalid():
    with pytest.raises(ValueError):
        get_chunker("unknown_strategy")
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_chunking_factory.py -v`  
Expected: FAIL with `ModuleNotFoundError`

- [ ] **Step 3: Implement `factory.py`, `__init__.py`, update `config.py` and `ingestion.py`**

`src/exocortex/chunking/factory.py`:
```python
from __future__ import annotations
from typing import Any
from exocortex.chunking.base import BaseChunker
from exocortex.chunking.fixed_size import FixedSizeChunker
from exocortex.chunking.recursive import RecursiveCharacterChunker
from exocortex.chunking.sentence_paragraph import SentenceParagraphChunker
from exocortex.chunking.semantic import SemanticChunker

def get_chunker(
    strategy: str = "fixed",
    **kwargs: Any,
) -> BaseChunker:
    """Factory to retrieve a chunker by strategy name."""
    s = strategy.lower().strip()
    if s in ("fixed", "fixed_size"):
        chunk_size = kwargs.get("chunk_size", 512)
        chunk_overlap = kwargs.get("chunk_overlap", 50)
        return FixedSizeChunker(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    elif s in ("recursive", "recursive_character"):
        chunk_size = kwargs.get("chunk_size", 512)
        chunk_overlap = kwargs.get("chunk_overlap", 50)
        separators = kwargs.get("separators")
        return RecursiveCharacterChunker(chunk_size=chunk_size, chunk_overlap=chunk_overlap, separators=separators)
    elif s in ("sentence_paragraph", "sentence", "paragraph"):
        chunk_size = kwargs.get("chunk_size", 512)
        sentence_overlap = kwargs.get("sentence_overlap", 1)
        return SentenceParagraphChunker(chunk_size=chunk_size, sentence_overlap=sentence_overlap)
    elif s == "semantic":
        return SemanticChunker(
            embedding_client=kwargs.get("embedding_client"),
            distance_threshold_percentile=kwargs.get("distance_threshold_percentile", 85.0),
            min_chunk_size=kwargs.get("min_chunk_size", 100),
            max_chunk_size=kwargs.get("max_chunk_size", 1000),
        )
    else:
        raise ValueError(f"Unknown chunking strategy: '{strategy}'. Supported: fixed, recursive, sentence_paragraph, semantic")
```

`src/exocortex/chunking/__init__.py`:
```python
from exocortex.chunking.base import BaseChunker
from exocortex.chunking.fixed_size import FixedSizeChunker
from exocortex.chunking.recursive import RecursiveCharacterChunker
from exocortex.chunking.sentence_paragraph import SentenceParagraphChunker
from exocortex.chunking.semantic import SemanticChunker
from exocortex.chunking.factory import get_chunker

__all__ = [
    "BaseChunker",
    "FixedSizeChunker",
    "RecursiveCharacterChunker",
    "SentenceParagraphChunker",
    "SemanticChunker",
    "get_chunker",
]
```

Update `src/exocortex/config.py`:
Add `chunking_strategy: str = "fixed"` to `Settings`.

Update `src/exocortex/ingestion.py`:
Update `ingest_pdf()` to accept `strategy: str = "fixed"` and delegate to `get_chunker()`.

- [ ] **Step 4: Run all chunking and ingestion tests**

Run: `uv run pytest tests/test_chunking_factory.py tests/test_ingestion.py -v`  
Expected: PASS

- [ ] **Step 5: Commit changes**

```bash
git add src/exocortex/chunking/ src/exocortex/config.py src/exocortex/ingestion.py tests/test_chunking_factory.py
git commit -m "feat(chunking): integrate chunking factory with ingestion pipeline and settings"
```

---

### Task 7: Golden Dataset Parser

**Files:**
- Create: `src/exocortex/eval/__init__.py`
- Create: `src/exocortex/eval/dataset.py`
- Test: `tests/test_eval_dataset.py`

**Interfaces:**
- Consumes: `data/GoldenDatset.md`
- Produces: `GoldenSample`, `load_golden_dataset(path: Path) -> list[GoldenSample]`

- [ ] **Step 1: Write test for dataset parser**

```python
# tests/test_eval_dataset.py
from pathlib import Path
from exocortex.eval.dataset import load_golden_dataset, GoldenSample

def test_load_golden_dataset():
    path = Path("data/GoldenDatset.md")
    dataset = load_golden_dataset(path)
    assert len(dataset) == 22
    sample1 = dataset[0]
    assert sample1.entry_id == 1
    assert "production machine learning system" in sample1.question.lower()
    assert len(sample1.ground_truth) > 20
    assert 1 in sample1.reference_pages
    assert len(sample1.excerpt_context) > 0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_eval_dataset.py -v`  
Expected: FAIL with `ModuleNotFoundError`

- [ ] **Step 3: Implement `src/exocortex/eval/dataset.py`**

```python
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
import re

@dataclass
class GoldenSample:
    """A single evaluation sample from the Golden Dataset."""
    entry_id: int
    question: str
    ground_truth: str
    reference_pages: list[int]
    reference_location: str
    excerpt_context: str

def parse_pages_from_reference(ref_str: str) -> list[int]:
    """Parse page numbers like 'Page 1', 'Page 4–5', 'Pages 15–16' into a list of ints."""
    pages: list[int] = []
    # Match patterns like Page 1, Pages 1-23, Page 4–5
    match_range = re.search(r'Pages?\s+(\d+)\s*[–\-]\s*(\d+)', ref_str, re.IGNORECASE)
    if match_range:
        start, end = int(match_range.group(1)), int(match_range.group(2))
        return list(range(start, end + 1))
    
    match_single = re.search(r'Pages?\s+(\d+)', ref_str, re.IGNORECASE)
    if match_single:
        return [int(match_single.group(1))]

    # Fallback to any numbers found
    numbers = re.findall(r'\b\d+\b', ref_str)
    return [int(n) for n in numbers] if numbers else []

def load_golden_dataset(path: Path | str = "data/GoldenDatset.md") -> list[GoldenSample]:
    """Load and parse Golden Dataset from markdown file."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Golden dataset not found at {path}")

    content = path.read_text(encoding="utf-8")
    # Split entries by "Entry N Question:"
    pattern = re.compile(r'Entry\s+(\d+)\s+Question:\s*(.*?)\s*Ground Truth Answer:\s*(.*?)\s*Reference Location:\s*(.*?)\s*Excerpt Context:\s*(.*?)(?=(?:Entry\s+\d+\s+Question:|$))', re.DOTALL)

    matches = pattern.findall(content)
    samples: list[GoldenSample] = []

    for entry_id_str, q, gt, ref_loc, excerpt in matches:
        entry_id = int(entry_id_str)
        ref_pages = parse_pages_from_reference(ref_loc)
        # Clean excerpt quotes if any
        excerpt_clean = excerpt.strip().strip('"')
        samples.append(
            GoldenSample(
                entry_id=entry_id,
                question=q.strip(),
                ground_truth=gt.strip(),
                reference_pages=ref_pages,
                reference_location=ref_loc.strip(),
                excerpt_context=excerpt_clean,
            )
        )

    return samples
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_eval_dataset.py -v`  
Expected: PASS

- [ ] **Step 5: Commit changes**

```bash
git add src/exocortex/eval/__init__.py src/exocortex/eval/dataset.py tests/test_eval_dataset.py
git commit -m "feat(eval): add Golden Dataset parser for 22 evaluation samples"
```

---

### Task 8: Traditional Retrieval & Operational Metrics

**Files:**
- Create: `src/exocortex/eval/metrics.py`
- Test: `tests/test_eval_metrics.py`

**Interfaces:**
- Consumes: `SearchResult` from `src/exocortex/vectorstore.py`, `GoldenSample`
- Produces: `compute_retrieval_metrics()`, `compute_chunk_statistics()`

- [ ] **Step 1: Write tests for retrieval and operational metrics**

```python
# tests/test_eval_metrics.py
from exocortex.vectorstore import SearchResult
from exocortex.eval.metrics import calculate_hit_rate_and_mrr, compute_chunk_statistics
from exocortex.ingestion import Chunk

def test_hit_rate_and_mrr():
    ref_pages = [4, 5]
    results = [
        SearchResult(text="c1", metadata={"page_numbers": "2,3"}, distance=0.1, chunk_id="1"),
        SearchResult(text="c2", metadata={"page_numbers": "4,5"}, distance=0.2, chunk_id="2"),
        SearchResult(text="c3", metadata={"page_numbers": "5"}, distance=0.3, chunk_id="3"),
    ]
    hit, mrr = calculate_hit_rate_and_mrr(results, ref_pages, top_k=3)
    assert hit == 1.0
    assert mrr == 0.5  # First match at rank 2 (1/2)

def test_chunk_statistics():
    chunks = [
        Chunk(text="Hello world", document_id="1", filename="f", page_numbers=[1], chunk_index=0),
        Chunk(text="Another test chunk with more tokens", document_id="1", filename="f", page_numbers=[1], chunk_index=1),
    ]
    stats = compute_chunk_statistics(chunks)
    assert stats["chunk_count"] == 2
    assert stats["mean_chars"] > 0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_eval_metrics.py -v`  
Expected: FAIL with `ModuleNotFoundError`

- [ ] **Step 3: Implement `src/exocortex/eval/metrics.py`**

```python
from __future__ import annotations
import statistics
from typing import Any
from exocortex.ingestion import Chunk
from exocortex.vectorstore import SearchResult

def parse_chunk_page_numbers(metadata: dict) -> list[int]:
    """Parse comma-separated page numbers string from chunk metadata."""
    raw = metadata.get("page_numbers", "")
    if isinstance(raw, list):
        return [int(x) for x in raw]
    if not raw or raw == "?":
        return []
    try:
        return [int(p.strip()) for p in str(raw).split(",") if p.strip().isdigit()]
    except Exception:
        return []

def calculate_hit_rate_and_mrr(
    search_results: list[SearchResult],
    reference_pages: list[int],
    top_k: int = 5,
) -> tuple[float, float]:
    """Calculate Hit@K (0 or 1) and Reciprocal Rank (1/rank) for a single query."""
    if not search_results or not reference_pages:
        return 0.0, 0.0

    ref_set = set(reference_pages)
    hit = 0.0
    reciprocal_rank = 0.0

    for rank, res in enumerate(search_results[:top_k], start=1):
        chunk_pages = parse_chunk_page_numbers(res.metadata)
        if any(p in ref_set for p in chunk_pages):
            hit = 1.0
            reciprocal_rank = 1.0 / rank
            break

    return hit, reciprocal_rank

def compute_chunk_statistics(chunks: list[Chunk]) -> dict[str, Any]:
    """Compute character length and count statistics across chunks."""
    if not chunks:
        return {
            "chunk_count": 0,
            "mean_chars": 0.0,
            "median_chars": 0.0,
            "min_chars": 0,
            "max_chars": 0,
        }

    lengths = [len(c.text) for c in chunks]
    return {
        "chunk_count": len(chunks),
        "mean_chars": round(statistics.mean(lengths), 2),
        "median_chars": round(statistics.median(lengths), 2),
        "min_chars": min(lengths),
        "max_chars": max(lengths),
    }
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_eval_metrics.py -v`  
Expected: PASS

- [ ] **Step 5: Commit changes**

```bash
git add src/exocortex/eval/metrics.py tests/test_eval_metrics.py
git commit -m "feat(eval): add traditional retrieval metrics and chunk statistics"
```

---

### Task 9: Ragas Evaluator Module

**Files:**
- Create: `src/exocortex/eval/ragas_evaluator.py`
- Test: `tests/test_ragas_evaluator.py`

**Interfaces:**
- Consumes: `Settings`, `ragas`, `langchain_openai.ChatOpenAI`
- Produces: `RagasEvaluator`, `evaluate_ragas_dataset()`

- [ ] **Step 1: Write test with mocks for Ragas evaluator initialization and evaluation**

```python
# tests/test_ragas_evaluator.py
from unittest.mock import MagicMock, patch
import pandas as pd
from exocortex.config import Settings
from exocortex.eval.ragas_evaluator import RagasEvaluator

def test_ragas_evaluator_init():
    settings = Settings(deepseek_api_key="test_key")
    evaluator = RagasEvaluator(settings=settings)
    assert evaluator.llm is not None
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_ragas_evaluator.py -v`  
Expected: FAIL with `ModuleNotFoundError`

- [ ] **Step 3: Implement `src/exocortex/eval/ragas_evaluator.py`**

```python
from __future__ import annotations
import logging
from typing import Any
import pandas as pd
from datasets import Dataset
from ragas import evaluate
from ragas.metrics import (
    answer_relevancy,
    context_precision,
    context_recall,
    faithfulness,
)
from langchain_openai import ChatOpenAI
from langchain_community.embeddings import OllamaEmbeddings
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
            return {"context_precision": 0.0, "context_recall": 0.0, "faithfulness": 0.0, "answer_relevancy": 0.0, "df": pd.DataFrame()}

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
            "context_precision": float(df_result["context_precision"].mean()) if "context_precision" in df_result else 0.0,
            "context_recall": float(df_result["context_recall"].mean()) if "context_recall" in df_result else 0.0,
            "faithfulness": float(df_result["faithfulness"].mean()) if "faithfulness" in df_result else 0.0,
            "answer_relevancy": float(df_result["answer_relevancy"].mean()) if "answer_relevancy" in df_result else 0.0,
            "df": df_result,
        }
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_ragas_evaluator.py -v`  
Expected: PASS

- [ ] **Step 5: Commit changes**

```bash
git add src/exocortex/eval/ragas_evaluator.py tests/test_ragas_evaluator.py
git commit -m "feat(eval): add RagasEvaluator module with DeepSeek and Ollama integration"
```

---

### Task 10: Benchmark Runner & Markdown/CSV Report Generator

**Files:**
- Create: `src/exocortex/eval/benchmark.py`
- Test: `tests/test_benchmark.py`

**Interfaces:**
- Consumes: All chunkers, `load_golden_dataset`, `RagasEvaluator`, `RAGEngine`, `VectorStore`
- Produces: `run_benchmark()`, Markdown report, CSV data file

- [ ] **Step 1: Write test for benchmark aggregation and table generation**

```python
# tests/test_benchmark.py
from exocortex.eval.benchmark import format_markdown_report

def test_format_markdown_report():
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
        }
    }
    report = format_markdown_report(results)
    assert "# Chunking Strategies Benchmark Report" in report
    assert "fixed" in report
    assert "0.850" in report
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_benchmark.py -v`  
Expected: FAIL with `ModuleNotFoundError`

- [ ] **Step 3: Implement `src/exocortex/eval/benchmark.py`**

```python
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
    rows = []
    for strat, data in benchmark_results.items():
        rows.append([
            strat,
            data["chunk_count"],
            f"{data['mean_chars']:.0f}",
            f"{data['hit_rate_top_k']:.3f}",
            f"{data['mrr']:.3f}",
            f"{data['context_precision']:.3f}",
            f"{data['context_recall']:.3f}",
            f"{data['faithfulness']:.3f}",
            f"{data['answer_relevancy']:.3f}",
            f"{data['avg_retrieval_ms']:.1f} ms",
        ])

    headers = [
        "Strategy", "Chunks", "Avg Len", "HitRate@K", "MRR",
        "Context Precision", "Context Recall", "Faithfulness", "Answer Relevancy", "Latency"
    ]
    table_md = tabulate(rows, headers=headers, tablefmt="github")

    # Find winning strategy based on combined Ragas + Retrieval score
    best_strategy = max(
        benchmark_results.keys(),
        key=lambda s: (
            benchmark_results[s]["context_recall"] * 0.3
            + benchmark_results[s]["context_precision"] * 0.3
            + benchmark_results[s]["faithfulness"] * 0.2
            + benchmark_results[s]["hit_rate_top_k"] * 0.2
        )
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
    pdf_path: Path = Path("data/ebooks/Designing Machine Learning Systems An Iterative Process for Production-Ready (Chip Huyen)[21-43].pdf"),
    golden_path: Path = Path("data/GoldenDatset.md"),
    output_md: Path = Path("docs/benchmark_results.md"),
    output_csv: Path = Path("data/benchmark_results.csv"),
    strategies: list[str] | None = None,
) -> dict[str, Any]:
    """Execute end-to-end benchmark across all chunking strategies."""
    strategies = strategies or STRATEGIES
    settings = get_settings()
    golden_samples = load_golden_dataset(golden_path)
    pages = extract_text_from_pdf(pdf_path)
    document_id = generate_document_id(pdf_path.name)

    embedding_client = EmbeddingClient(settings)
    llm_client = LLMClient(settings)
    ragas_eval = RagasEvaluator(settings)

    all_results = {}
    detailed_rows = []

    for strat in strategies:
        logger.info(f"\n==================== Benchmarking Strategy: {strat} ====================")
        chunker = get_chunker(strat, embedding_client=embedding_client)

        t0 = time.perf_counter()
        chunks = chunker.chunk(pages, document_id=document_id, filename=pdf_path.name)
        chunking_time = time.perf_counter() - t0

        stats = compute_chunk_statistics(chunks)
        logger.info(f"Created {len(chunks)} chunks in {chunking_time:.2f}s (mean length: {stats['mean_chars']} chars)")

        # Create isolated collection for this strategy
        coll_name = f"benchmark_{strat}_{int(time.time())}"
        settings_copy = settings.model_copy(update={"chroma_collection_name": coll_name})
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

            hit, mrr = calculate_hit_rate_and_mrr(results, sample.reference_pages, top_k=settings.top_k)
            hits.append(hit)
            mrrs.append(mrr)

            # Step 2: Generate answer with LLM
            llm_resp = llm_client.generate(query=sample.question, search_results=results)

            ragas_records.append({
                "question": sample.question,
                "answer": llm_resp.answer,
                "contexts": [r.text for r in results],
                "ground_truth": sample.ground_truth,
            })

            detailed_rows.append({
                "strategy": strat,
                "entry_id": sample.entry_id,
                "question": sample.question,
                "hit": hit,
                "mrr": mrr,
                "retrieval_ms": ret_duration,
            })

        # Run Ragas evaluation
        ragas_scores = ragas_eval.evaluate_records(ragas_records)

        all_results[strat] = {
            "strategy": strat,
            "chunk_count": stats["chunk_count"],
            "mean_chars": stats["mean_chars"],
            "hit_rate_top_k": sum(hits) / len(hits) if hits else 0.0,
            "mrr": sum(mrrs) / len(mrrs) if mrrs else 0.0,
            "context_precision": ragas_scores["context_precision"],
            "context_recall": ragas_scores["context_recall"],
            "faithfulness": ragas_scores["faithfulness"],
            "answer_relevancy": ragas_scores["answer_relevancy"],
            "chunking_time_s": chunking_time,
            "avg_retrieval_ms": sum(retrieval_times) / len(retrieval_times) if retrieval_times else 0.0,
        }

        # Clean up temporary chroma collection
        try:
            vector_store.client.delete_collection(coll_name)
        except Exception:
            pass

    # Export outputs
    report_md = format_markdown_report(all_results)
    output_md.parent.mkdir(parents=True, exist_ok=True)
    output_md.write_text(report_md, encoding="utf-8")

    df_details = pd.DataFrame(detailed_rows)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    df_details.to_csv(output_csv, index=False)

    print("\n" + report_md)
    return all_results

if __name__ == "__main__":
    run_benchmark()
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_benchmark.py -v`  
Expected: PASS

- [ ] **Step 5: Commit changes**

```bash
git add src/exocortex/eval/benchmark.py tests/test_benchmark.py
git commit -m "feat(eval): add end-to-end benchmark suite and markdown/csv report generator"
```

---

### Task 11: Execute Benchmark, Generate Report & Finalize Configuration

**Files:**
- Execute: `python -m exocortex.eval.benchmark`
- Generate: `docs/benchmark_results.md`
- Generate: `data/benchmark_results.csv`
- Modify: `src/exocortex/config.py:36` (set winning chunking strategy)

- [ ] **Step 1: Execute full benchmark run on Designing Machine Learning Systems**

Run: `uv run python -m exocortex.eval.benchmark`  
Verify: `docs/benchmark_results.md` and `data/benchmark_results.csv` are created and contain complete scores across all 4 strategies.

- [ ] **Step 2: Update `config.py` default `chunking_strategy` with the winning strategy**

- [ ] **Step 3: Run all project tests to ensure 100% pass**

Run: `uv run pytest -v`  
Expected: ALL PASS

- [ ] **Step 4: Commit benchmark results and config update**

```bash
git add docs/benchmark_results.md data/benchmark_results.csv src/exocortex/config.py
git commit -m "chore(eval): record benchmark results and configure optimal chunking strategy"
```
