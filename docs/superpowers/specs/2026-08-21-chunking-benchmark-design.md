# Design Specification: Chunking Strategies & Ragas Benchmarking Framework

**Date:** 2026-08-21  
**Status:** Approved  
**Author:** Pair Programming Agent & User  
**Target Branch:** `feature/improve-chunking`

---

## 1. Overview & Objectives

The goal of this project is to explore, implement, and rigorously benchmark various text chunking strategies for the Exocortex RAG system using **Ragas** and traditional retrieval metrics. Based on the evaluation against a curated **Golden Dataset** (22 Q&As from Chip Huyen's *Designing Machine Learning Systems*, Chapter 1), we will select and configure the optimal chunking strategy for Exocortex.

---

## 2. Architecture & Modular Chunking Framework

We refactor and modularize chunking using the **Strategy Pattern** under `src/exocortex/chunking/`.

### 2.1 Component Structure

```
src/exocortex/
├── chunking/
│   ├── __init__.py               # Exports factory and chunkers
│   ├── base.py                   # BaseChunker abstract class & interfaces
│   ├── fixed_size.py             # FixedSizeChunker (Baseline)
│   ├── recursive.py              # RecursiveCharacterChunker
│   ├── sentence_paragraph.py     # SentenceParagraphChunker
│   ├── semantic.py               # SemanticChunker (embedding-based similarity)
│   └── factory.py                # get_chunker(strategy: str, **kwargs)
├── eval/
│   ├── __init__.py
│   ├── dataset.py                # Golden Dataset parser from data/GoldenDatset.md
│   ├── ragas_evaluator.py        # Ragas evaluation wrapper with DeepSeek & Ollama
│   ├── metrics.py                # Traditional metrics (HitRate@K, MRR, Latency, Stats)
│   └── benchmark.py              # End-to-end benchmark runner & report generator
```

### 2.2 Strategy Specifications

1. **`BaseChunker`** (`src/exocortex/chunking/base.py`):
   - Method: `chunk(pages: list[PageContent], document_id: str, filename: str) -> list[Chunk]`
   - Maintains continuous character offset tracking across pages so `page_numbers` (1-indexed list) is accurately assigned to each resulting `Chunk`.

2. **`FixedSizeChunker`** (`src/exocortex/chunking/fixed_size.py`):
   - Fixed character/token window slicing with sliding overlap (`chunk_size=512`, `chunk_overlap=50`).
   - Serves as the baseline comparison against existing behavior.

3. **`RecursiveCharacterChunker`** (`src/exocortex/chunking/recursive.py`):
   - Recursively splits text using hierarchical separators (`["\n\n", "\n", ". ", " ", ""]`).
   - Merges atomic pieces up to `chunk_size` characters with `chunk_overlap` overlap, ensuring logical units (paragraphs/sentences) stay intact as much as possible.

4. **`SentenceParagraphChunker`** (`src/exocortex/chunking/sentence_paragraph.py`):
   - Parses text into paragraphs, and paragraphs into sentences (using regex / punctuation rules).
   - Groups sentences sequentially until `chunk_size` is reached, retaining `sentence_overlap` sentences between chunks to preserve grammatical continuity.

5. **`SemanticChunker`** (`src/exocortex/chunking/semantic.py`):
   - Splits text into discrete sentences.
   - Computes embedding vectors for adjacent sentences using Exocortex's `EmbeddingClient` (Ollama `qwen3-embedding:0.6b`).
   - Computes cosine distance between consecutive sentences.
   - Identifies semantic breakpoints using percentile or standard deviation thresholding (e.g. 90th percentile of distances or mean + 1.5 * std).
   - Merges sentences between breakpoints into unified semantic chunks.

6. **`Factory`** (`src/exocortex/chunking/factory.py`):
   - Provides `get_chunker(strategy_name: str, settings: Settings | None = None, **kwargs) -> BaseChunker`.
   - Strategies supported: `"fixed"`, `"recursive"`, `"sentence_paragraph"`, `"semantic"`.

---

## 3. Evaluation & Benchmarking System

### 3.1 Golden Dataset Parser (`src/exocortex/eval/dataset.py`)
- Parses `data/GoldenDatset.md`.
- Extracts 22 samples containing:
  - `question`: Query string.
  - `ground_truth`: Target reference answer string.
  - `reference_pages`: Extracted list of 1-indexed page integers (e.g. `[1]`, `[4, 5]`).
  - `excerpt_context`: Direct quote/context from the source book.

### 3.2 Evaluation Metrics
1. **Ragas Metrics**:
   - `context_precision`: Precision of retrieved chunks relative to ground truth.
   - `context_recall`: Coverage of ground truth facts by the retrieved context.
   - `faithfulness`: Degree to which the LLM response is grounded in the retrieved chunks.
   - `answer_relevancy`: Relevance of the LLM response to the user query.
2. **Traditional Retrieval Metrics**:
   - **Hit Rate@K**: Fraction of queries where at least one retrieved chunk in Top-K belongs to `reference_pages`.
   - **MRR (Mean Reciprocal Rank)**: `1 / rank` of the first retrieved chunk matching `reference_pages` (0 if none match).
3. **Operational / Profiling Metrics**:
   - Total chunk count.
   - Mean & median chunk length (characters / tokens).
   - Ingestion & chunking execution time (seconds).
   - Average retrieval latency per query (milliseconds).

### 3.3 Ragas Evaluator Integration (`src/exocortex/eval/ragas_evaluator.py`)
- Uses `ragas` library.
- Configures `evaluator_llm` wrapping DeepSeek via `langchain_openai.ChatOpenAI` / `openai.OpenAI` pointing to `deepseek_base_url` (`https://api.deepseek.com`) with `deepseek-v4-flash`.
- Configures `evaluator_embeddings` using Ollama `qwen3-embedding:0.6b` via `langchain_community.embeddings.OllamaEmbeddings` or custom LangChain-compatible embedding wrapper.

### 3.4 Benchmark Runner (`src/exocortex/eval/benchmark.py`)
- Workflow:
  1. Loads `data/ebooks/Designing Machine Learning Systems An Iterative Process for Production-Ready (Chip Huyen)[21-43].pdf`.
  2. Extracts pages once using `extract_text_from_pdf()`.
  3. For each of the 4 chunking strategies:
     - Applies chunker to generate chunks.
     - Creates an isolated temporary ChromaDB collection.
     - Embeds and indexes chunks into the collection.
     - Executes retrieval and generation for all 22 golden dataset questions.
     - Computes Ragas metrics and traditional retrieval metrics.
  4. Aggregates results into a comprehensive comparison summary.
  5. Exports:
     - Markdown report: `docs/benchmark_results.md` (and printed to console).
     - Raw CSV data: `data/benchmark_results.csv` with per-sample scores.
     - Automated recommendation of the best chunking strategy.

---

## 4. Dependencies & Testing

### 4.1 Dependencies
Add required evaluation packages to `pyproject.toml` (under `[dependency-groups] dev` or main dependencies):
- `ragas>=0.2.0`
- `langchain>=0.3.0`
- `langchain-openai>=0.2.0`
- `langchain-community>=0.3.0`
- `pandas>=2.2.0`
- `tabulate>=0.9.0`

### 4.2 Unit & Integration Testing
- `tests/test_chunking.py`: Unit tests for `FixedSizeChunker`, `RecursiveCharacterChunker`, `SentenceParagraphChunker`, `SemanticChunker`, and `get_chunker`.
- `tests/test_eval_dataset.py`: Unit tests verifying that `data/GoldenDatset.md` parses correctly into 22 valid samples.
- `tests/test_eval_metrics.py`: Unit tests for Hit Rate@K, MRR, and summary calculations.

---

## 5. Implementation Roadmap
1. Update `pyproject.toml` with Ragas & LangChain dependencies and sync environment with `uv`.
2. Implement `src/exocortex/chunking/` module with all 4 strategies and unit tests.
3. Update `src/exocortex/ingestion.py` and `src/exocortex/config.py` to support pluggable chunking strategies.
4. Implement `src/exocortex/eval/dataset.py`, `metrics.py`, `ragas_evaluator.py`, and `benchmark.py`.
5. Run the benchmark suite across the 4 strategies on `Designing Machine Learning Systems`.
6. Generate benchmark report artifact (`docs/benchmark_results.md` & `data/benchmark_results.csv`).
7. Update default `chunking_strategy` in `config.py` with the winning strategy.
