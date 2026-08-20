# Known Limitations & Future Roadmap

## Current Limitations

This document lists all known limitations of the basic Exocortex RAG system (v0.1.0).
Each limitation includes an explanation of **why** it's a limitation and a **suggested
improvement** for future development.

---

### 1. No OCR Support (Scanned PDFs)

**Impact:** High — many ebooks and documents are scanned images.

**Current behavior:** The system uses PyMuPDF to extract text from PDFs that have an
embedded text layer. Scanned PDFs (image-only) will produce no text, and ingestion
will fail with a `ValueError`.

**Suggested improvement:**
- Integrate an OCR engine (e.g., Tesseract via `pytesseract`, or `easyocr`)
- Add a fallback: if PyMuPDF extracts no text from a page, run OCR
- Consider GPU-accelerated OCR for large documents

**Estimated effort:** Medium (new module + fallback logic)

---

### 2. English Only (No Multi-Language Support)

**Impact:** Medium — limits use to English ebooks.

**Current behavior:** The embedding model (`qwen3-embedding:0.6b`) supports multiple
languages, but the system prompt, query instruction prefix, and chunking strategy are
optimized for English only.

**Suggested improvement:**
- Test and validate with non-English ebooks
- Adjust the query instruction prefix for different languages
- Consider language detection and dynamic prompt switching
- Potentially use a multilingual embedding model

**Estimated effort:** Low–Medium (mostly prompt changes + testing)

---

### 3. No Reranker (Retrieval May Be Suboptimal)

**Impact:** Medium — retrieval quality is limited by embedding similarity only.

**Current behavior:** Retrieval relies solely on cosine similarity between the query
embedding and chunk embeddings. The small embedding model (0.6B) may produce tighter
vector clusters, making it harder to distinguish relevant from irrelevant chunks.

**Suggested improvement:**
- Add a cross-encoder reranker (e.g., `qwen3-reranker` or a BERT-based model)
- Pipeline: retrieve top-20 with embeddings → rerank to top-5 with cross-encoder
- This significantly improves answer quality

**Estimated effort:** Medium (new reranker module + pipeline modification)

---

### 4. No Conversation History (Stateless Queries)

**Impact:** Medium — users cannot ask follow-up questions.

**Current behavior:** Each query is independent. The system has no memory of previous
questions or answers. Users cannot say "tell me more about that" or "what about X
mentioned earlier?"

**Suggested improvement:**
- Add a conversation session with message history
- Include previous Q&A pairs in the LLM context
- Implement a sliding window to manage context length
- Store sessions in a database (SQLite or Redis)

**Estimated effort:** Medium (session management + context window logic)

---

### 5. No Authentication/Authorization

**Impact:** Low (for demo) / High (for production).

**Current behavior:** All API endpoints are publicly accessible. No user accounts,
API keys, or rate limiting.

**Suggested improvement:**
- Add API key authentication (simple header-based)
- Implement OAuth2 for multi-user scenarios
- Add rate limiting per API key
- Separate document access by user/team

**Estimated effort:** Medium (auth middleware + user management)

---

### 6. Basic Chunking (May Split Semantic Units)

**Impact:** Medium — can degrade answer quality.

**Current behavior:** Fixed-size character chunking with overlap. This can split
sentences, paragraphs, or semantic units mid-thought, leading to chunks that lack
complete context.

**Suggested improvement:**
- **Sentence-aware chunking:** Split on sentence boundaries, then group sentences
  into chunks of target size
- **Semantic chunking:** Use embedding similarity to detect topic boundaries
- **Recursive chunking:** LangChain-style recursive character text splitting with
  prioritized separators (`\n\n` > `\n` > `. ` > ` `)
- **Agentic chunking:** Use LLM to determine optimal chunk boundaries

**Estimated effort:** Low–Medium (sentence-aware) to High (semantic/agentic)

---

### 7. No Table/Image/Code Block Handling

**Impact:** Medium — ebooks often contain tables, figures, and code.

**Current behavior:** PyMuPDF extracts text only. Tables are flattened into plain
text (losing structure). Images are ignored. Code blocks lose formatting.

**Suggested improvement:**
- Use PyMuPDF's table extraction (available in recent versions)
- Extract images and use vision models for description
- Detect and preserve code block formatting
- Store different content types with metadata tags

**Estimated effort:** High (significant parsing improvements)

---

### 8. No Production-Grade Monitoring/Logging

**Impact:** Low (for demo) / High (for production).

**Current behavior:** Basic Python `logging` to console. No structured logging,
no metrics collection, no alerting.

**Suggested improvement:**
- Add structured logging (JSON format) with `structlog`
- Integrate OpenTelemetry for distributed tracing
- Add Prometheus metrics (request latency, embedding time, LLM token usage)
- Set up health check monitoring with alerting
- Log query/answer pairs for quality evaluation

**Estimated effort:** Medium (structured logging) to High (full observability)

---

### 9. Embedding Model Limitations (0.6B)

**Impact:** Medium — smaller model = less nuanced embeddings.

**Current behavior:** `qwen3-embedding:0.6b` is a lightweight model. While fast and
resource-efficient, it produces less nuanced embeddings compared to larger models
(4B, 8B). For large corpora, embeddings may cluster tightly, reducing retrieval
discrimination.

**Suggested improvement:**
- Upgrade to `qwen3-embedding:4b` or `qwen3-embedding:8b` for better quality
- Use Matryoshka Representation Learning (MRL) to test reduced dimensions
- Benchmark retrieval quality across model sizes with your specific ebooks

**Estimated effort:** Low (model swap in config)

---

### 10. No Duplicate Document Detection

**Impact:** Low–Medium.

**Current behavior:** If the same PDF is uploaded twice, it will be re-indexed
with the same document ID (upsert), so chunks are replaced, not duplicated.
However, if the same content is uploaded with a different filename, it creates
a separate set of chunks.

**Suggested improvement:**
- Hash file content (not just filename) for document ID
- Check content hash before ingestion to skip duplicates
- Show warning to user if similar content already exists

**Estimated effort:** Low (content-based hashing)

---

## Future Roadmap

### Short-term (v0.2)
- [ ] Sentence-aware chunking
- [ ] Content-based document deduplication
- [ ] Conversation history (basic session)
- [ ] Upgrade to larger embedding model option

### Medium-term (v0.3)
- [ ] Cross-encoder reranker
- [ ] Table extraction from PDFs
- [ ] API key authentication
- [ ] Structured logging

### Long-term (v1.0)
- [ ] OCR support for scanned PDFs
- [ ] Multi-language support
- [ ] Hybrid search (keyword + vector)
- [ ] Production monitoring (OpenTelemetry)
- [ ] Multi-user with document isolation
- [ ] Streaming LLM responses (SSE)
- [ ] Migrate to Qdrant/Milvus for scale
