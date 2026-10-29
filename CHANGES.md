# Changes from the original DocIntel

Compared against the original project in `D:\nergy-assignment-1`, which is untouched.

## Did the architecture change?

**The core pipeline did not change.** It is still:

> Docling parsing → HybridChunker (512 tokens) → dense + sparse vectors in Pinecone → top-30 from each → weighted RRF (0.7 / 0.3, k = 60) → Cohere `rerank-v3.5` → top 5 → Gemini 2.5 Flash structured answer

The models, parameters and defaults in code are the same.

**What changed is around that core:**

1. A configurable embedding provider. This copy's `.env` uses the new option, so it runs on a different embedding model and index than the original.
2. Fallbacks when a rate-limited service fails: reranker and LLM.
3. Per-document search scoping and document management (list, delete, clean re-upload).
4. The citation format the LLM is asked to write.
5. API and UI additions.
6. A new evaluation harness (`eval/`).
7. One opt-in retrieval variant (rank fusion), off by default.

## Before and after

**Before**

```
PDF ─► Docling ─► HybridChunker ─┬─► Gemini embeddings (3072-d) ──► Pinecone dense index
                                 └─► Pinecone sparse vectors ─────► Pinecone sparse index   (default namespace)

Question ─► dense top-30 + sparse top-30 ─► RRF ─► Cohere rerank ─► top 5 ─► Gemini 2.5 Flash
                                                   (on error: fused order, silently)
         ─► { answer with [file.pdf, p.X] tags, insights, next_questions, citations = all 5 chunks }
```

**After** (`+` marks what's new)

```
PDF ─► + validate (.pdf name, %PDF- header, ≤ 50 MB)
    ─► Docling ─► HybridChunker
    ─┬─► dense embeddings: Gemini (3072-d, default)  OR  + Pinecone llama-text-embed-v2 (1024-d)
     └─► sparse vectors
    ─► Pinecone, + optional namespace
    ─► + manifest.json records each file's chunk IDs; re-uploading a changed file deletes its stale chunks

Question (+ optional list of filenames)
    ─► dense top-30 + sparse top-30, + filtered to those files ─► RRF
    ─► rerank: Cohere ─► + Pinecone bge-reranker-v2-m3 ─► fused order
       (+ optional RERANK_FUSION: blend rerank order with RRF order)
    ─► top 5 ─► Gemini 2.5 Flash ─► + fallback models if rate-limited
    ─► { answer with + [n] citations, insights, next_questions,
         citations (+ `cited` flag, + snippet), + model, + reranker, + timings }
```

## What changed, component by component

| # | Component | Before | After | Default changed? | Why |
|---|---|---|---|---|---|
| 1 | Dense embeddings | Gemini `gemini-embedding-2-preview`, 3072-d, hard-coded | `EMBEDDING_PROVIDER` = `gemini` (default) or `pinecone` (`llama-text-embed-v2`, 1024-d) | Code: no. **This copy's `.env`: yes**, it uses `pinecone` with a new 1024-d index `docintel-dense` | The Gemini free tier embeds 1,000 chunks a day, about 20 papers. The benchmark needed about 5,000. |
| 2 | Pinecone namespace | Default namespace | `PINECONE_NAMESPACE` (this copy: `docintel`) | Code: no | Keeps corpora apart inside shared indexes. Your original project's sparse vectors are untouched. |
| 3 | Reranker failure | Fell back to fused order without saying so | Cohere → Pinecone `bge-reranker-v2-m3` → fused order. The response names the reranker used. | Only when Cohere fails | The Cohere trial key allows 10 calls per minute. |
| 4 | Rank fusion | — | `RERANK_FUSION=1` blends the reranker's order with the RRF order | No (off) | The benchmark found the cross-encoder often puts the abstract first. Fusion: Hit@1 44.9% → 51.5%, Hit@5 90.4% → 88.2%. Neither is significant. |
| 5 | Generation failure | The SDK retried, then the API returned HTTP 500 | Tries `GENERATION_FALLBACK_MODELS` in order (default `gemini-2.5-flash-lite`, `gemini-3.1-flash-lite`). The response names the model that answered. | Only on 429/503 | The Gemini free tier allows 5 requests per minute per model. |
| 6 | Citations | Prompt asked for `[filename, p.X]`. All 5 retrieved chunks were returned as "citations". | Prompt asks for `[1]`, `[2][4]`. The server maps numbers to file and page. Each source is flagged `cited` and carries a 900-char snippet. | **Yes**: the answer text now contains `[n]` | The model wrote malformed tags like `[1, p.3]` that couldn't be traced to a source. |
| 7 | Search scope | Always searched every document | Optional `filenames` → Pinecone metadata filter on both indexes | No (omit = all documents) | Lets you ask about one paper. The benchmark runs this way. |
| 8 | Document lifecycle | No list or delete. Re-uploading a changed file left its old chunks behind. | `uploads/manifest.json`, `GET /documents`, `DELETE /documents/{filename}`, stale-chunk cleanup on re-upload | — | Managing a library from the UI. |
| 9 | Upload validation | `.pdf` extension only | + `%PDF-` header check, 50 MB limit | — | Rejects renamed non-PDFs early. |
| 10 | `/query` response | `answer, insights, next_questions, citations` | + `model, reranker, timings`; citations + `cited, snippet` | Additive | The UI shows where an answer came from and how long it took. |
| 11 | Retrieval chain input | `chain.invoke("question")` | `chain.invoke({"query": ..., "filenames": None})` | Internal API | Needed for document scoping. |
| 12 | UI | Standalone `index.html` opened from disk, pointed at `localhost:8000` | Rebuilt and served by the API at `/`. Three tabs: **Ask**, **Benchmark**, **Metrics** (plain-language metric guide). | — | You asked for a new UI and a metrics explanation page. |
| 13 | Evaluation | None | `eval/`: QASPER benchmark, ablations, bootstrap CIs, blind answer grading, results card | — | v1 had no metrics. |

## Unchanged

- **Docling**: OCR off, table structure on, CPU.
- **HybridChunker**: 512 tokens (MiniLM tokenizer), `merge_peers`, headings embedded with each chunk.
- **Chunk IDs**: `sha1(filename | section | content)`, deterministic.
- **Sparse model**: `pinecone-sparse-english-v0`.
- **Retrieval funnel**: `RETRIEVE_K = 30`, `FINAL_K = 5`, RRF `k = 60`, dense/sparse weights `0.7 / 0.3`.
- **Primary reranker**: Cohere `rerank-v3.5`.
- **Generation**: `gemini-2.5-flash`, temperature 0.1, structured `RAGAnswer` (answer, insights, next_questions).
- **App structure**: the FastAPI shared-engine pattern (one `RAGEngine` built at startup) and CORS settings.

## Files

| File | Change |
|---|---|
| `backend/hybrid_search.py` | Embedding provider, namespace, filename filter, `delete_chunks`, reranker fallback, optional rank fusion, dict chain input |
| `backend/rag_engine.py` | Manifest, list/delete, stale-chunk cleanup, scoped retrieve, numbered citations, model fallback, timings |
| `backend/api.py` | `GET /`, `GET /documents`, `DELETE /documents/{filename}`, `filenames` on `/query`, `GET /benchmark`, upload validation |
| `backend/.env` | This copy only: `EMBEDDING_PROVIDER=pinecone`, `PINECONE_INDEX_NAME=docintel-dense`, `PINECONE_NAMESPACE=docintel` |
| `backend/.env.example` | New; documents every setting |
| `backend/requirements.txt` | Adds the benchmark's dependencies (pypdf, pymupdf, pandas…) |
| `frontend/index.html` | Rewritten: Ask / Benchmark / Metrics |
| `README.md` | Rewritten. The old one described an earlier build (MarkdownHeaderTextSplitter, Gemini 1.5 Flash, text-embedding-004, BM25, top-50). |
| `FLOWCHART.md` | Query flow updated |
| `.gitignore` | Ignores benchmark caches and downloaded PDFs |
| New | `eval/` (harness and results), `SHOWCASE.md` (resume/LinkedIn copy), `CHANGES.md` (this file) |

## Compatibility notes

- **Running exactly like the original:** set `EMBEDDING_PROVIDER=gemini`, `PINECONE_INDEX_NAME=hybrid-search`, and leave `PINECONE_NAMESPACE` empty. The original 3072-d index and its data still work.
- **Embeddings from the two providers aren't interchangeable.** Switching provider means re-uploading documents into the matching index.
- **Documents ingested before this version** are still searchable, but they aren't in `manifest.json`, so they don't appear in `/documents` and can't be deleted through the API.
- **Code that parses old-style `[file.pdf, p.X]` tags** should read `[n]` and look it up in `citations[n-1]`.

## Changes outside this folder

- **Pinecone:**
  - New serverless index `docintel-dense` (1024-d, cosine).
  - Vectors written to namespace `docintel` of the existing sparse index `sprase-index`.
  - Three test papers ingested there (`1912.11637`, `1610.06510`, `1911.09709`). Delete them from the app's library or drop the index when you no longer need them.
- **`alfred-landing/.claude/launch.json`:** a `docintel` entry, so the app can be started from the desktop app.
- **`D:\DocIntel\.venv`:** Python 3.11. The global Python 3.14 can't import Docling.
