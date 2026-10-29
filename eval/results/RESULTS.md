# DocIntel benchmark results

**Dataset:** QASPER (test split): 45 arXiv papers (366 pages), 149 questions (136 answerable with human-annotated evidence, 13 unanswerable).  
**Embedding (all systems):** `llama-text-embed-v2`, top-5 context.

## Retrieval (does the answer-bearing passage reach the LLM?)

| System | Hit@1 | Hit@5 | MRR@5 | Evidence recall@5 | nDCG@5 | Avg context chars |
|---|---|---|---|---|---|---|
| Standard RAG | 31.6% | 74.3% | 0.472 | 65.6% | 0.414 | 4538 |
| Standard RAG (2k chunks) | 40.4% | 85.3% | 0.570 | 81.0% | 0.530 | 8426 |
| + Docling parsing & structure-aware chunks | 47.8% | 86.8% | 0.620 | 81.9% | 0.584 | 6451 |
| + Hybrid dense/sparse retrieval (RRF) | 49.3% | 86.0% | 0.627 | 81.3% | 0.588 | 6810 |
| + Cohere cross-encoder rerank (DocIntel) | 44.9% | 90.4% | 0.612 | 85.9% | 0.599 | 6833 |
| Variant: rerank + rank fusion (opt-in, RERANK_FUSION=1) | 51.5% | 88.2% | 0.655 | 84.3% | 0.623 | 6813 |

DocIntel vs standard RAG (paired bootstrap, 95% CI):

- hit@1: +13.2 pts [+3.7, +22.8], p=0.0076
- hit@5: +16.2 pts [+8.8, +23.5], p=0.0001
- mrr@5: +14.1 pts [+6.9, +21.3], p=0.0001
- recall@5: +20.3 pts [+12.9, +27.4], p=0.0001
- ndcg@5: +18.5 pts [+12.7, +24.4], p=0.0001

## Parsing & chunking fidelity (531 gold evidence paragraphs)

| Chunkset | Recovered anywhere in parsed text | Intact inside one chunk |
|---|---|---|
| pypdf_1000 | 94.0% | 62.5% |
| pypdf_2000 | 94.0% | 76.1% |
| docling | 97.0% | 94.7% |

## End-to-end answers (50 paired questions, blinded LLM judge)

Generators: {'gemini-3-flash-preview': 3, 'gemini-3.5-flash-lite': 38, 'gemini-2.5-flash-lite': 9} · Judges: {'claude-opus-5.5 (blinded)': 50}

| System | Judge correctness | Strict accuracy | Answer F1 | Faithful | Citation precision | Correct abstention (unanswerable) | False abstention |
|---|---|---|---|---|---|---|---|
| Standard RAG | 78.1% | 68.8% | 30.6% | 100.0% | 74.8% | 50.0% | 6.2% |
| DocIntel | 84.4% | 77.1% | 30.6% | 96.0% | 73.7% | 100.0% | 0.0% |

Head-to-head (answerable): {'tie': 39, 'docintel': 6, 'naive': 3}

Judge-correctness difference: +6.2 pts [-2.1, +14.6], p=0.175 (paired bootstrap).

When standard RAG's top-5 **missed** the evidence (n=6): standard 41.7% vs DocIntel 75.0%. When it **found** it (n=42): 83.3% vs 85.7%.

## Ingestion

- pypdf_1000: 1871 chunks, median 966 chars, 0.04s/page
- pypdf_2000: 937 chunks, median 1958 chars, 0.04s/page
- docling: 1252 chunks, median 1181 chars, 2.32s/page

## Latency (median ms)

- dense_embed_ms: 408
- dense_search_ms: 1188
- sparse_embed_ms: 370
- sparse_search_ms: 946
- rerank_ms: 500
