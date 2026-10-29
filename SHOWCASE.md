# DocIntel: numbers and copy you can use

Every number here comes from `eval/results/summary.json` (QASPER test split, 45 arXiv papers, 366 pages, 136 answerable questions with expert-annotated evidence). The baseline is a standard LangChain-style RAG: PyPDF, 1,000-character recursive chunks, dense top-5. Both systems share the embedding model, the top-k, the LLM and the prompt.

Image to attach to the post: [`eval/results/docintel_card.png`](eval/results/docintel_card.png) (1200×627, LinkedIn's link-preview size). Regenerate it with `python eval/make_card.py`.

## Headline numbers

| Metric | Standard RAG | DocIntel | Claim you can make |
|---|---|---|---|
| Hit@5 (answer evidence reaches the LLM) | 74.3% | 90.4% | **63% fewer retrieval misses** (25.7% → 9.6%) |
| Hit@1 (evidence ranked first) | 31.6% | 44.9% | +13.2 pts (p = 0.008) |
| MRR@5 | 0.472 | 0.612 | **+30%** |
| nDCG@5 | 0.414 | 0.599 | **+45%** |
| Evidence recall@5 | 65.6% | 85.9% | +20.3 pts |
| Evidence paragraphs intact in one chunk | 62.5% | 94.7% | +32.2 pts |

**End-to-end answers** (50 random questions, same LLM per pair, graded blind against the expert answers): judge-scored correctness **84.4% vs 78.1%**, fully correct **77.1% vs 68.8%**, false "I don't know" **0% vs 6.2%**. DocIntel was better on 6 questions, standard RAG on 3, and 39 tied. The difference is **not significant** (p = 0.18, n = 48 answerable). The mechanism is clear, though: when standard RAG's retrieval missed the evidence, its answers scored 42% vs DocIntel's 75%.

Use the **retrieval** numbers as the headline: 136 questions, all significant at p < 0.01 (paired bootstrap, 10,000 resamples). The answer-level numbers come from a smaller sample limited by free-tier quota. Quote them with their n.

## Resume bullets

Pick 2–3:

- Built **DocIntel**, an accuracy-first RAG system for research papers: Docling layout parsing, structure-aware chunking, hybrid dense + learned-sparse retrieval with RRF fusion, and cross-encoder reranking (FastAPI, Pinecone, Gemini, Cohere).
- Benchmarked against a standard LangChain RAG on **QASPER** (45 arXiv papers, 136 expert-annotated questions): **cut retrieval misses by 63%**, improved **MRR by 30%** and **nDCG@5 by 45%**, all significant at p < 0.01.
- Built a reproducible evaluation harness: component ablations, a context-size control, paired bootstrap CIs, blinded LLM-as-judge answer grading, and an interactive per-question results dashboard.
- Found that **parsing and chunking, not the retrieval algorithm, drove most of the gain**: evidence kept intact in a chunk rose from 62.5% to 94.7%. Traced a cross-encoder bias toward paper abstracts and shipped an opt-in rank-fusion fix (+6.6 pts Hit@1, +7% MRR, −2.2 pts Hit@5).
- Hardened the system for rate-limited APIs: reranker and LLM fallback chains, per-document query scoping, stale-vector cleanup on re-upload, and page-level citations mapped server-side.

## LinkedIn post (draft)

> I benchmarked my RAG system instead of just demoing it, and the results surprised me.
>
> **DocIntel** answers questions about research papers with page-level citations. I tested it against a standard "chat with your PDF" pipeline (PyPDF + fixed-size chunks + vector search) on **QASPER**, AllenAI's benchmark of real NLP papers with expert-annotated answers and evidence.
>
> Same embedding model, same LLM, same prompt, same top-5. Only the pipeline changed.
>
> 📉 63% fewer retrieval misses (evidence in top-5: 74% → 90%)
> 📈 +30% MRR, +45% nDCG@5
> 🧩 Evidence paragraphs kept intact: 62% → 95%
>
> What I learned:
> 1️⃣ **Parsing matters more than retrieval tricks.** PyPDF scrambles two-column papers and fixed-size chunks cut paragraphs in half. Layout-aware parsing (Docling) with structure-aware chunking did most of the work.
> 2️⃣ **Hybrid search barely helped when searching within one paper.** Lexical matching earns its keep across large corpora, not single documents.
> 3️⃣ **The reranker had a blind spot.** It loved promoting the abstract to #1. Fusing its ranking with the retriever's won back 6.6 points of top-1 accuracy but cost 2 points of top-5 coverage. Since the LLM reads all five passages, I made it an opt-in flag, not the default. Measuring beats intuition.
> 4️⃣ **Accuracy has a cost.** Layout parsing is ~60× slower than PyPDF (2.3 s/page on CPU).
>
> Everything is reproducible: cached, resumable harness, bootstrap confidence intervals, and a dashboard where you can inspect every question.
>
> #RAG #LLM #InformationRetrieval #NLP #MachineLearning

## Project-page blurb

**DocIntel: accuracy-first RAG for research papers.** Layout-aware PDF parsing, section-aware chunking, hybrid dense/sparse retrieval and cross-encoder reranking, with answers cited to the page. On the QASPER benchmark it gets the answer-bearing passage into the LLM's context 90% of the time vs 74% for a standard RAG pipeline: 63% fewer misses, with +30% MRR and +45% nDCG.

## Caveats to keep (they make the claims credible)

- Retrieval is scoped to the paper being asked about, which mirrors "chat with this PDF".
- The benchmark uses Pinecone's `llama-text-embed-v2` for every system. Production can use the same model (`EMBEDDING_PROVIDER=pinecone`) or Gemini embeddings.
- DocIntel's top-5 context is about 50% longer than the 1,000-char baseline. The 2,000-char control shows length alone doesn't explain the gain: DocIntel beats it with 19% less text.
- Docling ingestion is about 2.3 s/page on CPU vs 0.04 s/page for PyPDF.
