"""
DocIntel vs. standard RAG — benchmark runner
============================================
Every stage is cached (eval/cache/) and resumable, because the API keys are
free-tier / trial keys. Run the stages in order:

    python eval/bench.py ingest     # parse + chunk every paper with each pipeline
    python eval/bench.py embed      # dense (Gemini) + sparse (Pinecone) vectors
    python eval/bench.py retrieve   # run the retrieval configs
    python eval/bench.py generate   # paired answers: standard RAG vs DocIntel
    python eval/bench.py judge      # blinded LLM-as-judge on each answer pair
    python eval/bench.py latency    # per-stage API latency on a sample
    python eval/bench.py report     # metrics + bootstrap CIs → eval/results/

Systems under test (ablation ladder — each rung adds one DocIntel component):

    naive          PyPDF → 1000-char recursive chunks (200 overlap) → dense top-5
                   (the standard LangChain "chat with your PDF" RAG)
    naive_2k       same, 2000-char chunks — controls for context size
    docling_dense  Docling layout parsing + HybridChunker → dense top-5
    docling_hybrid + Pinecone sparse retriever, weighted RRF → top-5
    docintel       + Cohere rerank-v3.5 over the fused top-30 → top-5  (production)

All configs use the SAME embedding model (llama-text-embed-v2), the same top-k, and — for generation —
the same LLM and the same prompt. Only the pipeline differs.

Retrieval is scoped to the paper the question is about (QASPER questions say
"they"/"this work"), mirroring a user chatting with one document. Vector search
runs as exact in-memory search over the same vectors Pinecone would store, so
the benchmark never writes to the production indexes.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import re
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

EVAL_DIR = Path(__file__).resolve().parent
ROOT = EVAL_DIR.parent
BACKEND = ROOT / "backend"
sys.path[:0] = [str(EVAL_DIR), str(BACKEND)]

from dotenv import load_dotenv  # noqa: E402

load_dotenv(BACKEND / ".env")

from cache import Cache, key_of  # noqa: E402
import textmatch as tm  # noqa: E402

DATA_PATH = EVAL_DIR / "data" / "qasper_subset.json"
PAPERS_DIR = EVAL_DIR / "data" / "papers"
WORK = EVAL_DIR / "cache"
RESULTS = EVAL_DIR / "results"
WORK.mkdir(exist_ok=True)
RESULTS.mkdir(exist_ok=True)

FINAL_K = 5
RETRIEVE_K = 30
# One embedding model for EVERY system, so differences come from the pipeline.
# Pinecone-hosted (same account as the indexes); Gemini's free tier caps
# embeddings at 1,000/day, far below what 5 pipelines x 45 papers need.
DENSE_MODEL = "llama-text-embed-v2"
SEED = 13

CHUNKSETS = {
    "pypdf_1000": {"parser": "pypdf", "size": 1000, "overlap": 200},
    "pypdf_2000": {"parser": "pypdf", "size": 2000, "overlap": 200},
    "docling": {"parser": "docling"},
}

CONFIGS = {
    "naive": {"chunks": "pypdf_1000", "sparse": False, "rerank": False,
              "label": "Standard RAG"},
    "naive_2k": {"chunks": "pypdf_2000", "sparse": False, "rerank": False,
                 "label": "Standard RAG (2k chunks)"},
    "docling_dense": {"chunks": "docling", "sparse": False, "rerank": False,
                      "label": "+ Docling parsing & structure-aware chunks"},
    "docling_hybrid": {"chunks": "docling", "sparse": True, "rerank": False,
                       "label": "+ Hybrid dense/sparse retrieval (RRF)"},
    "docintel": {"chunks": "docling", "sparse": True, "rerank": True,
                 "label": "+ Cohere cross-encoder rerank (DocIntel)"},
    # Pre-registered follow-up (not tuned): fuse the reranker's order with the
    # first-stage fused order via RRF (k=60), so the cross-encoder's preference
    # for abstract-like chunks can't override strong first-stage evidence.
    "docintel_rrf": {"chunks": "docling", "sparse": True, "rerank": "fusion",
                     "label": "Variant: rerank + rank fusion (opt-in, RERANK_FUSION=1)"},
}

GEN_CONFIGS = ["naive", "docintel"]
# Which retrieval config backs the "docintel" answers (the production pipeline).
DOCINTEL_RETRIEVAL = "docintel"
# Generator: DocIntel's production model first. If its free-tier daily quota
# runs out, the *pair* (baseline + DocIntel answer) is regenerated on the next
# model, so every comparison is always generator-matched.
GEN_MODELS = ["gemini-2.5-flash", "gemini-3-flash-preview", "gemini-3.5-flash",
              "gemini-3.5-flash-lite", "gemini-2.5-flash-lite", "gemini-3.6-flash",
              "gemini-3.7-flash", "gemini-3.8-flash", "gemini-3.1-flash-lite"]
# Judge: a different (newer) model family than the generator where possible.
JUDGE_MODELS = ["gemini-3-flash-preview", "gemini-3.1-flash-lite", "gemini-3.5-flash-lite",
                "gemini-2.5-flash-lite"]

cache = Cache()


def load_dataset() -> dict:
    return json.loads(DATA_PATH.read_text(encoding="utf-8"))


def _read(path: Path, default):
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else default


def _write(path: Path, obj) -> None:
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(obj, ensure_ascii=False, indent=1), encoding="utf-8")
    tmp.replace(path)


# ─────────────────────────────────────────────────────────────────────────────
# Rate limiting / retries (free-tier friendly)
# ─────────────────────────────────────────────────────────────────────────────
class DailyQuotaExhausted(RuntimeError):
    pass


_last_call: dict[str, float] = {}


def throttle(bucket: str, min_interval: float) -> None:
    wait = _last_call.get(bucket, 0.0) + min_interval - time.monotonic()
    if wait > 0:
        time.sleep(wait)
    _last_call[bucket] = time.monotonic()


def with_retry(fn, what: str, bucket: str, min_interval: float, max_tries: int = 6):
    delay = 20.0
    for attempt in range(max_tries):
        throttle(bucket, min_interval)
        try:
            return fn()
        except Exception as exc:  # noqa: BLE001
            msg = str(exc)
            if "PerDay" in msg or "per day" in msg.lower():
                raise DailyQuotaExhausted(msg[:300]) from exc
            transient = any(s in msg for s in (
                "429", "RESOURCE_EXHAUSTED", "TooManyRequests", "rate limit", "quota",
                "503", "UNAVAILABLE", "overloaded", "500", "INTERNAL", "timed out",
                "Timeout", "Connection", "high demand"))
            if not transient or attempt == max_tries - 1:
                raise
            print(f"      {what}: {msg[:90]!r} → retry in {delay:.0f}s")
            time.sleep(delay)
            delay = min(delay * 1.7, 180)
    raise RuntimeError("unreachable")


# ─────────────────────────────────────────────────────────────────────────────
# Stage 1 — ingest
# ─────────────────────────────────────────────────────────────────────────────
def _ingest_pypdf(pdf: Path, size: int, overlap: int) -> list[dict]:
    """Exactly what LangChain's PyPDFLoader + RecursiveCharacterTextSplitter do."""
    from pypdf import PdfReader
    from langchain_core.documents import Document
    from langchain_text_splitters import RecursiveCharacterTextSplitter

    reader = PdfReader(str(pdf))
    pages = [
        Document(page_content=page.extract_text() or "", metadata={"page": i + 1})
        for i, page in enumerate(reader.pages)
    ]
    splitter = RecursiveCharacterTextSplitter(chunk_size=size, chunk_overlap=overlap)
    return [
        {"text": d.page_content, "page_start": d.metadata["page"], "page_end": d.metadata["page"],
         "section": ""}
        for d in splitter.split_documents(pages) if d.page_content.strip()
    ]


_converter = None


def _ingest_docling(pdf: Path) -> list[dict]:
    """DocIntel's production ingest path (hybrid_search.convert_pdf + chunk_documents)."""
    global _converter
    import hybrid_search as hs

    if _converter is None:
        _converter = hs._make_docling_converter()
    name, dl_doc = hs.convert_pdf(pdf, converter=_converter)
    return [
        {"text": d.page_content, "page_start": d.metadata["page_start"],
         "page_end": d.metadata["page_end"], "section": d.metadata["section"]}
        for d in hs.chunk_documents([(name, dl_doc)])
    ]


def stage_ingest(ds: dict, only: str | None = None) -> None:
    for name, spec in CHUNKSETS.items():
        if only and name != only:
            continue
        out = WORK / f"chunks_{name}.json"
        state = _read(out, {"chunks": [], "timing": {}})
        todo = [p for p in ds["papers"] if p["paper_id"] not in state["timing"]]
        print(f"\n[{name}] {len(todo)} papers to ingest")
        for i, p in enumerate(todo, 1):
            pdf = PAPERS_DIR / p["filename"]
            t0 = time.perf_counter()
            if spec["parser"] == "pypdf":
                chunks = _ingest_pypdf(pdf, spec["size"], spec["overlap"])
            else:
                chunks = _ingest_docling(pdf)
            dt = time.perf_counter() - t0
            for j, c in enumerate(chunks):
                c.update({"id": f"{name}:{p['paper_id']}:{j}", "paper_id": p["paper_id"],
                          "filename": p["filename"]})
            state["chunks"].extend(chunks)
            state["timing"][p["paper_id"]] = {"seconds": round(dt, 3), "pages": p["pages"],
                                             "chunks": len(chunks)}
            _write(out, state)
            print(f"  {i}/{len(todo)} {p['paper_id']}: {len(chunks)} chunks in {dt:.1f}s")


def load_chunks(name: str) -> list[dict]:
    return _read(WORK / f"chunks_{name}.json", {"chunks": []})["chunks"]


# ─────────────────────────────────────────────────────────────────────────────
# Stage 2 — embeddings (dense: Gemini, sparse: Pinecone hosted model)
# ─────────────────────────────────────────────────────────────────────────────
def _dense_ns(kind: str) -> str:
    return f"dense:{DENSE_MODEL}:{kind}"


def dense_embed(texts: list[str], kind: str) -> None:
    """kind = 'doc' (input_type=passage) or 'query' (input_type=query)."""
    import hybrid_search as hs
    from pinecone import Pinecone

    ns = _dense_ns(kind)
    todo = sorted({t for t in texts if cache.get_vec(ns, key_of(t)) is None})
    print(f"  dense/{kind}: {len(todo)} texts to embed")
    if not todo:
        return
    pc = Pinecone(api_key=hs.PINECONE_API_KEY)
    input_type = "passage" if kind == "doc" else "query"
    for i in range(0, len(todo), 48):
        batch = todo[i:i + 48]
        resp = with_retry(
            lambda: pc.inference.embed(model=DENSE_MODEL, inputs=batch,
                                       parameters={"input_type": input_type, "truncate": "END"}),
            "embed", "embed", min_interval=1.0)
        for t, e in zip(batch, resp.data):
            cache.put_vec(ns, key_of(t), e.values)
        print(f"    {min(i + 48, len(todo))}/{len(todo)}")


def sparse_embed(texts: list[str], input_type: str) -> None:
    import hybrid_search as hs
    from pinecone import Pinecone

    ns = f"sparse:{hs.SPARSE_MODEL}:{input_type}"
    todo = sorted({t for t in texts if cache.get(ns, key_of(t)) is None})
    print(f"  sparse/{input_type}: {len(todo)} texts to embed")
    if not todo:
        return
    pc = Pinecone(api_key=hs.PINECONE_API_KEY)
    for i in range(0, len(todo), 48):
        batch = todo[i:i + 48]
        vecs = with_retry(lambda: hs.embed_sparse(pc, batch, input_type=input_type),
                          "sparse", "sparse", min_interval=1.0)
        for t, v in zip(batch, vecs):
            cache.put(ns, key_of(t), {"indices": list(v["indices"]), "values": list(v["values"])})
        print(f"    {min(i + 48, len(todo))}/{len(todo)}")


def stage_embed(ds: dict) -> None:
    questions = [q["question"] for q in ds["questions"]]
    for name in CHUNKSETS:
        dense_embed([c["text"] for c in load_chunks(name)], "doc")
    dense_embed(questions, "query")
    sparse_embed([c["text"] for c in load_chunks("docling")], "passage")
    sparse_embed(questions, "query")


# ─────────────────────────────────────────────────────────────────────────────
# Stage 3 — retrieval
# ─────────────────────────────────────────────────────────────────────────────
class PaperIndex:
    """Exact dense + sparse search over one paper's chunks (what Pinecone stores)."""

    def __init__(self, chunks: list[dict], with_sparse: bool) -> None:
        import hybrid_search as hs
        self.chunks = chunks
        ns = _dense_ns("doc")
        m = np.stack([cache.get_vec(ns, key_of(c["text"])) for c in chunks])
        self.dense = m / np.linalg.norm(m, axis=1, keepdims=True)
        self.sparse = None
        if with_sparse:
            sns = f"sparse:{hs.SPARSE_MODEL}:passage"
            self.sparse = [cache.get(sns, key_of(c["text"])) for c in chunks]

    def dense_search(self, q: str, k: int) -> list[int]:
        v = cache.get_vec(_dense_ns("query"), key_of(q))
        sims = self.dense @ (v / np.linalg.norm(v))
        return [int(i) for i in np.argsort(-sims)[:k]]

    def sparse_search(self, q: str, k: int) -> list[int]:
        import hybrid_search as hs
        qv = cache.get(f"sparse:{hs.SPARSE_MODEL}:query", key_of(q))
        qmap = dict(zip(qv["indices"], qv["values"]))
        scores = []
        for i, sv in enumerate(self.sparse):
            s = sum(qmap.get(ix, 0.0) * val for ix, val in zip(sv["indices"], sv["values"]))
            scores.append((s, i))
        scores.sort(key=lambda x: -x[0])
        return [i for s, i in scores[:k] if s > 0]


def cohere_rerank(query: str, texts: list[str], top_n: int) -> list[tuple[int, float]]:
    """Same call as production (hybrid_search.make_reranker_fn), cached + trial-key throttled."""
    import cohere
    import hybrid_search as hs

    ns = f"cohere:{hs.RERANK_MODEL}"
    k = key_of(query, str(top_n), *texts)
    hit = cache.get(ns, k)
    if hit is not None:
        return [tuple(x) for x in hit["results"]]
    client = cohere.ClientV2(api_key=hs.COHERE_API_KEY)
    t0 = time.perf_counter()
    resp = with_retry(
        lambda: client.rerank(model=hs.RERANK_MODEL, query=query, documents=texts,
                              top_n=min(top_n, len(texts))),
        "rerank", "cohere", min_interval=6.5)  # trial key: 10 calls / minute
    ms = (time.perf_counter() - t0) * 1000
    results = [(r.index, float(r.relevance_score)) for r in resp.results]
    cache.put(ns, k, {"results": results, "ms": ms})
    return results


def run_config(cfg: dict, index: PaperIndex, q: str) -> list[dict]:
    """Returns the final top-K chunks for one query under one config."""
    import hybrid_search as hs
    from langchain_core.documents import Document

    if not cfg["sparse"]:
        return [index.chunks[i] for i in index.dense_search(q, FINAL_K)]

    def as_docs(ids):
        return [Document(page_content=index.chunks[i]["text"], metadata={"i": i}) for i in ids]

    fused = hs.rrf_fusion({"dense": as_docs(index.dense_search(q, RETRIEVE_K)),
                           "sparse": as_docs(index.sparse_search(q, RETRIEVE_K))})
    fused_ids = [d.metadata["i"] for d in fused]
    if not cfg["rerank"]:
        return [index.chunks[i] for i in fused_ids[:FINAL_K]]
    if cfg["rerank"] == "fusion":
        full = cohere_rerank(q, [index.chunks[i]["text"] for i in fused_ids], len(fused_ids))
        rr_rank = {pos: r for r, (pos, _) in enumerate(full)}
        rr_score = dict(full)
        order = sorted(range(len(fused_ids)),
                       key=lambda pos: -(1 / (61 + pos) + 1 / (61 + rr_rank.get(pos, len(fused_ids)))))
        out = []
        for pos in order[:FINAL_K]:
            c = dict(index.chunks[fused_ids[pos]])
            c["rerank_score"] = round(rr_score.get(pos, 0.0), 4)
            out.append(c)
        return out
    ranked = cohere_rerank(q, [index.chunks[i]["text"] for i in fused_ids], FINAL_K)
    out = []
    for pos, score in ranked:
        c = dict(index.chunks[fused_ids[pos]])
        c["rerank_score"] = round(score, 4)
        out.append(c)
    return out


def stage_retrieve(ds: dict, configs: list[str] | None = None) -> None:
    configs = configs or list(CONFIGS)
    out_path = WORK / "retrieval.json"
    out = _read(out_path, {})
    by_set = {name: defaultdict(list) for name in CHUNKSETS}
    for name in CHUNKSETS:
        for c in load_chunks(name):
            by_set[name][c["paper_id"]].append(c)

    indexes: dict[tuple[str, str, bool], PaperIndex] = {}
    for cname in configs:
        cfg = CONFIGS[cname]
        res = out.setdefault(cname, {})
        print(f"\n[{cname}] {cfg['label']}")
        for i, q in enumerate(ds["questions"], 1):
            if q["qid"] in res:
                continue
            key = (cfg["chunks"], q["paper_id"], cfg["sparse"])
            if key not in indexes:
                indexes[key] = PaperIndex(by_set[cfg["chunks"]][q["paper_id"]], cfg["sparse"])
            top = run_config(cfg, indexes[key], q["question"])
            res[q["qid"]] = [{k: c[k] for k in ("id", "text", "page_start", "page_end", "section",
                                                "filename") if k in c}
                             | ({"rerank_score": c["rerank_score"]} if "rerank_score" in c else {})
                             for c in top]
            if cfg["rerank"]:
                _write(out_path, out)  # rerank calls are slow + precious: checkpoint often
                print(f"  {i}/{len(ds['questions'])}", end="\r")
        _write(out_path, out)
    print("\nretrieval done")


# ─────────────────────────────────────────────────────────────────────────────
# Stage 4 — paired generation (same model + production prompt for both systems)
# ─────────────────────────────────────────────────────────────────────────────
_llms: dict[str, object] = {}
_cooldown: dict[str, float] = {}   # model → monotonic time until which it is skipped


def gemini_json(model: str, system: str, user: str, schema: dict, temperature: float) -> dict:
    """
    One structured-output call to the Gemini REST API.

    Equivalent to production's `PROMPT | llm.with_structured_output(RAGAnswer)`
    (same system prompt, same user template, same temperature, same JSON
    schema) but without the SDK's internal retry/backoff, so an overloaded
    free-tier model fails fast and the next model can take the pair.
    """
    import httpx

    r = httpx.post(
        f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent",
        params={"key": os.environ["GOOGLE_API_KEY"]},
        json={
            "systemInstruction": {"parts": [{"text": system}]},
            "contents": [{"role": "user", "parts": [{"text": user}]}],
            "generationConfig": {"temperature": temperature, "responseMimeType": "application/json",
                                 "responseJsonSchema": schema},
        },
        timeout=120,
    )
    if r.status_code != 200:
        raise RuntimeError(f"{r.status_code} {r.text[:4000]}")
    parts = r.json()["candidates"][0]["content"]["parts"]
    return json.loads("".join(p.get("text", "") for p in parts if not p.get("thought")))


def generate_one(model: str, cfg: str, q: dict, chunks: list[dict]) -> dict:
    from langchain_core.documents import Document
    from rag_engine import SYSTEM_PROMPT, RAGAnswer, _format_context

    docs = [Document(page_content=c["text"], metadata=c) for c in chunks]
    context = _format_context(docs)
    ns = f"gen:{model}"
    k = key_of(cfg, q["qid"], context)
    hit = cache.get(ns, k)
    if hit is not None:
        return hit
    t0 = time.perf_counter()
    user = f"Context:\n{context}\n\nQuestion: {q['question']}"   # production human template
    raw = with_retry(lambda: gemini_json(model, SYSTEM_PROMPT, user, RAGAnswer.model_json_schema(), 0.1),
                     f"gen {model}", f"gen:{model}", min_interval=12.5, max_tries=2)
    res = RAGAnswer(**raw)
    rec = {"answer": res.answer, "insights": res.insights, "next_questions": res.next_questions,
           "ms": round((time.perf_counter() - t0) * 1000), "context_chars": len(context)}
    cache.put(ns, k, rec)
    return rec


def _ret_cfg(cfg: str) -> str:
    return DOCINTEL_RETRIEVAL if cfg == "docintel" else cfg


def stage_generate(ds: dict, limit: int | None = None) -> None:
    """
    One worker per generator model, all pulling from one shuffled question queue
    (each free-tier model has its own per-minute and per-day quota). A worker
    always generates BOTH answers of a pair itself, so every comparison is
    generator-matched; a question that fails goes back on the queue for another
    model. Workers retire when their model's daily quota is gone.
    """
    import queue
    import threading

    retrieval = _read(WORK / "retrieval.json", {})
    out_path = WORK / "generation.json"
    out = _read(out_path, {})
    qs = list(ds["questions"])
    random.Random(SEED).shuffle(qs)  # partial runs are still a uniform random sample
    if limit:
        qs = qs[:limit]
    todo: queue.Queue = queue.Queue()
    for q in qs:
        if q["qid"] not in out:
            todo.put(q)
    lock = threading.Lock()
    print(f"{todo.qsize()} pairs to generate with {len(GEN_MODELS)} model workers")

    def worker(model: str) -> None:
        failures = 0
        while True:
            try:
                q = todo.get_nowait()
            except queue.Empty:
                return
            try:
                pair = {cfg: generate_one(model, cfg, q, retrieval[_ret_cfg(cfg)][q["qid"]])
                        for cfg in GEN_CONFIGS}
            except DailyQuotaExhausted:
                print(f"  {model}: daily quota exhausted — worker retiring")
                todo.put(q)
                return
            except Exception as exc:  # noqa: BLE001 — overloaded / malformed output
                todo.put(q)
                failures += 1
                print(f"  {model}: {str(exc)[:90]!r} (failure {failures})")
                if failures >= 4:
                    print(f"  {model}: too many failures — worker retiring")
                    return
                time.sleep(60 * failures)
                continue
            failures = 0
            with lock:
                out[q["qid"]] = {"model": model, "answers": pair}
                _write(out_path, out)
                print(f"  [{len(out)}] {model}  {q['question'][:60]}")

    threads = [threading.Thread(target=worker, args=(m,), daemon=True) for m in GEN_MODELS]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    left = todo.qsize()
    print(f"done: {len(out)} pairs" + (f"; {left} left — rerun tomorrow to resume" if left else ""))


# ─────────────────────────────────────────────────────────────────────────────
# Stage 5 — blinded LLM-as-judge
# ─────────────────────────────────────────────────────────────────────────────
JUDGE_SYSTEM = """You are an expert grader for question answering over scientific papers.
You receive a question about one research paper, reference answers written by expert
annotators (they may differ in wording or detail; matching ANY reference is enough),
and two candidate answers produced by two anonymous systems, each followed by the
context passages that system retrieved from the paper.

Grade EACH candidate independently (do not compare them):
- correctness: "correct" if it conveys the key information of at least one reference
  answer (extra correct detail is fine); "partial" if it gets some key information
  right but is incomplete or mixes in wrong information; "incorrect" if it is wrong,
  irrelevant, or does not answer. If the question is UNANSWERABLE from the paper
  (no references given), "correct" means the candidate says the information is not
  available, and "incorrect" means it asserts an answer.
- abstained: true if the candidate says it cannot answer / the context lacks the
  information.
- faithful: true if every factual claim in the candidate is supported by ITS OWN
  context passages (citations aside). An abstention is faithful.
Be strict and consistent. Keep rationales to one sentence."""

JUDGE_HUMAN = """QUESTION: {question}

REFERENCE ANSWERS:
{references}

=== CANDIDATE 1 ===
Answer: {a1}

Context retrieved by candidate 1:
{c1}

=== CANDIDATE 2 ===
Answer: {a2}

Context retrieved by candidate 2:
{c2}"""


def _verdict_model():
    from typing import Literal

    from pydantic import BaseModel, Field

    class Grade(BaseModel):
        rationale: str = Field(description="One sentence justifying the grade.")
        correctness: Literal["correct", "partial", "incorrect"]
        abstained: bool
        faithful: bool

    class Verdict(BaseModel):
        candidate_1: Grade
        candidate_2: Grade

    return Verdict


def judge_once(model: str, payload: dict) -> dict:
    Verdict = _verdict_model()
    raw = gemini_json(model, JUDGE_SYSTEM, JUDGE_HUMAN.format(**payload), Verdict.model_json_schema(), 0.0)
    v = Verdict(**raw)
    return {"candidate_1": v.candidate_1.model_dump(), "candidate_2": v.candidate_2.model_dump()}


def _ctx(chunks: list[dict]) -> str:
    return "\n\n".join(f"[{i}] (p.{c['page_start']}) {c['text']}" for i, c in enumerate(chunks, 1))


def stage_judge(ds: dict) -> None:
    gens = _read(WORK / "generation.json", {})
    retrieval = _read(WORK / "retrieval.json", {})
    out_path = WORK / "judgments.json"
    out = _read(out_path, {})
    qmap = {q["qid"]: q for q in ds["questions"]}
    exhausted: set[str] = set()
    todo = [qid for qid in gens if qid not in out]
    print(f"{len(todo)} answer pairs to judge")
    for n, qid in enumerate(todo, 1):
        q, g = qmap[qid], gens[qid]
        order, payload = _judge_payload(q, g, retrieval)
        for model in JUDGE_MODELS:
            if model in exhausted or model == g["model"]:
                continue
            ns = f"judge:{model}"
            k = key_of(json.dumps(payload, sort_keys=True))
            hit = cache.get(ns, k)
            try:
                if hit is None:
                    hit = with_retry(lambda: judge_once(model, payload),
                                     f"judge {model}", f"gen:{model}", min_interval=12.5, max_tries=2)
                    cache.put(ns, k, hit)
            except DailyQuotaExhausted:
                print(f"  daily quota exhausted for judge {model}; falling back")
                exhausted.add(model)
                continue
            except Exception as exc:  # noqa: BLE001
                print(f"  judge {model} failed on {qid}: {str(exc)[:120]}")
                continue
            out[qid] = {"judge": model,
                        order[0]: hit["candidate_1"], order[1]: hit["candidate_2"]}
            _write(out_path, out)
            print(f"  [{n}/{len(todo)}] judged by {model}")
            break
        else:
            if len(exhausted) >= len(JUDGE_MODELS) - 1:
                print("Judge models exhausted for today — rerun to resume.")
                break


def _judge_payload(q: dict, g: dict, retrieval: dict) -> tuple[list[str], dict]:
    flip = random.Random(f"{SEED}:{q['qid']}").random() < 0.5  # blind the position
    order = GEN_CONFIGS[::-1] if flip else GEN_CONFIGS
    refs = "\n".join(f"- {a}" for a in q["answers"]) if q["answerable"] else \
        "(none — every annotator marked this question UNANSWERABLE from the paper)"
    return order, {
        "question": q["question"], "references": refs,
        "a1": g["answers"][order[0]]["answer"], "c1": _ctx(retrieval[_ret_cfg(order[0])][q["qid"]]),
        "a2": g["answers"][order[1]]["answer"], "c2": _ctx(retrieval[_ret_cfg(order[1])][q["qid"]]),
    }


def stage_judge_export(ds: dict) -> None:
    """
    Offline judging path for when the judge model's API quota is gone: writes
    anonymised packets (same rubric, same randomised candidate order as the API
    judge) for an external LLM judge. The order key is written to a separate
    file so the grader never sees which system produced which candidate.
    """
    gens = _read(WORK / "generation.json", {})
    retrieval = _read(WORK / "retrieval.json", {})
    judged = _read(WORK / "judgments.json", {})
    qmap = {q["qid"]: q for q in ds["questions"]}
    packets, key = [], {}
    for qid, g in gens.items():
        if qid in judged:
            continue
        order, p = _judge_payload(qmap[qid], g, retrieval)
        key[qid] = order
        packets.append(f"## {qid}\n\n" + JUDGE_HUMAN.format(**p))
    (WORK / "judge_packets.md").write_text(JUDGE_SYSTEM + "\n\n" + "\n\n---\n\n".join(packets), encoding="utf-8")
    _write(WORK / "judge_key.json", key)
    print(f"exported {len(packets)} packets → {WORK / 'judge_packets.md'}")


def stage_judge_import(judge_name: str) -> None:
    """Merge grades from cache/external_grades.json ({qid: {candidate_1, candidate_2}})."""
    grades = _read(WORK / "external_grades.json", {})
    key = _read(WORK / "judge_key.json", {})
    out_path = WORK / "judgments.json"
    out = _read(out_path, {})
    n = 0
    for qid, v in grades.items():
        if qid in out or qid not in key:
            continue
        order = key[qid]
        out[qid] = {"judge": judge_name, order[0]: v["candidate_1"], order[1]: v["candidate_2"]}
        n += 1
    _write(out_path, out)
    print(f"imported {n} judgments from {judge_name}")


# ─────────────────────────────────────────────────────────────────────────────
# Stage 6 — per-stage latency probe (live API calls on a small sample)
# ─────────────────────────────────────────────────────────────────────────────
def stage_latency(ds: dict, n: int = 8) -> None:
    import cohere
    import hybrid_search as hs
    from pinecone import Pinecone

    pc = Pinecone(api_key=hs.PINECONE_API_KEY)
    emb = hs.make_embeddings(pc)   # the production provider (EMBEDDING_PROVIDER)
    co = cohere.ClientV2(api_key=hs.COHERE_API_KEY)
    sparse_index = pc.Index(hs.PINECONE_SPARSE_INDEX_NAME)
    dense_index = pc.Index(hs.PINECONE_INDEX_NAME)
    by_paper = defaultdict(list)
    for c in load_chunks("docling"):
        by_paper[c["paper_id"]].append(c["text"])

    rows = []
    qs = [q for q in ds["questions"] if q["answerable"]]
    random.Random(SEED).shuffle(qs)
    for q in qs[:n]:
        r = {}
        t = time.perf_counter(); qv = emb.embed_query(q["question"]); r["dense_embed_ms"] = (time.perf_counter() - t) * 1000
        t = time.perf_counter(); dense_index.query(vector=qv, top_k=RETRIEVE_K, namespace=hs.PINECONE_NAMESPACE); r["dense_search_ms"] = (time.perf_counter() - t) * 1000
        t = time.perf_counter(); sv = hs.embed_sparse(pc, [q["question"]], "query")[0]; r["sparse_embed_ms"] = (time.perf_counter() - t) * 1000
        t = time.perf_counter(); sparse_index.query(sparse_vector=sv, top_k=RETRIEVE_K, namespace=hs.PINECONE_NAMESPACE); r["sparse_search_ms"] = (time.perf_counter() - t) * 1000
        docs = by_paper[q["paper_id"]][:RETRIEVE_K]
        throttle("cohere", 6.5)
        t = time.perf_counter(); co.rerank(model=hs.RERANK_MODEL, query=q["question"], documents=docs, top_n=FINAL_K); r["rerank_ms"] = (time.perf_counter() - t) * 1000
        rows.append({k: round(v, 1) for k, v in r.items()})
        print(rows[-1])
    gens = _read(WORK / "generation.json", {})
    gen_ms = {cfg: [g["answers"][cfg]["ms"] for g in gens.values() if g["model"] == GEN_MODELS[0]]
              for cfg in GEN_CONFIGS}
    _write(WORK / "latency.json", {"samples": rows, "generation_ms": gen_ms})


# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["ingest", "embed", "retrieve", "generate", "judge",
                                      "judge-export", "judge-import", "latency", "report", "all"])
    ap.add_argument("--judge-name", default="external LLM judge (blinded)")
    ap.add_argument("--only", help="ingest: a single chunkset; retrieve: comma-separated configs")
    ap.add_argument("--limit", type=int, help="generate: max questions")
    args = ap.parse_args()
    ds = load_dataset()
    if args.stage in ("ingest", "all"):
        stage_ingest(ds, args.only)
    if args.stage in ("embed", "all"):
        stage_embed(ds)
    if args.stage in ("retrieve", "all"):
        stage_retrieve(ds, args.only.split(",") if args.only else None)
    if args.stage in ("generate", "all"):
        stage_generate(ds, args.limit)
    if args.stage in ("judge", "all"):
        stage_judge(ds)
    if args.stage == "judge-export":
        stage_judge_export(ds)
    if args.stage == "judge-import":
        stage_judge_import(args.judge_name)
    if args.stage == "latency":
        stage_latency(ds)
    if args.stage in ("report", "all"):
        import report
        report.main()
