"""
RAGEngine — the reusable, long-lived RAG instance behind the API
================================================================
This is the object the API endpoints share. It is built ONCE at process startup
(see api.py's lifespan) and reused for every request, so the expensive pieces —
Pinecone client, the embedding client, the Docling converter, the reranker, the
chat LLMs, and the assembled retrieval chain — are initialized a single time
rather than per request.

Why this is cleanly reusable
----------------------------
Both retrievers (dense + sparse) are server-side Pinecone queries, so the engine
holds NO mutable corpus in memory. There is nothing to rebuild on upload and
nothing to rehydrate on restart: an upload just upserts to Pinecone, and the
next query sees it live. The retrieval chain is built once in __init__.

The only local state is a small manifest (uploads/manifest.json) recording which
chunk IDs belong to which document, so documents can be listed, deleted, and
re-uploaded without leaving stale vectors behind.

Operations:
    • ingest(path)                 ── POST /upload
    • answer(query, filenames)     ── POST /query   (retrieve → rerank → answer)
    • list_documents()             ── GET  /documents
    • delete_document(filename)    ── DELETE /documents/{filename}
"""

from __future__ import annotations

import asyncio
import json
import os
import re
import sys
import threading
import time
from datetime import datetime, timezone
from pathlib import Path

from dotenv import load_dotenv

# ─────────────────────────────────────────────────────────────────────────────
# Path anchoring — must come before any sibling-module imports
# ─────────────────────────────────────────────────────────────────────────────
_BACKEND_DIR = Path(__file__).resolve().parent   # .../DocIntel/backend
_PROJECT_ROOT = _BACKEND_DIR.parent              # .../DocIntel

for _p in (str(_BACKEND_DIR), str(_PROJECT_ROOT)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

# Load .env from the backend/ directory regardless of CWD.
load_dotenv(_BACKEND_DIR / ".env")

from langchain_google_genai import ChatGoogleGenerativeAI  # noqa: E402
from langchain_core.documents import Document  # noqa: E402
from langchain_core.prompts import ChatPromptTemplate  # noqa: E402
from pydantic import BaseModel, Field  # noqa: E402

import hybrid_search as hs  # noqa: E402


# Generation model — gemini-2.5-flash is fast and plenty for grounded RAG.
GENERATION_MODEL = os.getenv("GENERATION_MODEL", "gemini-2.5-flash")
# Tried in order when the primary model is rate-limited or overloaded (the
# Gemini free tier allows only 5 requests/minute per model). The response says
# which model answered.
GENERATION_FALLBACKS = [
    m.strip()
    for m in os.getenv(
        "GENERATION_FALLBACK_MODELS", "gemini-2.5-flash-lite,gemini-3.1-flash-lite"
    ).split(",")
    if m.strip()
]

# Read-only mode (public demo / serverless): no uploads or deletes, and no
# Docling. The document list comes from a committed library file instead of
# the local manifest that uploads maintain.
READ_ONLY = os.getenv("DOCINTEL_READ_ONLY", "").lower() in ("1", "true", "yes")
MANIFEST_PATH = Path(
    os.getenv("DOCINTEL_LIBRARY")
    or (_PROJECT_ROOT / "deploy" / "library.json" if READ_ONLY else hs.UPLOADS_DIR / "manifest.json")
)


class RAGAnswer(BaseModel):
    """Structured generation output: all components returned in one shot."""

    answer: str = Field(
        description=(
            "The direct answer to the user's question, grounded ONLY in the "
            "provided context. Cite sources inline by their block number, e.g. "
            "[1] or [2][4]. If the context does not contain the answer, say you "
            "don't know."
        )
    )
    insights: list[str] = Field(
        default_factory=list,
        description=(
            "Additional insights, implications, or concrete next steps derived "
            "from the documents. Each item is one short, self-contained point, "
            "grounded in the context and cited by block number, e.g. [3]. Empty "
            "if the context supports none."
        ),
    )
    next_questions: list[str] = Field(
        default_factory=list,
        description=(
            "2-4 suggested follow-up questions the user could ask next, each "
            "answerable from these same documents. Empty if none are natural."
        ),
    )


SYSTEM_PROMPT = (
    "You are a precise assistant that answers ONLY from the provided context.\n"
    "Rules:\n"
    "- Use only the information in the context below. Do not use outside knowledge.\n"
    "- If the context does not contain the answer, say you don't know.\n"
    "- Cite the source after each claim with the number of its context block in\n"
    "  square brackets, e.g. [1] or [2][3]. Never put filenames or page numbers\n"
    "  inside citations; the number is enough.\n"
    "- Be concise and factual.\n"
    "Return a structured object with three components:\n"
    "  1. `answer`         — the direct answer to the question.\n"
    "  2. `insights`       — a list of document-grounded insights or next steps\n"
    "     that go beyond the literal answer (implications, related findings,\n"
    "     actions the documents suggest). Leave empty if the context supports none.\n"
    "  3. `next_questions` — a few natural follow-up questions the user could ask\n"
    "     next, each answerable from these same documents."
)

PROMPT = ChatPromptTemplate.from_messages(
    [
        ("system", SYSTEM_PROMPT),
        ("human", "Context:\n{context}\n\nQuestion: {question}"),
    ]
)


def _page_label(meta: dict) -> str:
    ps, pe = meta.get("page_start", 0), meta.get("page_end", 0)
    return f"p.{ps}" if ps == pe else f"pp.{ps}-{pe}"


def _format_context(docs: list[Document]) -> str:
    """Render retrieved chunks into numbered context blocks (cited as [n])."""
    blocks = []
    for i, doc in enumerate(docs, 1):
        meta = doc.metadata
        source = f"(source: {meta.get('filename', '?')}, {_page_label(meta)})"
        blocks.append(f"[{i}] {source}\n{doc.page_content}")
    return "\n\n".join(blocks)


def _page_number(meta: dict) -> str:
    """Human-friendly page label for a citation: '5' or '5-6'."""
    ps, pe = meta.get("page_start", 0), meta.get("page_end", 0)
    return f"{ps}" if ps == pe else f"{ps}-{pe}"


_BRACKET = re.compile(r"\[([^\[\]]{1,60})\]")


def cited_block_ids(*texts: str) -> set[int]:
    """Block numbers the model cited: [2], [2][4], [2, 4] and stray '[2, p.5]' forms."""
    ids: set[int] = set()
    for text in texts:
        for inner in _BRACKET.findall(text or ""):
            for part in re.split(r"[;,]", inner):
                if part.strip().isdigit():
                    ids.add(int(part.strip()))
    return ids


def _citations(docs: list[Document], cited: set[int]) -> list[dict]:
    """Structured citation payload: every retrieved chunk, flagged if the answer cites it."""
    out = []
    for i, doc in enumerate(docs, 1):
        meta = doc.metadata
        out.append(
            {
                "n": i,
                "filename": meta.get("filename", ""),
                "section": meta.get("section", ""),
                "page": _page_number(meta),
                "page_start": meta.get("page_start", 0),
                "page_end": meta.get("page_end", 0),
                "pages": meta.get("pages", []),
                "rerank_score": meta.get("rerank_score"),
                "cited": i in cited,
                "snippet": doc.page_content[:900],
            }
        )
    return out


def _is_capacity_error(exc: Exception) -> bool:
    msg = str(exc)
    return any(
        s in msg
        for s in ("429", "RESOURCE_EXHAUSTED", "quota", "503", "UNAVAILABLE", "overloaded", "high demand")
    )


def list_documents() -> list[dict]:
    """Documents in the library (manifest, or the committed library file in read-only mode)."""
    if not MANIFEST_PATH.exists():
        return []
    manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    docs = [
        {k: v for k, v in entry.items() if k != "ids"} | {"filename": name}
        for name, entry in manifest.items()
    ]
    return sorted(docs, key=lambda d: d.get("uploaded_at", ""), reverse=True)


class RAGEngine:
    """Long-lived RAG instance shared across all API requests."""

    def __init__(self) -> None:
        self._validate_env()

        # ── Heavy, long-lived clients (built ONCE) ───────────────────────
        self.pc = hs.ensure_pinecone_index()
        self.embeddings = hs.make_embeddings(self.pc)
        # Shared Docling converter, built on the first upload (see `converter`)
        # so a query-only process never loads Docling's models.
        self._converter = None

        # ── The stateless retrieval chain (built ONCE, reused forever) ───
        self.retrieval_chain = hs.build_retrieval_chain(self.pc, self.embeddings)

        # ── Generation chains (prompt → LLM → structured RAGAnswer) ──────
        # Primary model first, then fallbacks for free-tier rate limits.
        self.generation_chains = [
            (
                model,
                PROMPT
                | ChatGoogleGenerativeAI(
                    model=model, temperature=0.1, max_retries=1
                ).with_structured_output(RAGAnswer),
            )
            for model in [GENERATION_MODEL, *GENERATION_FALLBACKS]
        ]

        if not READ_ONLY:
            hs.UPLOADS_DIR.mkdir(parents=True, exist_ok=True)
        self._manifest_lock = threading.Lock()

    @property
    def converter(self):
        if self._converter is None:
            self._converter = hs._make_docling_converter()
        return self._converter

    @staticmethod
    def _check_writable() -> None:
        if READ_ONLY:
            raise PermissionError(
                "This is a read-only demo: uploads and deletes are disabled. "
                "Run DocIntel locally to add your own PDFs."
            )

    @staticmethod
    def _validate_env() -> None:
        missing = [
            k
            for k in ("PINECONE_API_KEY", "GOOGLE_API_KEY", "COHERE_API_KEY")
            if not os.getenv(k)
        ]
        if missing:
            raise RuntimeError(f"Missing required env vars: {', '.join(missing)}")

    # ── Manifest (which chunk IDs belong to which document) ──────────────
    def _read_manifest(self) -> dict:
        if MANIFEST_PATH.exists():
            return json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
        return {}

    def _write_manifest(self, manifest: dict) -> None:
        tmp = MANIFEST_PATH.with_suffix(".tmp")
        tmp.write_text(json.dumps(manifest, indent=1), encoding="utf-8")
        tmp.replace(MANIFEST_PATH)

    def list_documents(self) -> list[dict]:
        return list_documents()

    def delete_document(self, filename: str) -> bool:
        self._check_writable()
        with self._manifest_lock:
            manifest = self._read_manifest()
            entry = manifest.pop(filename, None)
            if entry is None:
                return False
            hs.delete_chunks(self.pc, entry.get("ids", []))
            self._write_manifest(manifest)
        (hs.UPLOADS_DIR / filename).unlink(missing_ok=True)
        return True

    # ── Upload endpoint ──────────────────────────────────────────────────
    def ingest(self, pdf_path: str | Path) -> dict:
        """
        Ingest one PDF: parse → chunk → upsert to dense + sparse indexes.

        Synchronous + CPU/IO-heavy (Docling parse, embeddings). The API layer
        runs it in a threadpool so it doesn't block the event loop. Idempotent:
        deterministic chunk IDs mean re-uploading the same file overwrites, and
        chunks from an older version of the same file are deleted.
        """
        self._check_writable()
        t0 = time.perf_counter()
        pdf_path = Path(pdf_path)
        name, dl_doc = hs.convert_pdf(pdf_path, converter=self.converter)
        chunks = hs.chunk_documents([(name, dl_doc)])
        if not chunks:
            return {"filename": name, "chunks_added": 0}

        hs.index_corpus(self.pc, chunks, self.embeddings)
        ids = [hs.make_chunk_id(c) for c in chunks]

        with self._manifest_lock:
            manifest = self._read_manifest()
            stale = sorted(set(manifest.get(name, {}).get("ids", [])) - set(ids))
            hs.delete_chunks(self.pc, stale)
            manifest[name] = {
                "chunks": len(chunks),
                "pages": dl_doc.num_pages(),
                "sections": len({c.metadata["section"] for c in chunks}),
                "uploaded_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                "ids": ids,
            }
            self._write_manifest(manifest)

        return {
            "filename": name,
            "chunks_added": len(chunks),
            "pages": dl_doc.num_pages(),
            "stale_chunks_removed": len(stale),
            "seconds": round(time.perf_counter() - t0, 1),
        }

    # ── Query endpoint ─────────────────────────────────────────────────────
    def retrieve(self, query: str, filenames: list[str] | None = None) -> list[Document]:
        """Run the (sync) retrieval+rerank chain. Returns the final top-K docs."""
        return self.retrieval_chain.invoke({"query": query, "filenames": filenames or None})

    async def _generate(self, context: str, query: str) -> tuple[RAGAnswer, str]:
        last_exc: Exception | None = None
        for model, chain in self.generation_chains:
            try:
                result = await chain.ainvoke({"context": context, "question": query})
                if result is not None:
                    return result, model
            except Exception as exc:  # noqa: BLE001
                if not _is_capacity_error(exc):
                    raise
                print(f"  [WARN] {model} unavailable ({str(exc)[:100]}); trying next model.")
                last_exc = exc
        raise RuntimeError(f"All generation models are rate-limited: {last_exc}")

    async def answer(self, query: str, filenames: list[str] | None = None) -> dict:
        """
        Answer a query in one shot (no streaming) and return the whole result:

            {
                "answer":         "<direct grounded answer>",
                "insights":       ["<insight / next step>", ...],
                "next_questions": ["<suggested follow-up>", ...],
                "citations":      [... each flagged `cited` if the answer uses it],
                "model", "reranker", "timings": {...}
            }

        Retrieval and generation are both network-bound; retrieval is sync so it
        runs in a thread to avoid blocking the event loop.
        """
        t0 = time.perf_counter()
        # 1) Retrieve + rerank (sync work off the event loop).
        docs = await asyncio.to_thread(self.retrieve, query, filenames)
        t_retrieve = time.perf_counter() - t0

        if not docs:
            return {
                "answer": "I don't have any indexed documents to answer from."
                if not filenames
                else "Nothing relevant was found in the selected documents.",
                "insights": [],
                "next_questions": [],
                "citations": [],
                "model": None,
                "reranker": None,
                "timings": {"retrieval_ms": round(t_retrieve * 1000), "generation_ms": 0,
                            "total_ms": round(t_retrieve * 1000)},
            }

        # 2) Generate the grounded, structured answer in a single call.
        context = _format_context(docs)
        t1 = time.perf_counter()
        result, model = await self._generate(context, query)
        t_generate = time.perf_counter() - t1

        cited = cited_block_ids(result.answer, *result.insights)
        return {
            "answer": result.answer,
            "insights": result.insights,
            "next_questions": result.next_questions,
            "citations": _citations(docs, cited),
            "model": model,
            "reranker": docs[0].metadata.get("reranker"),
            "timings": {
                "retrieval_ms": round(t_retrieve * 1000),
                "generation_ms": round(t_generate * 1000),
                "total_ms": round((time.perf_counter() - t0) * 1000),
            },
        }
