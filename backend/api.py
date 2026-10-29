"""
FastAPI surface for the hybrid-search RAG
=========================================
One shared engine behind a small API, plus the UI:

    GET    /                       — the DocIntel web app (frontend/index.html)
    POST   /upload                 — multipart PDF → ingest into dense + sparse indexes
    GET    /documents              — indexed documents (pages, chunks, upload time)
    DELETE /documents/{filename}   — remove a document and all of its vectors
    POST   /query                  — {"question": "...", "filenames": [...]?}
                                     → {answer, insights, next_questions, citations,
                                        model, reranker, timings}
    GET    /benchmark              — results of the QASPER benchmark (eval/)
    GET    /health                 — liveness

Reusable-instance pattern
-------------------------
The RAGEngine is built ONCE in the lifespan handler and stored on app.state.
Every request gets the same instance via the get_engine dependency — no per
request re-initialization of Pinecone / embeddings / Docling / the LLM.

Run:  uvicorn api:app --host 0.0.0.0 --port 8000   →  open http://localhost:8000
"""

from __future__ import annotations

import asyncio
import json
import os
import shutil
import sys
import threading
import time
from collections import defaultdict, deque
from contextlib import asynccontextmanager
from pathlib import Path

# ─────────────────────────────────────────────────────────────────────────────
# Path anchoring — must come before sibling-module imports
# ─────────────────────────────────────────────────────────────────────────────
# Resolve backend/ from __file__ so uvicorn can be launched from any directory
# (e.g. from the project root: uvicorn backend.api:app).
_BACKEND_DIR = Path(__file__).resolve().parent
if str(_BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(_BACKEND_DIR))

from fastapi import FastAPI, HTTPException, Request, UploadFile, File  # noqa: E402
from fastapi.middleware.cors import CORSMiddleware  # noqa: E402
from fastapi.responses import FileResponse  # noqa: E402
from pydantic import AliasChoices, BaseModel, Field  # noqa: E402

import hybrid_search as hs  # noqa: E402
from rag_engine import READ_ONLY, RAGEngine, list_documents  # noqa: E402

FRONTEND = _BACKEND_DIR.parent / "frontend" / "index.html"
BENCHMARK_SUMMARY = _BACKEND_DIR.parent / "eval" / "results" / "summary.json"
MAX_UPLOAD_BYTES = 50 * 1024 * 1024

# Questions allowed per client IP per 10 minutes (0 = unlimited). On by default
# in read-only mode, so a public demo can't drain the free-tier API quotas.
# In-memory, so it is per server instance: a speed bump, not a hard quota.
RATE_LIMIT = int(os.getenv("DOCINTEL_RATE_LIMIT", "10" if READ_ONLY else "0"))
RATE_WINDOW_S = 600
_hits: dict[str, deque] = defaultdict(deque)

_engine: RAGEngine | None = None
_engine_lock = threading.Lock()


def get_engine() -> RAGEngine:
    """
    The one shared engine, built on first use and reused by every request.

    Built lazily (not only in the lifespan hook) so the same app also works
    where lifespan events don't run, e.g. as a mounted sub-app on Vercel.
    """
    global _engine
    if _engine is None:
        with _engine_lock:
            if _engine is None:
                _engine = RAGEngine()
    return _engine


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Warm the engine at startup so the first question isn't slow. A missing
    # key shouldn't take /health and /benchmark down with it, so log and go on.
    try:
        await asyncio.to_thread(get_engine)
    except Exception as exc:  # noqa: BLE001
        print(f"  [WARN] Engine not ready at startup: {exc}")
    yield
    # Nothing to tear down — clients are stateless network handles.


app = FastAPI(title="DocIntel", version="2.0.0", lifespan=lifespan)

# ── CORS ──────────────────────────────────────────────────────────────────────
# The UI is served from this app at "/", but it may also be opened directly
# (file:// → Origin "null") or from another localhost port. Allow all origins for
# local dev. The app uses no cookies/credentials, so credentials stay off — the
# CORS spec forbids "*" together with allow_credentials=True. Tighten in prod.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


def _client_ip(request: Request) -> str:
    forwarded = request.headers.get("x-forwarded-for", "")
    return forwarded.split(",")[0].strip() or (request.client.host if request.client else "?")


def _rate_limited(request: Request) -> bool:
    if RATE_LIMIT <= 0:
        return False
    now = time.monotonic()
    hits = _hits[_client_ip(request)]
    while hits and now - hits[0] > RATE_WINDOW_S:
        hits.popleft()
    if len(hits) >= RATE_LIMIT:
        return True
    hits.append(now)
    return False


def _require_writable() -> None:
    if READ_ONLY:
        raise HTTPException(
            status_code=403,
            detail="This is a read-only demo: uploads and deletes are disabled. "
                   "Run DocIntel locally to add your own PDFs.",
        )


class QueryRequest(BaseModel):
    # The frontend posts {"question": "..."}; accept "query" as an alias too.
    question: str = Field(validation_alias=AliasChoices("question", "query"))
    # Optional: restrict retrieval to these documents (exact filenames).
    filenames: list[str] | None = None


# ─────────────────────────────────────────────────────────────────────────────
# UI
# ─────────────────────────────────────────────────────────────────────────────
@app.get("/", include_in_schema=False)
async def index():
    return FileResponse(FRONTEND)


# ─────────────────────────────────────────────────────────────────────────────
# Documents
# ─────────────────────────────────────────────────────────────────────────────
@app.post("/upload")
async def upload(
    file: UploadFile = File(...),
):
    _require_writable()
    if not file.filename or not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only .pdf files are supported.")

    head = await file.read(5)
    if head != b"%PDF-":
        raise HTTPException(status_code=400, detail="File is not a valid PDF.")
    await file.seek(0)

    # Persist to uploads/ (also the source of truth for the corpus on disk).
    hs.UPLOADS_DIR.mkdir(parents=True, exist_ok=True)
    dest = hs.UPLOADS_DIR / Path(file.filename).name
    with dest.open("wb") as f:
        shutil.copyfileobj(file.file, f)
    if dest.stat().st_size > MAX_UPLOAD_BYTES:
        dest.unlink(missing_ok=True)
        raise HTTPException(status_code=413, detail="PDF is larger than 50 MB.")

    # Ingestion is sync + heavy → run off the event loop.
    try:
        engine = await asyncio.to_thread(get_engine)
        result = await asyncio.to_thread(engine.ingest, dest)
    except Exception as exc:  # surface a clean error to the client
        raise HTTPException(status_code=500, detail=f"Ingestion failed: {exc}")

    return {"status": "ok", **result}


@app.get("/documents")
async def documents():
    # Only reads the library file, so it doesn't need (or wait for) the engine.
    return {"documents": list_documents(), "read_only": READ_ONLY}


@app.delete("/documents/{filename}")
async def delete_document(filename: str):
    _require_writable()
    engine = await asyncio.to_thread(get_engine)
    removed = await asyncio.to_thread(engine.delete_document, Path(filename).name)
    if not removed:
        raise HTTPException(status_code=404, detail="No such document.")
    return {"status": "deleted", "filename": filename}


# ─────────────────────────────────────────────────────────────────────────────
# Query endpoint — single JSON response (no streaming)
# ─────────────────────────────────────────────────────────────────────────────
@app.post("/query")
async def query(body: QueryRequest, request: Request):
    q = body.question.strip()
    if not q:
        raise HTTPException(status_code=400, detail="question must not be empty.")
    if _rate_limited(request):
        raise HTTPException(
            status_code=429,
            detail=f"Demo limit reached ({RATE_LIMIT} questions per 10 minutes). Please try again shortly.",
        )

    filenames = body.filenames
    if READ_ONLY:
        # The demo shares its Pinecone namespace with whatever was ingested
        # locally, so never search beyond the published library — even if the
        # client asks for "all documents" or names a file outside it.
        library = {d["filename"] for d in list_documents()}
        filenames = [f for f in (filenames or library) if f in library]
        if not filenames:
            raise HTTPException(status_code=400, detail="Pick documents from the demo library.")

    try:
        engine = await asyncio.to_thread(get_engine)
        result = await engine.answer(q, filenames)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Query failed: {exc}")

    return result


# ─────────────────────────────────────────────────────────────────────────────
# Benchmark results (produced by eval/bench.py + eval/report.py)
# ─────────────────────────────────────────────────────────────────────────────
@app.get("/benchmark")
async def benchmark():
    if not BENCHMARK_SUMMARY.exists():
        raise HTTPException(status_code=404, detail="Run eval/bench.py first.")
    return json.loads(BENCHMARK_SUMMARY.read_text(encoding="utf-8"))


@app.get("/health")
async def health():
    return {"status": "ok", "read_only": READ_ONLY}
