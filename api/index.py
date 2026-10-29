"""
Vercel entry point: the DocIntel API as one Python serverless function.

Vercel routes /api/* here (see vercel.json). The backend app is mounted under
/api, so /api/query reaches the backend's /query route unchanged.

This deployment is query-only (DOCINTEL_READ_ONLY=1): it searches the documents
already indexed in Pinecone and answers questions. Ingestion needs Docling and
PyTorch, which are far too large for a serverless function, so PDFs are
indexed by running DocIntel locally. See README → Deployment.
"""

import importlib.util
from pathlib import Path

from fastapi import FastAPI

_BACKEND_API = Path(__file__).resolve().parent.parent / "backend" / "api.py"

# Loaded by file path: Vercel's `api/` folder would shadow a plain `import api`.
_spec = importlib.util.spec_from_file_location("docintel_backend_api", _BACKEND_API)
_backend = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_backend)

app = FastAPI(title="DocIntel (Vercel)")
app.mount("/api", _backend.app)
