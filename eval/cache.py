"""
Tiny SQLite key-value cache for every paid / rate-limited call the benchmark
makes (embeddings, sparse vectors, rerank, generation, judging).

The Gemini key is free-tier (5 req/min per model) and the Cohere key is a trial
key, so every result is persisted the moment it comes back. Re-running any
stage is free and the benchmark resumes exactly where it stopped.
"""

from __future__ import annotations

import hashlib
import json
import sqlite3
import threading
from pathlib import Path

import numpy as np

CACHE_PATH = Path(__file__).resolve().parent / "cache" / "cache.sqlite"


def key_of(*parts: str) -> str:
    return hashlib.sha1("\x1f".join(parts).encode("utf-8")).hexdigest()


class Cache:
    def __init__(self, path: Path = CACHE_PATH) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(path), check_same_thread=False)
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute(
            "CREATE TABLE IF NOT EXISTS kv (ns TEXT, k TEXT, v BLOB, PRIMARY KEY (ns, k))"
        )
        self._lock = threading.Lock()

    def _get(self, ns: str, k: str):
        with self._lock:
            row = self._conn.execute("SELECT v FROM kv WHERE ns=? AND k=?", (ns, k)).fetchone()
        return None if row is None else row[0]

    def _put(self, ns: str, k: str, v) -> None:
        with self._lock:
            self._conn.execute("INSERT OR REPLACE INTO kv VALUES (?,?,?)", (ns, k, v))
            self._conn.commit()

    # JSON values
    def get(self, ns: str, k: str):
        v = self._get(ns, k)
        return None if v is None else json.loads(v)

    def put(self, ns: str, k: str, value) -> None:
        self._put(ns, k, json.dumps(value, ensure_ascii=False))

    # float32 vectors
    def get_vec(self, ns: str, k: str):
        v = self._get(ns, k)
        return None if v is None else np.frombuffer(v, dtype=np.float32)

    def put_vec(self, ns: str, k: str, vec) -> None:
        self._put(ns, k, np.asarray(vec, dtype=np.float32).tobytes())
