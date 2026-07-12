"""Exact + semantic caching for LLM calls and generated audio.

L1 exact match: sha256(params_key, prompt) -> response, in SQLite.
L2 semantic match: MiniLM cosine similarity >= threshold against prior prompts
sharing the same params_key (never cross task/model/schema boundaries).
"""
from __future__ import annotations

import hashlib
import sqlite3
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from context import estimate_tokens

_embedder = None  # lazy singleton, shared with guardrails.ground_claims


def _get_embedder():
    global _embedder
    if _embedder is None:
        from sentence_transformers import SentenceTransformer
        _embedder = SentenceTransformer("all-MiniLM-L6-v2")
    return _embedder


def embed(text: str) -> np.ndarray:
    return _get_embedder().encode(text, normalize_embeddings=True)


@dataclass
class CacheHit:
    response: str
    kind: str  # "exact" | "semantic"
    similarity: float = 1.0


class SemanticCache:
    def __init__(self, db_path: str | Path, similarity_threshold: float = 0.95):
        self.db_path = str(db_path)
        self.threshold = similarity_threshold
        Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(self.db_path)
        self._conn.execute(
            """CREATE TABLE IF NOT EXISTS entries (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                params_key TEXT NOT NULL,
                prompt_hash TEXT NOT NULL,
                prompt_text TEXT NOT NULL,
                embedding BLOB NOT NULL,
                response TEXT NOT NULL,
                created_at REAL NOT NULL,
                tokens_est INTEGER NOT NULL
            )"""
        )
        self._conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_params_hash ON entries(params_key, prompt_hash)"
        )
        self._conn.commit()
        self._stats = {"exact_hits": 0, "semantic_hits": 0, "misses": 0, "tokens_saved": 0}

    @staticmethod
    def _hash(prompt: str) -> str:
        return hashlib.sha256(prompt.encode("utf-8")).hexdigest()

    def get(self, prompt: str, params_key: str) -> CacheHit | None:
        h = self._hash(prompt)
        row = self._conn.execute(
            "SELECT response FROM entries WHERE params_key=? AND prompt_hash=? LIMIT 1",
            (params_key, h),
        ).fetchone()
        if row:
            self._stats["exact_hits"] += 1
            self._stats["tokens_saved"] += estimate_tokens(row[0])
            return CacheHit(response=row[0], kind="exact", similarity=1.0)

        rows = self._conn.execute(
            "SELECT response, embedding FROM entries WHERE params_key=?", (params_key,)
        ).fetchall()
        if rows:
            query_emb = embed(prompt)
            best_sim, best_response = -1.0, None
            for response, blob in rows:
                cand = np.frombuffer(blob, dtype=np.float32)
                sim = float(np.dot(query_emb, cand))  # both normalized -> cosine
                if sim > best_sim:
                    best_sim, best_response = sim, response
            if best_sim >= self.threshold:
                self._stats["semantic_hits"] += 1
                self._stats["tokens_saved"] += estimate_tokens(best_response)
                return CacheHit(response=best_response, kind="semantic", similarity=best_sim)

        self._stats["misses"] += 1
        return None

    def put(self, prompt: str, params_key: str, response: str) -> None:
        emb = embed(prompt).astype(np.float32).tobytes()
        self._conn.execute(
            "INSERT INTO entries (params_key, prompt_hash, prompt_text, embedding, response, created_at, tokens_est) "
            "VALUES (?, ?, ?, ?, ?, ?, ?)",
            (params_key, self._hash(prompt), prompt, emb, response, time.time(), estimate_tokens(response)),
        )
        self._conn.commit()

    def stats(self) -> dict:
        total = sum(self._stats[k] for k in ("exact_hits", "semantic_hits", "misses"))
        hit_rate = (self._stats["exact_hits"] + self._stats["semantic_hits"]) / total if total else 0.0
        return {**self._stats, "total_lookups": total, "hit_rate": round(hit_rate, 3)}


class FileCache:
    """Content-hash -> file path cache, for TTS audio blobs and similar artifacts."""

    def __init__(self, cache_dir: str | Path):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self._hits = 0
        self._misses = 0

    def _key_path(self, key: str, suffix: str) -> Path:
        h = hashlib.sha256(key.encode("utf-8")).hexdigest()
        return self.cache_dir / f"{h}{suffix}"

    def get(self, key: str, suffix: str) -> Path | None:
        p = self._key_path(key, suffix)
        if p.exists():
            self._hits += 1
            return p
        self._misses += 1
        return None

    def path_for(self, key: str, suffix: str) -> Path:
        """Path to write a new artifact to (caller writes the file)."""
        return self._key_path(key, suffix)

    def stats(self) -> dict:
        total = self._hits + self._misses
        return {
            "hits": self._hits,
            "misses": self._misses,
            "hit_rate": round(self._hits / total, 3) if total else 0.0,
        }
