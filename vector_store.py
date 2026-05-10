import hashlib
import json
import os
import sqlite3
import time
from pathlib import Path

import numpy as np
from openai import OpenAI

_DB_DIR = Path("./chroma_db")  # reuse existing volume mount path


class VectorStore:
    def __init__(self, user_id: str = "default"):
        uid_hash = hashlib.md5(user_id.encode()).hexdigest()[:16]
        _DB_DIR.mkdir(parents=True, exist_ok=True)
        self._db_path = _DB_DIR / f"store_{uid_hash}.db"
        self._init_db()
        self.openai = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        self.embedding_model = os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small")

    def _init_db(self) -> None:
        with sqlite3.connect(self._db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS chunks (
                    id       TEXT PRIMARY KEY,
                    text     TEXT NOT NULL,
                    embedding TEXT NOT NULL,
                    page     INTEGER,
                    filename TEXT,
                    author   TEXT,
                    title    TEXT
                )
            """)

    # ------------------------------------------------------------------
    # Embeddings
    # ------------------------------------------------------------------

    def _embed(self, texts: list[str]) -> list[list[float]]:
        for attempt in range(6):
            try:
                response = self.openai.embeddings.create(
                    model=self.embedding_model,
                    input=texts,
                )
                return [item.embedding for item in response.data]
            except Exception as e:
                msg = str(e)
                if "429" in msg or "rate_limit" in msg.lower():
                    wait = min(2 ** attempt, 60)
                    time.sleep(wait)
                else:
                    raise
        raise RuntimeError("OpenAI rate limit: превышено число попыток")

    # ------------------------------------------------------------------
    # Write
    # ------------------------------------------------------------------

    def add_chunks(self, chunks: list[dict]) -> None:
        BATCH = 20
        for i in range(0, len(chunks), BATCH):
            batch = chunks[i : i + BATCH]
            embeddings = self._embed([c["text"] for c in batch])
            with sqlite3.connect(self._db_path) as conn:
                conn.executemany(
                    "INSERT OR IGNORE INTO chunks VALUES (?,?,?,?,?,?,?)",
                    [
                        (
                            c["chunk_id"],
                            c["text"],
                            json.dumps(emb),
                            c["page"],
                            c["filename"],
                            c["author"],
                            c["title"],
                        )
                        for c, emb in zip(batch, embeddings)
                    ],
                )

    def delete_all(self) -> None:
        with sqlite3.connect(self._db_path) as conn:
            conn.execute("DELETE FROM chunks")

    def delete_book(self, filename: str) -> None:
        with sqlite3.connect(self._db_path) as conn:
            conn.execute("DELETE FROM chunks WHERE filename = ?", (filename,))

    # ------------------------------------------------------------------
    # Read
    # ------------------------------------------------------------------

    def search(self, query: str, top_k: int = 5) -> list[dict]:
        query_vec = np.array(self._embed([query])[0], dtype=np.float32)

        with sqlite3.connect(self._db_path) as conn:
            rows = conn.execute(
                "SELECT text, embedding, page, filename, author, title FROM chunks"
            ).fetchall()

        if not rows:
            return []

        texts, emb_jsons, pages, filenames, authors, titles = zip(*rows)
        matrix = np.array([json.loads(e) for e in emb_jsons], dtype=np.float32)

        # cosine similarity
        norms = np.linalg.norm(matrix, axis=1) * np.linalg.norm(query_vec)
        norms = np.where(norms == 0, 1e-9, norms)
        sims = (matrix @ query_vec) / norms

        top_idx = np.argsort(sims)[::-1][:top_k]
        return [
            {
                "text": texts[i],
                "page": pages[i],
                "filename": filenames[i],
                "author": authors[i],
                "title": titles[i],
                "distance": float(1.0 - sims[i]),
            }
            for i in top_idx
        ]

    def list_books(self) -> list[dict]:
        with sqlite3.connect(self._db_path) as conn:
            rows = conn.execute(
                "SELECT filename, author, title, COUNT(*) FROM chunks GROUP BY filename"
            ).fetchall()
        return [
            {"filename": r[0], "author": r[1], "title": r[2], "chunk_count": r[3]}
            for r in rows
        ]

    def book_exists(self, filename: str) -> bool:
        with sqlite3.connect(self._db_path) as conn:
            row = conn.execute(
                "SELECT 1 FROM chunks WHERE filename = ? LIMIT 1", (filename,)
            ).fetchone()
        return row is not None
