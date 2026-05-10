import hashlib
import os

import chromadb
from openai import OpenAI


class VectorStore:
    def __init__(self, user_id: str = "default"):
        uid_hash = hashlib.md5(user_id.encode()).hexdigest()[:16]
        self._collection_name = f"books_{uid_hash}"
        self.chroma = chromadb.PersistentClient(path="./chroma_db")
        self._collection = self._get_or_create_collection()
        self.openai = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        self.embedding_model = os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small")

    def _get_or_create_collection(self):
        return self.chroma.get_or_create_collection(
            name=self._collection_name,
            metadata={"hnsw:space": "cosine"},
        )

    # ------------------------------------------------------------------
    # Embeddings
    # ------------------------------------------------------------------

    def _embed(self, texts: list[str]) -> list[list[float]]:
        response = self.openai.embeddings.create(
            model=self.embedding_model,
            input=texts,
        )
        return [item.embedding for item in response.data]

    # ------------------------------------------------------------------
    # Write
    # ------------------------------------------------------------------

    def add_chunks(self, chunks: list[dict]) -> None:
        """Embed and store chunks in ChromaDB (in batches of 100)."""
        BATCH = 100
        for i in range(0, len(chunks), BATCH):
            batch = chunks[i : i + BATCH]
            texts = [c["text"] for c in batch]
            embeddings = self._embed(texts)
            self._collection.add(
                ids=[c["chunk_id"] for c in batch],
                embeddings=embeddings,
                documents=texts,
                metadatas=[
                    {
                        "page": c["page"],
                        "filename": c["filename"],
                        "author": c["author"],
                        "title": c["title"],
                    }
                    for c in batch
                ],
            )

    def delete_all(self) -> None:
        """Delete the entire collection and recreate it empty."""
        self.chroma.delete_collection(self._collection_name)
        self._collection = self._get_or_create_collection()

    # ------------------------------------------------------------------
    # Read
    # ------------------------------------------------------------------

    def search(self, query: str, top_k: int = 5) -> list[dict]:
        """Return the top_k most relevant chunks for the query."""
        query_embedding = self._embed([query])[0]
        results = self._collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k,
            include=["documents", "metadatas", "distances"],
        )
        chunks = []
        for doc, meta, dist in zip(
            results["documents"][0],
            results["metadatas"][0],
            results["distances"][0],
        ):
            chunks.append(
                {
                    "text": doc,
                    "author": meta["author"],
                    "title": meta["title"],
                    "page": meta["page"],
                    "filename": meta["filename"],
                    "distance": dist,
                }
            )
        return chunks

    def list_books(self) -> list[dict]:
        """Return list of unique books with chunk counts."""
        all_items = self._collection.get(include=["metadatas"])
        books: dict[str, dict] = {}
        for meta in all_items["metadatas"]:
            fn = meta["filename"]
            if fn not in books:
                books[fn] = {
                    "filename": fn,
                    "author": meta["author"],
                    "title": meta["title"],
                    "chunk_count": 0,
                }
            books[fn]["chunk_count"] += 1
        return list(books.values())

    def book_exists(self, filename: str) -> bool:
        """Check if a book with this filename is already indexed."""
        results = self._collection.get(
            where={"filename": filename},
            limit=1,
            include=["metadatas"],
        )
        return len(results["ids"]) > 0
