"""
storage/vector_store.py

Wraps FAISS for storing and searching chunk embeddings.

Why FAISS:
  - Runs locally, no API cost
  - Fast even with 100k+ chunks
  - Easy to persist to disk and reload

In Week 2 you'll swap this for a hybrid store (FAISS + BM25).
The interface here is designed so that swap is one file change.
"""

import os
import json
import pickle
from pathlib import Path
from typing import Optional

import faiss
import numpy as np
from openai import OpenAI
from dotenv import load_dotenv

from ingestion.chunker import Chunk

load_dotenv()

EMBED_MODEL = "text-embedding-3-small"   # cheap, fast, good quality
EMBED_DIM = 1536                          # dimension for text-embedding-3-small
INDEX_PATH = "data/faiss.index"
METADATA_PATH = "data/metadata.pkl"


class VectorStore:
    """
    Stores chunk embeddings in FAISS and metadata in a parallel list.

    The FAISS index maps integer IDs → embedding vectors.
    self._chunks[i] holds the Chunk object for vector at position i.
    """

    def __init__(self):
        self._client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        self._index: Optional[faiss.IndexFlatIP] = None   # inner product = cosine after normalisation
        self._chunks: list[Chunk] = []

    # ------------------------------------------------------------------
    # Embedding helpers
    # ------------------------------------------------------------------

    def _embed(self, texts: list[str]) -> np.ndarray:
        """
        Call OpenAI embeddings API in batches.
        Returns float32 array of shape (n, EMBED_DIM), L2-normalised
        so that inner product == cosine similarity.
        """
        batch_size = 100   # OpenAI allows up to 2048 inputs per call
        all_embeddings = []

        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            response = self._client.embeddings.create(
                input=batch,
                model=EMBED_MODEL,
            )
            batch_vecs = [item.embedding for item in response.data]
            all_embeddings.extend(batch_vecs)

        arr = np.array(all_embeddings, dtype="float32")
        # L2 normalise so cosine sim = dot product (required for IndexFlatIP)
        faiss.normalize_L2(arr)
        return arr

    # ------------------------------------------------------------------
    # Build / persist
    # ------------------------------------------------------------------

    def build(self, chunks: list[Chunk]) -> None:
        """Embed all chunks and build the FAISS index from scratch."""
        print(f"Embedding {len(chunks)} chunks...")
        texts = [c.text for c in chunks]
        embeddings = self._embed(texts)

        self._index = faiss.IndexFlatIP(EMBED_DIM)
        self._index.add(embeddings)
        self._chunks = chunks
        print("FAISS index built.")

    def save(self) -> None:
        """Persist index + metadata to disk so we don't re-embed on restart."""
        Path("data").mkdir(exist_ok=True)
        faiss.write_index(self._index, INDEX_PATH)
        with open(METADATA_PATH, "wb") as f:
            pickle.dump(self._chunks, f)
        print(f"Saved index ({len(self._chunks)} chunks) to {INDEX_PATH}")

    def load(self) -> bool:
        """Load a previously saved index. Returns True if successful."""
        if not Path(INDEX_PATH).exists():
            return False
        self._index = faiss.read_index(INDEX_PATH)
        with open(METADATA_PATH, "rb") as f:
            self._chunks = pickle.load(f)
        print(f"Loaded index with {len(self._chunks)} chunks.")
        return True

    # ------------------------------------------------------------------
    # Search
    # ------------------------------------------------------------------

    def search(self, query: str, top_k: int = 20) -> list[tuple[Chunk, float]]:
        """
        Find the top_k most semantically similar chunks.

        Returns list of (Chunk, score) tuples, sorted best-first.
        top_k=20 is intentionally large — the reranker (Week 2) will
        cut this down to 5 high-quality results.
        """
        if self._index is None or self._index.ntotal == 0:
            raise RuntimeError("Index is empty. Run build() first.")

        query_vec = self._embed([query])                   # shape (1, dim)
        scores, indices = self._index.search(query_vec, top_k)

        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx == -1:    # FAISS returns -1 for empty slots
                continue
            results.append((self._chunks[idx], float(score)))

        return results
