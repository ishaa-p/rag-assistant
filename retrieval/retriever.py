"""
retrieval/retriever.py

Week 1: simple vector search retriever.
Week 2: will be upgraded to hybrid (BM25 + FAISS) + reranker.

Keeping this as a thin wrapper means the rest of the app doesn't
need to change when we upgrade retrieval logic here.
"""

from storage.vector_store import VectorStore
from ingestion.chunker import Chunk


class Retriever:
    def __init__(self, vector_store: VectorStore):
        self._store = vector_store

    def retrieve(self, query: str, top_k: int = 5) -> list[tuple[Chunk, float]]:
        """
        Return top_k chunks for a query.

        Week 1: pure vector search (top_k from index directly).
        Week 2: vector + BM25 → merge → rerank → top_k.
        """
        # We fetch 20 from FAISS, then take top_k
        # This prepares for reranker integration in Week 2
        # without changing the interface
        candidates = self._store.search(query, top_k=20)
        return candidates[:top_k]
