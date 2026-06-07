"""
retrieval/reranker.py

A cross-encoder reranker takes (query, chunk) pairs and scores them
together — unlike bi-encoders (FAISS) which encode query and chunk
separately.

Why cross-encoders are more accurate:
  Bi-encoder: embed(query) · embed(chunk) = score
    → Fast, scales to millions, but query and chunk never "see" each other
  Cross-encoder: model([query, chunk]) = score
    → Slow (can't pre-compute), but reads both together = much better judgment

Standard production pattern:
  1. Bi-encoder retrieves top-50 cheaply (FAISS)
  2. Cross-encoder reranks to top-5 accurately
  This gives you near-reranker quality at near-retriever speed.

Model: cross-encoder/ms-marco-MiniLM-L-6-v2
  - ~80MB, runs on CPU in ~100ms for 20 pairs
  - Trained on MS MARCO passage ranking (140M question-passage pairs)
  - Standard choice in open-source RAG stacks
"""

from sentence_transformers import CrossEncoder
from ingestion.chunker import Chunk

RERANKER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"


class Reranker:
    def __init__(self):
        # Downloaded from HuggingFace on first run (~80MB)
        # Cached locally after that — no repeated downloads
        print("Loading reranker model (first run downloads ~80MB)...")
        self._model = CrossEncoder(RERANKER_MODEL)
        print("Reranker ready.")

    def rerank(
        self,
        query: str,
        candidates: list[tuple[Chunk, float]],
        top_k: int = 5,
    ) -> list[tuple[Chunk, float]]:
        """
        Rerank candidate chunks using the cross-encoder.

        Args:
            query:      the user's question
            candidates: (Chunk, score) pairs from hybrid retrieval
            top_k:      how many to return after reranking

        Returns:
            top_k (Chunk, rerank_score) pairs, best first.

        The rerank scores are logits (not probabilities), but
        higher = more relevant. We keep them for UI display.
        """
        if not candidates:
            return []

        # Build (query, chunk_text) pairs for the cross-encoder
        pairs = [(query, chunk.text) for chunk, _ in candidates]

        # Score all pairs in one batch — much faster than one-by-one
        scores = self._model.predict(pairs)    # returns numpy array

        # Zip scores back with chunks, sort descending
        scored = sorted(
            zip([c for c, _ in candidates], scores.tolist()),
            key=lambda x: x[1],
            reverse=True,
        )

        return scored[:top_k]
