"""
retrieval/hybrid_retriever.py

Replaces the Week 1 simple retriever with a 3-stage pipeline:

  Stage 1 — Recall:    FAISS + BM25 independently fetch top-20 each
  Stage 2 — Merge:     Reciprocal Rank Fusion combines both lists
  Stage 3 — Precision: Cross-encoder reranks merged set to top-5

Why Reciprocal Rank Fusion (RRF) for merging:
  Naively averaging scores fails because FAISS returns cosine sims
  (0.0–1.0) and BM25 returns raw term-frequency scores (0–50+).
  They're on completely different scales.

  RRF sidesteps this: it only looks at *rank position*, not score value.
    rrf_score(chunk) = 1/(k + rank_in_faiss) + 1/(k + rank_in_bm25)
  where k=60 is a smoothing constant (standard value from the 2009 paper).

  This is the same technique used in Elasticsearch's hybrid search and
  many production RAG systems. It's robust, parameter-free, and works.

Interface is identical to Week 1's Retriever — the app doesn't change.
"""

from ingestion.chunker import Chunk
from storage.vector_store import VectorStore
from storage.bm25_store import BM25Store
from retrieval.reranker import Reranker

RRF_K = 60          # standard smoothing constant
CANDIDATE_POOL = 20 # how many each retriever fetches before merge


def _reciprocal_rank_fusion(
    faiss_results: list[tuple[Chunk, float]],
    bm25_results: list[tuple[Chunk, float]],
    k: int = RRF_K,
) -> list[tuple[Chunk, float]]:
    """
    Merge two ranked lists using Reciprocal Rank Fusion.

    For each chunk that appears in either list, compute:
      score = Σ 1/(k + rank)  summed across the lists it appears in.

    Chunks appearing in both lists get a double boost — which is
    exactly what we want: cross-method agreement signals relevance.
    """
    scores: dict[str, float] = {}
    chunk_map: dict[str, Chunk] = {}

    for rank, (chunk, _) in enumerate(faiss_results):
        scores[chunk.chunk_id] = scores.get(chunk.chunk_id, 0) + 1 / (k + rank)
        chunk_map[chunk.chunk_id] = chunk

    for rank, (chunk, _) in enumerate(bm25_results):
        scores[chunk.chunk_id] = scores.get(chunk.chunk_id, 0) + 1 / (k + rank)
        chunk_map[chunk.chunk_id] = chunk

    # Sort by RRF score descending
    ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    return [(chunk_map[cid], score) for cid, score in ranked]


class HybridRetriever:
    """
    Drop-in replacement for the Week 1 Retriever.
    Same .retrieve(query, top_k) interface — app code unchanged.
    """

    def __init__(
        self,
        vector_store: VectorStore,
        bm25_store: BM25Store,
        reranker: Reranker,
    ):
        self._vector_store = vector_store
        self._bm25_store = bm25_store
        self._reranker = reranker

    def retrieve(self, query: str, top_k: int = 5) -> list[tuple[Chunk, float]]:
        """
        Full pipeline:
          1. FAISS top-20 (semantic)
          2. BM25  top-20 (keyword)
          3. RRF merge → deduplicated ranked list
          4. Cross-encoder rerank → top_k

        Returns (Chunk, rerank_score) pairs.
        """
        # Stage 1: parallel retrieval
        faiss_results = self._vector_store.search(query, top_k=CANDIDATE_POOL)
        bm25_results  = self._bm25_store.search(query, top_k=CANDIDATE_POOL)

        # Stage 2: merge with RRF
        merged = _reciprocal_rank_fusion(faiss_results, bm25_results)

        # Stage 3: rerank the merged pool
        reranked = self._reranker.rerank(query, merged, top_k=top_k)

        return reranked
