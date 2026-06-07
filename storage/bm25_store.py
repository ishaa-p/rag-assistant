"""
storage/bm25_store.py

BM25 (Best Match 25) is a classic keyword-based ranking algorithm.
It scores chunks by exact word frequency, penalising very long chunks
so they don't dominate just by having more words.

Why we need this alongside FAISS:
  - FAISS (semantic): finds conceptually similar chunks even with
    different words. "automobile" matches "car".
  - BM25 (keyword):   finds exact terminology. If a user asks about
    "RFC 7231" or a specific model name, FAISS may miss it because
    the embedding space doesn't sharply separate rare terms.

Combining both (hybrid search) catches what either alone misses.
This is a well-established pattern in production search systems
(Elasticsearch uses BM25 as its default scorer).
"""

from rank_bm25 import BM25Okapi
from ingestion.chunker import Chunk


def _tokenize(text: str) -> list[str]:
    """
    Simple whitespace + lowercase tokenizer.
    Good enough for most RAG use cases.
    Production systems would add stopword removal + stemming,
    but that adds complexity without major RAG accuracy gains.
    """
    return text.lower().split()


class BM25Store:
    """
    Wraps rank_bm25.BM25Okapi for keyword search over chunks.

    BM25 is not a neural model — it runs instantly, no GPU needed,
    and costs nothing beyond RAM. That's why we use it as the
    keyword leg of hybrid search.
    """

    def __init__(self):
        self._bm25: BM25Okapi | None = None
        self._chunks: list[Chunk] = []

    def build(self, chunks: list[Chunk]) -> None:
        """Tokenize all chunks and build the BM25 index."""
        self._chunks = chunks
        tokenized = [_tokenize(c.text) for c in chunks]
        self._bm25 = BM25Okapi(tokenized)

    def search(self, query: str, top_k: int = 20) -> list[tuple[Chunk, float]]:
        """
        Return top_k chunks by BM25 score.

        Scores are NOT normalized — they're relative within a query.
        We normalize them in the hybrid merger so they're comparable
        to FAISS cosine scores.
        """
        if self._bm25 is None:
            raise RuntimeError("BM25 index not built. Call build() first.")

        query_tokens = _tokenize(query)
        scores = self._bm25.get_scores(query_tokens)   # one score per chunk

        # Get top_k indices sorted by score descending
        top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_k]

        return [(self._chunks[i], float(scores[i])) for i in top_indices]
