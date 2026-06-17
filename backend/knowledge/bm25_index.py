"""
knowledge/bm25_index.py
────────────────────────
Sparse BM25 index built on top of all ChromaDB chunks.
Catches exact tech terms like "BERT", "ISRO", "S3" that vector search misses.
Rebuilt from ChromaDB on each server start (fast, in-memory).
"""
import re
from typing import List, Dict, Any, Optional

from rank_bm25 import BM25Okapi


class BM25Index:
    def __init__(self):
        self._index: Optional[BM25Okapi] = None
        self._chunks: List[Dict[str, Any]] = []

    def build(self, chunks: List[Dict[str, Any]]) -> None:
        """
        Build BM25 index from a list of {text, metadata} dicts.
        Called after ingestion or on startup.
        """
        self._chunks = chunks
        tokenized = [self._tokenize(c["text"]) for c in chunks]
        if tokenized:
            self._index = BM25Okapi(tokenized)

    def query(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """
        Return top_k chunks by BM25 score.
        Returns list of {text, metadata, score}.
        """
        if self._index is None or not self._chunks:
            return []

        tokens = self._tokenize(query)
        scores = self._index.get_scores(tokens)

        scored = sorted(
            zip(scores, self._chunks),
            key=lambda x: x[0],
            reverse=True
        )[:top_k]

        results = []
        for score, chunk in scored:
            if score > 0:
                results.append({
                    "text": chunk["text"],
                    "metadata": chunk["metadata"],
                    "score": float(score),
                })
        return results

    def add_chunk(self, chunk: Dict[str, Any]) -> None:
        """Add a single chunk and rebuild (used for incremental ingestion)."""
        self._chunks.append(chunk)
        self.build(self._chunks)

    @staticmethod
    def _tokenize(text: str) -> List[str]:
        """
        Tokenize for BM25.
        Preserves acronyms (BERT, AWS, TensorFlow) by splitting on whitespace
        and punctuation but keeping alphanumeric sequences intact.
        """
        text = text.lower()
        tokens = re.findall(r"[a-z0-9]+", text)
        return tokens

    @property
    def size(self) -> int:
        return len(self._chunks)


# Module-level singleton
_bm25_index = BM25Index()


def get_bm25_index() -> BM25Index:
    return _bm25_index


def rebuild_bm25_from_chroma() -> int:
    """Pull all chunks from ChromaDB and rebuild BM25 index. Returns chunk count."""
    from backend.ingestion.document import get_all_chunks
    chunks = get_all_chunks()
    _bm25_index.build(chunks)
    return len(chunks)
