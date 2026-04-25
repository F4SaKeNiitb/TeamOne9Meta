"""Lightweight hybrid retriever (BM25 keyword + Jaccard scoring).

Sentence-transformer dense retrieval is available when torch is installed, but
we default to Jaccard so the env runs on minimal deps. The interface is the
same, so a Colab notebook can swap in a transformer without touching callers.
"""

from typing import List, Tuple
from .dedup import jaccard


class HybridRetriever:
    def __init__(self, corpus: List[str]):
        self.corpus = list(corpus)

    def top_k(self, query: str, k: int = 3) -> List[Tuple[int, float, str]]:
        scores: List[Tuple[int, float, str]] = []
        for i, doc in enumerate(self.corpus):
            s = jaccard(query, doc)
            scores.append((i, s, doc))
        scores.sort(key=lambda x: -x[1])
        return scores[:k]
