import math
import numpy as np
from collections import Counter
from dataclasses import dataclass
from typing import Optional
import re

from .topic_detector import STOPWORDS


def _tokenize_query(text: str) -> list[str]:
    tokens = re.findall(r"[a-z']+", text.lower())
    return [t for t in tokens if t not in STOPWORDS and len(t) > 2]


@dataclass
class StoredChunk:
    chunk_id: int
    text: str
    kind: str
    meta: dict
    bow: Counter


class VectorStore:
    def __init__(self):
        self.chunks: list[StoredChunk] = []
        self._df: Counter = Counter()
        self._n_docs: int = 0

    def add(self, texts: list[str], kinds: list[str], metas: list[dict]) -> None:
        new_chunks = []
        for text, kind, meta in zip(texts, kinds, metas):
            tokens = _tokenize_query(text)
            bow = Counter(tokens)
            chunk = StoredChunk(
                chunk_id=len(self.chunks) + len(new_chunks),
                text=text,
                kind=kind,
                meta=meta,
                bow=bow,
            )
            new_chunks.append(chunk)
            for term in set(tokens):
                self._df[term] += 1

        self.chunks.extend(new_chunks)
        self._n_docs = len(self.chunks)

    def _tfidf_vec(self, bow: Counter) -> dict[str, float]:
        n = self._n_docs or 1
        total = sum(bow.values()) or 1
        vec = {}
        for term, count in bow.items():
            tf = count / total
            idf = math.log((n + 1) / (self._df.get(term, 0) + 1)) + 1
            vec[term] = tf * idf
        return vec

    def _cosine(self, a: dict[str, float], b: dict[str, float]) -> float:
        keys = set(a) & set(b)
        if not keys:
            return 0.0
        dot = sum(a[k] * b[k] for k in keys)
        mag_a = math.sqrt(sum(v * v for v in a.values()))
        mag_b = math.sqrt(sum(v * v for v in b.values()))
        if mag_a == 0 or mag_b == 0:
            return 0.0
        return dot / (mag_a * mag_b)

    def search(self, query: str, top_k: int = 5) -> list[tuple[StoredChunk, float]]:
        if not self.chunks:
            return []
        q_tokens = _tokenize_query(query)
        q_bow = Counter(q_tokens)
        q_vec = self._tfidf_vec(q_bow)

        scored = []
        for chunk in self.chunks:
            c_vec = self._tfidf_vec(chunk.bow)
            score = self._cosine(q_vec, c_vec)
            if score > 0:
                scored.append((chunk, score))

        scored.sort(key=lambda x: x[1], reverse=True)
        return scored[:top_k]

    def search_by_kind(self, query: str, kind: str, top_k: int = 5) -> list[tuple[StoredChunk, float]]:
        if not self.chunks:
            return []
        q_tokens = _tokenize_query(query)
        q_bow = Counter(q_tokens)
        q_vec = self._tfidf_vec(q_bow)

        scored = []
        for chunk in self.chunks:
            if chunk.kind != kind:
                continue
            c_vec = self._tfidf_vec(chunk.bow)
            score = self._cosine(q_vec, c_vec)
            if score > 0:
                scored.append((chunk, score))

        scored.sort(key=lambda x: x[1], reverse=True)
        return scored[:top_k]

    @property
    def total(self) -> int:
        return len(self.chunks)
