from __future__ import annotations

from typing import List

from sentence_transformers import CrossEncoder

from .config import settings
from .types import RetrievedChunk


class Reranker:
    def __init__(self) -> None:
        self.model = CrossEncoder(settings.rerank_model)

    def rerank(self, query: str, chunks: List[RetrievedChunk]) -> List[RetrievedChunk]:
        import time
        start_time = time.time()
        if not chunks:
            return []
        pairs = [(query, chunk.record.text) for chunk in chunks]
        scores = self.model.predict(pairs)
        for chunk, score in zip(chunks, scores, strict=False):
            chunk.rerank_score = float(score)
        chunks.sort(key=lambda item: item.rerank_score or 0.0, reverse=True)
        end_time = time.time()
        print(f"Rerank time: {end_time - start_time} seconds")
        return chunks
