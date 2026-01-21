from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ChunkRecord:
    chunk_id: int
    text: str
    source: str
    doc_id: int
    position: int


@dataclass
class RetrievedChunk:
    record: ChunkRecord
    fused_score: float
    vector_score: float
    bm25_score: float
    rerank_score: float | None = None
