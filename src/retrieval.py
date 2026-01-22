from __future__ import annotations

import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List

from langchain_community.vectorstores import FAISS
from langchain_ollama import OllamaEmbeddings
from rank_bm25 import BM25Okapi

from .config import settings
from .types import ChunkRecord, RetrievedChunk
from .utils import simple_tokenize


@dataclass(frozen=True)
class RetrieverResources:
    vector_store: FAISS
    bm25: BM25Okapi
    records: List[ChunkRecord]


def _load_records(records_path: Path) -> List[ChunkRecord]:
    if not records_path.exists():
        raise FileNotFoundError("Missing records. Run ingest first.")
    with records_path.open("rb") as handle:
        raw_records = pickle.load(handle)
    return [ChunkRecord(**record) for record in raw_records]


def _load_bm25(tokens_path: Path) -> BM25Okapi:
    if not tokens_path.exists():
        raise FileNotFoundError("Missing BM25 data. Run ingest first.")
    with tokens_path.open("rb") as handle:
        tokenized_docs = pickle.load(handle)
    return BM25Okapi(tokenized_docs)


def _load_vector_store() -> FAISS:
    if not settings.faiss_dir.exists():
        raise FileNotFoundError("Missing FAISS index. Run ingest first.")
    embeddings = OllamaEmbeddings(
        model=settings.ollama_embed_model,
        base_url=settings.ollama_base_url,
    )
    return FAISS.load_local(
        str(settings.faiss_dir),
        embeddings,
        allow_dangerous_deserialization=True,
    )


def load_resources() -> RetrieverResources:
    records = _load_records(settings.records_path)
    bm25 = _load_bm25(settings.bm25_tokens_path)
    vector_store = _load_vector_store()
    return RetrieverResources(vector_store=vector_store, bm25=bm25, records=records)


class HybridRetriever:
    def __init__(self, resources: RetrieverResources) -> None:
        self.vector_store = resources.vector_store
        self.bm25 = resources.bm25
        self.records = resources.records

    def _vector_similarity(self, distance: float) -> float:
        return 1.0 / (1.0 + distance)

    def _min_max_normalize(self, scores: Dict[int, float]) -> Dict[int, float]:
        if not scores:
            return {}
        values = list(scores.values())
        min_score = min(values)
        max_score = max(values)
        if abs(max_score - min_score) < 1e-9:
            return {key: 1.0 for key in scores}
        return {key: (value - min_score) / (max_score - min_score) for key, value in scores.items()}

    def _jaccard_similarity(self, left: set[str], right: set[str]) -> float:
        if not left or not right:
            return 0.0
        intersection = left.intersection(right)
        union = left.union(right)
        if not union:
            return 0.0
        return len(intersection) / len(union)

    def _deduplicate(self, chunks: List[RetrievedChunk]) -> List[RetrievedChunk]:
        threshold = settings.dedup_similarity_threshold
        if threshold <= 0.0:
            return chunks
        unique: List[RetrievedChunk] = []
        token_sets: List[set[str]] = []
        for chunk in chunks:
            tokens = set(simple_tokenize(chunk.record.text))
            is_duplicate = False
            for existing_tokens in token_sets:
                if self._jaccard_similarity(tokens, existing_tokens) >= threshold:
                    is_duplicate = True
                    print(f"Duplicate chunk: {chunk.record.chunk_id}")
                    break
            if not is_duplicate:
                unique.append(chunk)
                token_sets.append(tokens)
        return unique

    def _vector_search(self, query: str) -> Dict[int, float]:
        results = self.vector_store.similarity_search_with_score(query, k=settings.vector_top_n)
        scores: Dict[int, float] = {}
        for doc, distance in results:
            chunk_id = int(doc.metadata.get("chunk_id", -1))
            if chunk_id < 0:
                continue
            scores[chunk_id] = self._vector_similarity(distance)
        return scores

    def _bm25_search(self, query: str) -> Dict[int, float]:
        query_tokens = simple_tokenize(query)
        bm25_scores = self.bm25.get_scores(query_tokens)
        if bm25_scores is None or len(bm25_scores) == 0:
            return {}
        top_indices = sorted(range(len(bm25_scores)), key=lambda i: bm25_scores[i], reverse=True)
        scores: Dict[int, float] = {}
        for idx in top_indices[: settings.bm25_top_n]:
            scores[idx] = float(bm25_scores[idx])
        return scores

    def retrieve(self, query: str) -> List[RetrievedChunk]:
        vector_scores = self._vector_search(query)
        bm25_scores = self._bm25_search(query)

        vector_norm = self._min_max_normalize(vector_scores)
        bm25_norm = self._min_max_normalize(bm25_scores)

        all_ids = set(vector_norm) | set(bm25_norm)
        fused: List[RetrievedChunk] = []
        for chunk_id in all_ids:
            record = self.records[chunk_id]
            fused_score = (
                settings.vector_weight * vector_norm.get(chunk_id, 0.0)
                + settings.bm25_weight * bm25_norm.get(chunk_id, 0.0)
            )
            fused.append(
                RetrievedChunk(
                    record=record,
                    fused_score=fused_score,
                    vector_score=vector_scores.get(chunk_id, 0.0),
                    bm25_score=bm25_scores.get(chunk_id, 0.0),
                )
            )

        fused.sort(key=lambda item: item.fused_score, reverse=True)
        deduped = self._deduplicate(fused)
        return deduped[: settings.top_k]
