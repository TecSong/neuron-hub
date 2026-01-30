from __future__ import annotations

import json
from typing import Any, Dict, List, Tuple

from .types import RetrievedChunk


def serialize_chunk(chunk: RetrievedChunk) -> Dict[str, Any]:
    record = chunk.record
    return {
        "chunk_id": record.chunk_id,
        "source": record.source,
        "text": record.text,
        "doc_id": record.doc_id,
        "position": record.position,
        "fused_score": float(chunk.fused_score),
        "vector_score": float(chunk.vector_score),
        "bm25_score": float(chunk.bm25_score),
        "rerank_score": float(chunk.rerank_score) if chunk.rerank_score is not None else None,
    }


def coerce_history(raw: Any) -> List[Tuple[str, str]]:
    if not raw:
        return []
    history: List[Tuple[str, str]] = []
    if isinstance(raw, list):
        for item in raw:
            if isinstance(item, (list, tuple)) and len(item) == 2:
                history.append((str(item[0]), str(item[1])))
                continue
            if isinstance(item, dict):
                question = item.get("question")
                answer = item.get("answer")
                if question is not None and answer is not None:
                    history.append((str(question), str(answer)))
    return history


def parse_payload(message: str) -> Dict[str, Any]:
    try:
        payload = json.loads(message)
        if isinstance(payload, dict):
            return payload
    except json.JSONDecodeError:
        pass
    return {"question": message}
