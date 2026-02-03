from __future__ import annotations

import heapq
import json
import math
from typing import Dict, Iterable, List, Sequence, Tuple

from langchain_ollama import OllamaEmbeddings

from .config import settings
from .session_store import append_embedding_record, embeddings_path, get_metadata, load_events


def _cosine_similarity(left: Sequence[float], right: Sequence[float]) -> float:
    if not left or not right:
        return 0.0
    dot = 0.0
    left_norm = 0.0
    right_norm = 0.0
    for l, r in zip(left, right):
        dot += l * r
        left_norm += l * l
        right_norm += r * r
    if left_norm <= 0.0 or right_norm <= 0.0:
        return 0.0
    return dot / (math.sqrt(left_norm) * math.sqrt(right_norm))


def _build_turns(events: Iterable[Dict[str, str]]) -> List[Tuple[str, str]]:
    turns: List[Tuple[str, str]] = []
    pending_question: str | None = None
    for event in events:
        event_type = event.get("type")
        if event_type == "user":
            pending_question = str(event.get("content") or "").strip()
            continue
        if event_type == "assistant":
            answer = str(event.get("content") or "").strip()
            if pending_question and answer:
                turns.append((pending_question, answer))
            pending_question = None
    return turns


def _build_turn_text(question: str, answer: str) -> str:
    return f"User: {question}\nAssistant: {answer}"


def append_turn_embedding(session_id: str, question: str, answer: str) -> None:
    text = _build_turn_text(question, answer)
    embeddings = OllamaEmbeddings(
        model=settings.ollama_embed_model,
        base_url=settings.ollama_base_url,
    )
    vector = embeddings.embed_query(text)
    metadata = get_metadata(session_id) or {}
    turn_index = int(metadata.get("embedding_count") or 0)
    append_embedding_record(
        session_id,
        {"turn_index": turn_index, "text": text, "vector": vector},
    )


def _semantic_top_k_from_cache(question: str, session_id: str, top_k: int) -> List[int]:
    if top_k <= 0:
        return []
    path = embeddings_path(session_id)
    if not path.exists():
        return []
    embeddings = OllamaEmbeddings(
        model=settings.ollama_embed_model,
        base_url=settings.ollama_base_url,
    )
    query_vec = embeddings.embed_query(question)
    heap: List[Tuple[float, int]] = []
    try:
        with path.open("rb") as handle:
            for line in handle:
                try:
                    payload = json.loads(line.decode("utf-8"))
                except (json.JSONDecodeError, UnicodeDecodeError):
                    continue
                if not isinstance(payload, dict):
                    continue
                vector = payload.get("vector")
                if not isinstance(vector, list):
                    continue
                turn_index = payload.get("turn_index")
                if not isinstance(turn_index, int):
                    continue
                score = _cosine_similarity(query_vec, vector)
                if len(heap) < top_k:
                    heapq.heappush(heap, (score, turn_index))
                else:
                    if score > heap[0][0]:
                        heapq.heapreplace(heap, (score, turn_index))
    except OSError:
        return []
    heap.sort(key=lambda item: item[0], reverse=True)
    return [turn_index for _, turn_index in heap]


def load_session_history(session_id: str, question: str) -> List[Tuple[str, str]]:
    max_events = settings.session_semantic_max_events
    events = load_events(session_id, limit=max_events)
    turns = _build_turns(events)
    if not turns:
        return []
    recent_turns = turns[-settings.history_max_turns :]
    semantic_top_k = settings.session_semantic_top_k
    semantic_indices = _semantic_top_k_from_cache(question, session_id, semantic_top_k)
    recent_indices = list(range(max(0, len(turns) - len(recent_turns)), len(turns)))
    merged_indices = sorted({idx for idx in (recent_indices + semantic_indices) if 0 <= idx < len(turns)})
    return [turns[idx] for idx in merged_indices]
