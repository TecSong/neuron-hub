from __future__ import annotations

import asyncio
import json
import threading
from typing import Any, Dict, List, Sequence, Tuple

import websockets
from websockets.exceptions import ConnectionClosed
from websockets.server import WebSocketServerProtocol

from .rag import RAGAgent
from .types import RetrievedChunk


def _serialize_chunk(chunk: RetrievedChunk) -> Dict[str, Any]:
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


def _coerce_history(raw: Any) -> List[Tuple[str, str]]:
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


def _parse_payload(message: str) -> Dict[str, Any]:
    try:
        payload = json.loads(message)
        if isinstance(payload, dict):
            return payload
    except json.JSONDecodeError:
        pass
    return {"question": message}


async def _stream_answer(
    websocket: WebSocketServerProtocol,
    agent: RAGAgent,
    question: str,
    history: Sequence[Tuple[str, str]],
    return_sources: bool,
) -> None:
    loop = asyncio.get_running_loop()
    queue: asyncio.Queue[tuple[str, Any]] = asyncio.Queue()
    answer_parts: List[str] = []

    def _worker() -> None:
        try:
            stream, chunks = agent.answer_stream(question, history=history)
            for token in stream:
                loop.call_soon_threadsafe(queue.put_nowait, ("token", token))
            loop.call_soon_threadsafe(queue.put_nowait, ("done", chunks))
        except Exception as exc:  # pragma: no cover - defensive
            loop.call_soon_threadsafe(queue.put_nowait, ("error", str(exc)))

    threading.Thread(target=_worker, daemon=True).start()

    while True:
        kind, payload = await queue.get()
        try:
            if kind == "token":
                answer_parts.append(payload)
                await websocket.send(json.dumps({"type": "token", "content": payload}))
            elif kind == "done":
                response: Dict[str, Any] = {
                    "type": "done",
                    "answer": "".join(answer_parts),
                }
                if return_sources:
                    response["sources"] = [_serialize_chunk(chunk) for chunk in payload]
                await websocket.send(json.dumps(response))
                break
            elif kind == "error":
                await websocket.send(json.dumps({"type": "error", "message": payload}))
                break
        except ConnectionClosed:
            break


async def _handle_connection(websocket: WebSocketServerProtocol, agent: RAGAgent) -> None:
    async for message in websocket:
        if isinstance(message, bytes):
            try:
                message = message.decode("utf-8")
            except UnicodeDecodeError:
                await websocket.send(
                    json.dumps({"type": "error", "message": "Invalid UTF-8 payload."})
                )
                continue
        payload = _parse_payload(message)
        question = str(payload.get("question", "")).strip()
        if not question:
            await websocket.send(json.dumps({"type": "error", "message": "Question is required."}))
            continue
        history = _coerce_history(payload.get("history"))
        return_sources = bool(payload.get("return_sources", True))
        await _stream_answer(websocket, agent, question, history, return_sources)


async def _serve(host: str, port: int) -> None:
    agent = RAGAgent()
    async with websockets.serve(
        lambda websocket: _handle_connection(websocket, agent),
        host,
        port,
    ):
        await asyncio.Future()


def serve(host: str, port: int) -> None:
    asyncio.run(_serve(host, port))
