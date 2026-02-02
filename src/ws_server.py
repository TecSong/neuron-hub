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
from .utils import get_token_counter
from .ws_protocol import coerce_history, parse_payload, serialize_chunk

_TOKEN_COUNTER = get_token_counter()


def _finalize_usage(usage: Dict[str, Any], answer: str) -> Dict[str, Any]:
    prompt_tokens = int(usage.get("prompt_tokens") or 0)
    context_window = int(usage.get("context_window_tokens") or 0)
    estimated_remaining = usage.get("remaining_tokens")
    completion_tokens = int(_TOKEN_COUNTER(answer))
    total_tokens = prompt_tokens + completion_tokens
    updated = dict(usage)
    updated["completion_tokens"] = completion_tokens
    updated["total_tokens"] = total_tokens
    if estimated_remaining is not None:
        updated["remaining_tokens_estimate"] = int(estimated_remaining)
    if context_window:
        updated["remaining_tokens"] = max(context_window - total_tokens, 0)
    return updated


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
            stream, chunks, usage = agent.answer_stream(question, history=history)
            if return_sources:
                loop.call_soon_threadsafe(queue.put_nowait, ("sources", chunks))
            for token in stream:
                loop.call_soon_threadsafe(queue.put_nowait, ("token", token))
            loop.call_soon_threadsafe(
                queue.put_nowait, ("done", {"chunks": chunks, "usage": usage})
            )
        except Exception as exc:  # pragma: no cover - defensive
            loop.call_soon_threadsafe(queue.put_nowait, ("error", str(exc)))

    threading.Thread(target=_worker, daemon=True).start()

    while True:
        kind, payload = await queue.get()
        try:
            if kind == "token":
                answer_parts.append(payload)
                await websocket.send(json.dumps({"type": "token", "content": payload}))
            elif kind == "sources":
                await websocket.send(
                    json.dumps({"type": "sources", "sources": [serialize_chunk(c) for c in payload]})
                )
            elif kind == "done":
                chunks = payload.get("chunks", [])
                answer = "".join(answer_parts)
                response: Dict[str, Any] = {
                    "type": "done",
                    "answer": answer,
                }
                usage = payload.get("usage")
                if usage:
                    response["usage"] = _finalize_usage(usage, answer)
                if return_sources:
                    response["sources"] = [serialize_chunk(chunk) for chunk in chunks]
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
        payload = parse_payload(message)
        question = str(payload.get("question", "")).strip()
        if not question:
            await websocket.send(json.dumps({"type": "error", "message": "Question is required."}))
            continue
        history = coerce_history(payload.get("history"))
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
