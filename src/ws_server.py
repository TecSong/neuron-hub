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
from .ws_protocol import coerce_history, parse_payload, serialize_chunk


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
                    response["sources"] = [serialize_chunk(chunk) for chunk in payload]
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
