from __future__ import annotations

import asyncio
import json
import os
import threading
import uuid
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional, Sequence

from fastapi import FastAPI, Request, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse, Response

from .agent_manager import AgentManager
from .config import settings
from .config_store import get_active_config, load_store, save_store
from .ingest import build_indexes
from .session_history import append_turn_embedding, load_session_history
from .session_store import append_event, ensure_session, list_sessions, new_session_id
from .types import RetrievedChunk
from .ws_protocol import parse_payload, serialize_chunk
from .utils import get_token_counter


WEB_DIR = Path(__file__).resolve().parents[1] / "web"
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


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _coerce_int(value: object, default: int) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _coerce_float(value: object, default: float) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _normalize_kb_dir(value: str) -> str:
    if not value:
        return ""
    return str(Path(value).expanduser())


ALLOWED_FIELDS = {
    "name",
    "description",
    "kb_dir",
    "ollama_embed_model",
    "rerank_model",
    "chunk_size_tokens",
    "chunk_overlap_ratio",
    "vector_top_n",
    "bm25_top_n",
    "top_k",
    "dedup_similarity_threshold",
    "vector_weight",
    "bm25_weight",
}


def _apply_payload(config: Dict[str, Any], payload: Dict[str, Any]) -> Dict[str, Any]:
    updated = dict(config)
    for field in ALLOWED_FIELDS:
        if field not in payload:
            continue
        value = payload.get(field)
        if field == "kb_dir":
            updated[field] = _normalize_kb_dir(str(value or ""))
        elif field in {
            "chunk_size_tokens",
            "vector_top_n",
            "bm25_top_n",
            "top_k",
        }:
            updated[field] = _coerce_int(value, updated.get(field, 0))
        elif field in {
            "chunk_overlap_ratio",
            "dedup_similarity_threshold",
            "vector_weight",
            "bm25_weight",
        }:
            updated[field] = _coerce_float(value, updated.get(field, 0.0))
        else:
            updated[field] = str(value or "")
    return updated


def _new_config(payload: Dict[str, Any], base: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    base = base or {}
    config_id = uuid.uuid4().hex[:8]
    now = _now_iso()
    config = {
        "id": config_id,
        "name": "New KB",
        "description": "",
        "kb_dir": "",
        "ollama_embed_model": base.get("ollama_embed_model", "bge-m3"),
        "rerank_model": base.get("rerank_model", "BAAI/bge-reranker-v2-m3"),
        "chunk_size_tokens": base.get("chunk_size_tokens", 300),
        "chunk_overlap_ratio": base.get("chunk_overlap_ratio", 0.15),
        "vector_top_n": base.get("vector_top_n", 8),
        "bm25_top_n": base.get("bm25_top_n", 8),
        "top_k": base.get("top_k", 4),
        "dedup_similarity_threshold": base.get("dedup_similarity_threshold", 0.85),
        "vector_weight": base.get("vector_weight", 0.6),
        "bm25_weight": base.get("bm25_weight", 0.4),
        "storage_namespace": f"kb_{config_id}",
        "created_at": now,
        "updated_at": now,
    }
    config = _apply_payload(config, payload)
    config["updated_at"] = now
    return config


async def _get_json(request: Request) -> Dict[str, Any]:
    try:
        payload = await request.json()
        if isinstance(payload, dict):
            return payload
    except Exception:
        pass
    return {}


async def _stream_answer(
    websocket: WebSocket,
    stream: Any,
    chunks: Sequence[RetrievedChunk],
    usage: Dict[str, Any],
    meta: Dict[str, Any],
    question: str,
    return_sources: bool,
    session_id: str,
) -> None:
    loop = asyncio.get_running_loop()
    queue: asyncio.Queue[tuple[str, Any]] = asyncio.Queue()
    answer_parts: list[str] = []

    def _worker() -> None:
        try:
            if return_sources:
                loop.call_soon_threadsafe(queue.put_nowait, ("sources", list(chunks)))
            for token in stream:
                loop.call_soon_threadsafe(queue.put_nowait, ("token", token))
            loop.call_soon_threadsafe(
                queue.put_nowait,
                ("done", {"chunks": list(chunks), "usage": usage, "meta": meta}),
            )
        except Exception as exc:
            loop.call_soon_threadsafe(queue.put_nowait, ("error", str(exc)))

    threading.Thread(target=_worker, daemon=True).start()

    while True:
        kind, payload = await queue.get()
        if kind == "token":
            answer_parts.append(payload)
            await websocket.send_text(json.dumps({"type": "token", "content": payload}))
        elif kind == "sources":
            await websocket.send_text(
                json.dumps({"type": "sources", "sources": [serialize_chunk(c) for c in payload]})
            )
        elif kind == "done":
            chunks = payload.get("chunks", [])
            answer = "".join(answer_parts)
            sources_payload = [serialize_chunk(chunk) for chunk in chunks] if return_sources else []
            response: Dict[str, Any] = {"type": "done", "answer": answer}
            meta = payload.get("meta") or {}
            usage = payload.get("usage")
            finalized_usage = _finalize_usage(usage, answer) if usage else None
            if finalized_usage:
                response["usage"] = finalized_usage
            if return_sources:
                response["sources"] = sources_payload
            response["session_id"] = session_id
            if meta.get("history_compressed"):
                response["history_compressed"] = True
            if meta.get("session_reset"):
                response["session_reset"] = True
            try:
                append_event(
                    session_id,
                    {
                        "session_id": session_id,
                        "ts": _now_iso(),
                        "type": "assistant",
                        "content": answer,
                        "sources": sources_payload,
                        "usage": finalized_usage,
                    },
                )
                append_turn_embedding(session_id, question, answer)
            except Exception:
                pass
            await websocket.send_text(json.dumps(response))
            break
        elif kind == "error":
            try:
                append_event(
                    session_id,
                    {
                        "session_id": session_id,
                        "ts": _now_iso(),
                        "type": "error",
                        "message": payload,
                    },
                )
            except Exception:
                pass
            await websocket.send_text(json.dumps({"type": "error", "message": payload}))
            break


def create_app() -> FastAPI:
    app = FastAPI()
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    agent_manager = AgentManager()

    @app.get("/api/health")
    async def health() -> JSONResponse:
        active = get_active_config() or {}
        return JSONResponse(
            {
                "status": "ok",
                "active_id": active.get("id"),
                "active_name": active.get("name"),
            }
        )

    @app.get("/api/configs")
    async def list_configs() -> JSONResponse:
        return JSONResponse(load_store())

    @app.get("/api/configs/active")
    async def active_config() -> JSONResponse:
        return JSONResponse({"active": get_active_config()})

    @app.get("/api/sessions")
    async def sessions() -> JSONResponse:
        return JSONResponse({"sessions": list_sessions()})

    @app.post("/api/configs")
    async def create_config(request: Request) -> JSONResponse:
        payload = await _get_json(request)
        store = load_store()
        base = get_active_config() or {}
        config = _new_config(payload, base=base)
        store.setdefault("configs", []).append(config)
        if payload.get("activate"):
            store["active_id"] = config["id"]
            settings.reload()
            agent_manager.reset()
        save_store(store)
        return JSONResponse(config, status_code=201)

    @app.put("/api/configs/{config_id}")
    async def update_config(config_id: str, request: Request) -> JSONResponse:
        payload = await _get_json(request)
        store = load_store()
        configs = store.get("configs", [])
        for idx, config in enumerate(configs):
            if config.get("id") != config_id:
                continue
            updated = _apply_payload(config, payload)
            updated["updated_at"] = _now_iso()
            configs[idx] = updated
            store["configs"] = configs
            save_store(store)
            if store.get("active_id") == config_id:
                settings.reload()
                agent_manager.reset()
            return JSONResponse(updated)
        return JSONResponse({"error": "Config not found."}, status_code=404)

    @app.delete("/api/configs/{config_id}")
    async def delete_config(config_id: str) -> JSONResponse:
        store = load_store()
        if store.get("active_id") == config_id:
            return JSONResponse({"error": "Cannot delete active config."}, status_code=400)
        configs = [config for config in store.get("configs", []) if config.get("id") != config_id]
        store["configs"] = configs
        save_store(store)
        return JSONResponse({"ok": True})

    @app.post("/api/configs/{config_id}/activate")
    async def activate_config(config_id: str) -> JSONResponse:
        store = load_store()
        configs = store.get("configs", [])
        if not any(config.get("id") == config_id for config in configs):
            return JSONResponse({"error": "Config not found."}, status_code=404)
        store["active_id"] = config_id
        save_store(store)
        settings.reload()
        agent_manager.reset()
        return JSONResponse({"active_id": config_id})

    @app.post("/api/ingest")
    async def ingest() -> JSONResponse:
        settings.reload()
        try:
            stats = build_indexes()
        except Exception as exc:
            return JSONResponse({"error": str(exc)}, status_code=400)
        agent_manager.reset()
        return JSONResponse(asdict(stats))

    @app.websocket("/ws/chat")
    async def chat(websocket: WebSocket) -> None:
        await websocket.accept()
        session_id: str | None = None
        while True:
            try:
                message = await websocket.receive_text()
            except WebSocketDisconnect:
                break
            payload = parse_payload(message)
            incoming_session_id = payload.get("session_id")
            if incoming_session_id:
                session_id = str(incoming_session_id)
            if session_id is None:
                session_id = new_session_id()
            question = str(payload.get("question", "")).strip()
            if not question:
                await websocket.send_text(
                    json.dumps({"type": "error", "message": "Question is required."})
                )
                continue
            ensure_session(session_id)
            history = load_session_history(session_id, question)
            return_sources = bool(payload.get("return_sources", True))
            agent = agent_manager.get_agent()
            stream, chunks, usage, meta = agent.answer_stream(question, history=history)
            if meta.get("session_reset"):
                old_session_id = session_id
                session_id = new_session_id()
                ensure_session(session_id)
                try:
                    append_event(
                        old_session_id,
                        {
                            "session_id": old_session_id,
                            "ts": _now_iso(),
                            "type": "reset",
                            "new_session_id": session_id,
                        },
                    )
                except Exception:
                    pass
            try:
                append_event(
                    session_id,
                    {
                        "session_id": session_id,
                        "ts": _now_iso(),
                        "type": "user",
                        "content": question,
                        "history": history,
                    },
                )
            except Exception:
                pass
            await _stream_answer(
                websocket,
                stream,
                chunks,
                usage,
                meta,
                question,
                return_sources,
                session_id,
            )

    @app.get("/web-config.js")
    async def web_config(request: Request) -> Response:
        host = request.url.hostname or "localhost"
        protocol = "wss" if request.url.scheme == "https" else "ws"
        ws_url = f"{protocol}://{host}{request.url.port and f':{request.url.port}' or ''}/ws/chat"
        script = f"(function(){{window.PKBA_WS_URL='{ws_url}';}})();"
        return Response(script, media_type="application/javascript")

    @app.get("/")
    async def index() -> FileResponse:
        return FileResponse(WEB_DIR / "index.html", headers={"Cache-Control": "no-store"})

    @app.get("/{path:path}")
    async def static_proxy(path: str) -> FileResponse:
        target = WEB_DIR / path
        if target.exists() and target.is_file():
            return FileResponse(target, headers={"Cache-Control": "no-store"})
        return FileResponse(WEB_DIR / "index.html", headers={"Cache-Control": "no-store"})

    return app


app = create_app()


def run(host: str = "0.0.0.0", port: int = 5000) -> None:
    import uvicorn

    uvicorn.run("src.web_app:app", host=host, port=port, reload=False)


if __name__ == "__main__":
    run()
