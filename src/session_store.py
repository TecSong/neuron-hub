from __future__ import annotations

import json
import os
import threading
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from .config import settings

SCHEMA_VERSION = 1

_LOCKS: dict[str, threading.Lock] = {}
_LOCKS_GUARD = threading.Lock()


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _sessions_dir() -> Path:
    return settings.storage_dir / "sessions"


def _session_meta_path(session_id: str) -> Path:
    return _sessions_dir() / f"{session_id}.json"


def _session_events_path(session_id: str) -> Path:
    return _sessions_dir() / f"{session_id}.jsonl"


def _session_embeddings_path(session_id: str) -> Path:
    return _sessions_dir() / f"{session_id}.emb.jsonl"


def _get_lock(session_id: str) -> threading.Lock:
    with _LOCKS_GUARD:
        lock = _LOCKS.get(session_id)
        if lock is None:
            lock = threading.Lock()
            _LOCKS[session_id] = lock
        return lock


def _atomic_write_json(path: Path, data: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    with tmp_path.open("w", encoding="utf-8") as handle:
        json.dump(data, handle, ensure_ascii=False, indent=2)
    os.replace(tmp_path, path)


def _load_metadata(session_id: str) -> Optional[Dict[str, Any]]:
    path = _session_meta_path(session_id)
    if not path.exists():
        return None
    try:
        with path.open("r", encoding="utf-8") as handle:
            data = json.load(handle)
        if isinstance(data, dict):
            return data
    except (OSError, json.JSONDecodeError):
        return None
    return None


def get_metadata(session_id: str) -> Optional[Dict[str, Any]]:
    return _load_metadata(session_id)


def _build_metadata(session_id: str) -> Dict[str, Any]:
    now = _now_iso()
    return {
        "schema_version": SCHEMA_VERSION,
        "session_id": session_id,
        "created_at": now,
        "updated_at": now,
        "event_file": str(_session_events_path(session_id)),
        "embedding_file": str(_session_embeddings_path(session_id)),
        "event_count": 0,
        "embedding_count": 0,
        "offsets": [],
    }


def new_session_id() -> str:
    return uuid.uuid4().hex


def ensure_session(session_id: str) -> Dict[str, Any]:
    lock = _get_lock(session_id)
    with lock:
        metadata = _load_metadata(session_id)
        if metadata is None:
            metadata = _build_metadata(session_id)
            _atomic_write_json(_session_meta_path(session_id), metadata)
        return metadata


@dataclass
class AppendResult:
    offset: int
    event_count: int


def append_event(session_id: str, event: Dict[str, Any]) -> AppendResult:
    lock = _get_lock(session_id)
    with lock:
        metadata = _load_metadata(session_id)
        if metadata is None:
            metadata = _build_metadata(session_id)
        events_path = _session_events_path(session_id)
        events_path.parent.mkdir(parents=True, exist_ok=True)
        payload = json.dumps(event, ensure_ascii=False).encode("utf-8") + b"\n"
        with events_path.open("ab") as handle:
            handle.seek(0, os.SEEK_END)
            offset = handle.tell()
            handle.write(payload)
        offsets: List[int] = list(metadata.get("offsets") or [])
        offsets.append(int(offset))
        event_count = int(metadata.get("event_count") or 0) + 1
        metadata["offsets"] = offsets
        metadata["event_count"] = event_count
        metadata["updated_at"] = _now_iso()
        _atomic_write_json(_session_meta_path(session_id), metadata)
        return AppendResult(offset=offset, event_count=event_count)


def list_sessions() -> List[Dict[str, Any]]:
    sessions_dir = _sessions_dir()
    if not sessions_dir.exists():
        return []
    sessions: List[Dict[str, Any]] = []
    for path in sessions_dir.glob("*.json"):
        if not path.is_file():
            continue
        session_id = path.stem
        metadata = _load_metadata(session_id)
        if not metadata:
            continue
        sessions.append(
            {
                "session_id": metadata.get("session_id", session_id),
                "created_at": metadata.get("created_at"),
                "updated_at": metadata.get("updated_at"),
                "event_count": metadata.get("event_count", 0),
            }
        )
    sessions.sort(key=lambda item: item.get("updated_at") or "", reverse=True)
    return sessions


def load_events(session_id: str, limit: Optional[int] = None) -> List[Dict[str, Any]]:
    metadata = _load_metadata(session_id)
    if metadata is None:
        return []
    offsets: List[int] = list(metadata.get("offsets") or [])
    if limit is not None and limit > 0:
        offsets = offsets[-limit:]
    events_path = _session_events_path(session_id)
    if not events_path.exists():
        return []
    events: List[Dict[str, Any]] = []
    try:
        with events_path.open("rb") as handle:
            for offset in offsets:
                handle.seek(int(offset))
                line = handle.readline()
                if not line:
                    continue
                try:
                    payload = json.loads(line.decode("utf-8"))
                except (json.JSONDecodeError, UnicodeDecodeError):
                    continue
                if isinstance(payload, dict):
                    events.append(payload)
    except OSError:
        return []
    return events


def embeddings_path(session_id: str) -> Path:
    return _session_embeddings_path(session_id)


def append_embedding_record(session_id: str, record: Dict[str, Any]) -> int:
    lock = _get_lock(session_id)
    with lock:
        metadata = _load_metadata(session_id)
        if metadata is None:
            metadata = _build_metadata(session_id)
        path = _session_embeddings_path(session_id)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = json.dumps(record, ensure_ascii=False).encode("utf-8") + b"\n"
        with path.open("ab") as handle:
            handle.seek(0, os.SEEK_END)
            handle.write(payload)
        embedding_count = int(metadata.get("embedding_count") or 0) + 1
        metadata["embedding_count"] = embedding_count
        metadata["updated_at"] = _now_iso()
        _atomic_write_json(_session_meta_path(session_id), metadata)
        return embedding_count
