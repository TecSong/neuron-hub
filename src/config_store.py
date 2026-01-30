from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional

from dotenv import load_dotenv

STORE_VERSION = 1
STORE_FILENAME = "kb_configs.json"


def _project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _store_path() -> Path:
    return _project_root() / "storage" / STORE_FILENAME


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _get_env_str(name: str, default: str) -> str:
    value = os.getenv(name)
    return value if value else default


def _get_env_int(name: str, default: int) -> int:
    value = os.getenv(name)
    if not value:
        return default
    try:
        return int(value)
    except ValueError:
        return default


def _get_env_float(name: str, default: float) -> float:
    value = os.getenv(name)
    if not value:
        return default
    try:
        return float(value)
    except ValueError:
        return default


def _normalize_kb_dir(value: str) -> str:
    if not value:
        return ""
    return str(Path(value).expanduser())


def _env_defaults() -> Dict[str, Any]:
    load_dotenv()
    return {
        "name": "Default KB",
        "description": "Imported from .env",
        "kb_dir": _normalize_kb_dir(_get_env_str("KB_DIR", "")),
        "ollama_embed_model": _get_env_str("OLLAMA_EMBED_MODEL", "bge-m3"),
        "rerank_model": _get_env_str("RERANK_MODEL", "BAAI/bge-reranker-v2-m3"),
        "chunk_size_tokens": _get_env_int("CHUNK_SIZE_TOKENS", 300),
        "chunk_overlap_ratio": _get_env_float("CHUNK_OVERLAP_RATIO", 0.15),
        "vector_top_n": _get_env_int("VECTOR_TOP_N", 8),
        "bm25_top_n": _get_env_int("BM25_TOP_N", 8),
        "top_k": _get_env_int("TOP_K", 4),
        "dedup_similarity_threshold": _get_env_float("DEDUP_SIMILARITY_THRESHOLD", 0.85),
        "vector_weight": _get_env_float("VECTOR_WEIGHT", 0.6),
        "bm25_weight": _get_env_float("BM25_WEIGHT", 0.4),
        "storage_namespace": "",
    }


def _build_default_config() -> Dict[str, Any]:
    now = _now_iso()
    defaults = _env_defaults()
    return {
        "id": "default",
        "name": defaults["name"],
        "description": defaults["description"],
        "kb_dir": defaults["kb_dir"],
        "ollama_embed_model": defaults["ollama_embed_model"],
        "rerank_model": defaults["rerank_model"],
        "chunk_size_tokens": defaults["chunk_size_tokens"],
        "chunk_overlap_ratio": defaults["chunk_overlap_ratio"],
        "vector_top_n": defaults["vector_top_n"],
        "bm25_top_n": defaults["bm25_top_n"],
        "top_k": defaults["top_k"],
        "dedup_similarity_threshold": defaults["dedup_similarity_threshold"],
        "vector_weight": defaults["vector_weight"],
        "bm25_weight": defaults["bm25_weight"],
        "storage_namespace": defaults["storage_namespace"],
        "created_at": now,
        "updated_at": now,
    }


def _init_store() -> Dict[str, Any]:
    default_config = _build_default_config()
    store = {
        "version": STORE_VERSION,
        "active_id": default_config["id"],
        "configs": [default_config],
    }
    save_store(store)
    return store


def load_store() -> Dict[str, Any]:
    path = _store_path()
    if not path.exists():
        return _init_store()
    try:
        with path.open("r", encoding="utf-8") as handle:
            data = json.load(handle)
        if not isinstance(data, dict):
            return _init_store()
        if "configs" not in data:
            return _init_store()
        return data
    except (OSError, json.JSONDecodeError):
        return _init_store()


def save_store(data: Dict[str, Any]) -> None:
    path = _store_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(data, handle, indent=2)


def get_active_config() -> Optional[Dict[str, Any]]:
    store = load_store()
    active_id = store.get("active_id")
    configs = store.get("configs", [])
    for config in configs:
        if config.get("id") == active_id:
            return config
    return None
