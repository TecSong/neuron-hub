from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

from .config_store import get_active_config


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


def _coerce_int(value: object, default: int) -> int:
    if value is None:
        return default
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _coerce_float(value: object, default: float) -> float:
    if value is None:
        return default
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _coerce_str(value: object, default: str) -> str:
    if value is None:
        return default
    if isinstance(value, str):
        return value
    return default


@dataclass(frozen=True)
class Settings:
    project_root: Path
    kb_dir: Path
    storage_dir: Path
    faiss_dir: Path
    records_path: Path
    bm25_tokens_path: Path
    deepseek_api_key: str
    deepseek_api_base: str
    deepseek_model: str
    ollama_embed_model: str
    ollama_base_url: str
    rerank_model: str
    chunk_size_tokens: int
    chunk_overlap_ratio: float
    chunk_overlap_tokens: int
    vector_top_n: int
    bm25_top_n: int
    top_k: int
    dedup_similarity_threshold: float
    vector_weight: float
    bm25_weight: float
    temperature: float
    max_tokens: int
    context_window_tokens: int
    history_max_turns: int
    session_semantic_top_k: int
    session_semantic_max_events: int
    supported_extensions: tuple[str, ...]

    @classmethod
    def from_env(cls) -> "Settings":
        load_dotenv()
        project_root = Path(__file__).resolve().parents[1]
        kb_dir = Path(_get_env_str("KB_DIR", "")).expanduser()

        ollama_embed_model = _get_env_str("OLLAMA_EMBED_MODEL", "bge-m3")
        rerank_model = _get_env_str("RERANK_MODEL", "BAAI/bge-reranker-v2-m3")

        chunk_size_tokens = _get_env_int("CHUNK_SIZE_TOKENS", 300)
        overlap_ratio = _get_env_float("CHUNK_OVERLAP_RATIO", 0.15)

        vector_top_n = _get_env_int("VECTOR_TOP_N", 8)
        bm25_top_n = _get_env_int("BM25_TOP_N", 8)
        top_k = _get_env_int("TOP_K", 4)
        dedup_similarity_threshold = _get_env_float("DEDUP_SIMILARITY_THRESHOLD", 0.85)

        vector_weight = _get_env_float("VECTOR_WEIGHT", 0.6)
        bm25_weight = _get_env_float("BM25_WEIGHT", 0.4)

        storage_namespace = ""
        active_config = get_active_config()
        if active_config:
            kb_dir_value = _coerce_str(active_config.get("kb_dir"), "")
            if kb_dir_value.strip():
                kb_dir = Path(kb_dir_value).expanduser()

            ollama_embed_model = _coerce_str(
                active_config.get("ollama_embed_model"), ollama_embed_model
            )
            rerank_model = _coerce_str(active_config.get("rerank_model"), rerank_model)

            chunk_size_tokens = _coerce_int(
                active_config.get("chunk_size_tokens"), chunk_size_tokens
            )
            overlap_ratio = _coerce_float(
                active_config.get("chunk_overlap_ratio"), overlap_ratio
            )

            vector_top_n = _coerce_int(active_config.get("vector_top_n"), vector_top_n)
            bm25_top_n = _coerce_int(active_config.get("bm25_top_n"), bm25_top_n)
            top_k = _coerce_int(active_config.get("top_k"), top_k)
            dedup_similarity_threshold = _coerce_float(
                active_config.get("dedup_similarity_threshold"), dedup_similarity_threshold
            )

            vector_weight = _coerce_float(active_config.get("vector_weight"), vector_weight)
            bm25_weight = _coerce_float(active_config.get("bm25_weight"), bm25_weight)

            storage_namespace = _coerce_str(active_config.get("storage_namespace"), "").strip()

        storage_root = project_root / "storage"
        storage_dir = storage_root / storage_namespace if storage_namespace else storage_root
        faiss_dir = storage_dir / "faiss_index"
        records_path = storage_dir / "records.pkl"
        bm25_tokens_path = storage_dir / "bm25_tokens.pkl"

        weight_sum = vector_weight + bm25_weight
        if weight_sum <= 0:
            vector_weight, bm25_weight = 0.6, 0.4
            weight_sum = 1.0
        vector_weight /= weight_sum
        bm25_weight /= weight_sum

        overlap_tokens = max(0, int(chunk_size_tokens * overlap_ratio))

        return cls(
            project_root=project_root,
            kb_dir=kb_dir,
            storage_dir=storage_dir,
            faiss_dir=faiss_dir,
            records_path=records_path,
            bm25_tokens_path=bm25_tokens_path,
            deepseek_api_key=_get_env_str("DEEPSEEK_API_KEY", ""),
            deepseek_api_base=_get_env_str("DEEPSEEK_API_BASE", "https://api.deepseek.com/v1"),
            deepseek_model=_get_env_str("DEEPSEEK_MODEL", "deepseek-chat"),
            ollama_embed_model=ollama_embed_model,
            ollama_base_url=_get_env_str("OLLAMA_BASE_URL", "http://localhost:11434"),
            rerank_model=rerank_model,
            chunk_size_tokens=chunk_size_tokens,
            chunk_overlap_ratio=overlap_ratio,
            chunk_overlap_tokens=overlap_tokens,
            vector_top_n=vector_top_n,
            bm25_top_n=bm25_top_n,
            top_k=top_k,
            dedup_similarity_threshold=dedup_similarity_threshold,
            vector_weight=vector_weight,
            bm25_weight=bm25_weight,
            temperature=_get_env_float("TEMPERATURE", 0.2),
            max_tokens=_get_env_int("MAX_TOKENS", 512),
            context_window_tokens=_get_env_int("CONTEXT_WINDOW_TOKENS", 32768),
            history_max_turns=_get_env_int("HISTORY_MAX_TURNS", 5),
            session_semantic_top_k=_get_env_int("SESSION_SEMANTIC_TOP_K", 6),
            session_semantic_max_events=_get_env_int("SESSION_SEMANTIC_MAX_EVENTS", 200),
            supported_extensions=(".txt", ".md"),
        )


class SettingsProxy:
    def __init__(self) -> None:
        self._settings: Settings | None = None

    def reload(self) -> Settings:
        self._settings = Settings.from_env()
        return self._settings

    def __getattr__(self, name: str):
        if self._settings is None:
            self.reload()
        return getattr(self._settings, name)


settings = SettingsProxy()
