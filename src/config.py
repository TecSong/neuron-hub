from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv


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
    history_max_turns: int
    supported_extensions: tuple[str, ...]

    @classmethod
    def from_env(cls) -> "Settings":
        load_dotenv()
        project_root = Path(__file__).resolve().parents[1]
        kb_dir = Path(_get_env_str("KB_DIR", "")).expanduser()
        storage_dir = project_root / "storage"
        faiss_dir = storage_dir / "faiss_index"
        records_path = storage_dir / "records.pkl"
        bm25_tokens_path = storage_dir / "bm25_tokens.pkl"

        vector_weight = _get_env_float("VECTOR_WEIGHT", 0.6)
        bm25_weight = _get_env_float("BM25_WEIGHT", 0.4)
        weight_sum = vector_weight + bm25_weight
        if weight_sum <= 0:
            vector_weight, bm25_weight = 0.6, 0.4
            weight_sum = 1.0
        vector_weight /= weight_sum
        bm25_weight /= weight_sum

        chunk_size_tokens = _get_env_int("CHUNK_SIZE_TOKENS", 300)
        overlap_ratio = _get_env_float("CHUNK_OVERLAP_RATIO", 0.15)
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
            ollama_embed_model=_get_env_str("OLLAMA_EMBED_MODEL", "bge-m3"),
            ollama_base_url=_get_env_str("OLLAMA_BASE_URL", "http://localhost:11434"),
            rerank_model=_get_env_str("RERANK_MODEL", "BAAI/bge-reranker-v2-m3"),
            chunk_size_tokens=chunk_size_tokens,
            chunk_overlap_ratio=overlap_ratio,
            chunk_overlap_tokens=overlap_tokens,
            vector_top_n=_get_env_int("VECTOR_TOP_N", 8),
            bm25_top_n=_get_env_int("BM25_TOP_N", 8),
            top_k=_get_env_int("TOP_K", 4),
            dedup_similarity_threshold=_get_env_float("DEDUP_SIMILARITY_THRESHOLD", 0.85),
            vector_weight=vector_weight,
            bm25_weight=bm25_weight,
            temperature=_get_env_float("TEMPERATURE", 0.2),
            max_tokens=_get_env_int("MAX_TOKENS", 512),
            history_max_turns=_get_env_int("HISTORY_MAX_TURNS", 5),
            supported_extensions=(".txt", ".md"),
        )


settings = Settings.from_env()
