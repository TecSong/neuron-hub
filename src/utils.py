from __future__ import annotations

import re
from pathlib import Path
from typing import Callable, List

from langchain_text_splitters import RecursiveCharacterTextSplitter


def read_text_file(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="ignore")


def normalize_text(text: str) -> str:
    lines = [line.rstrip() for line in text.splitlines()]
    return "\n".join(lines).strip()


def simple_tokenize(text: str) -> List[str]:
    try:
        import jieba

        tokens = [token.strip().lower() for token in jieba.lcut(text)]
        return [token for token in tokens if token]
    except Exception:
        return re.findall(r"\w+", text.lower())


def get_token_counter() -> Callable[[str], int]:
    try:
        import tiktoken

        encoder = tiktoken.get_encoding("cl100k_base")
        return lambda text: len(encoder.encode(text))
    except Exception:
        return lambda text: len(text.split())


def build_text_splitter(chunk_size_tokens: int, overlap_tokens: int) -> RecursiveCharacterTextSplitter:
    return RecursiveCharacterTextSplitter(
        chunk_size=chunk_size_tokens,
        chunk_overlap=overlap_tokens,
        length_function=get_token_counter(),
        separators=["\n\n", "\n", " ", ""],
    )
