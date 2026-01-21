from __future__ import annotations

import pickle
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, List

from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_ollama import OllamaEmbeddings

from .config import settings
from .types import ChunkRecord
from .utils import build_text_splitter, normalize_text, read_text_file, simple_tokenize


@dataclass(frozen=True)
class IngestStats:
    file_count: int
    chunk_count: int


def _is_hidden(path: Path) -> bool:
    return any(part.startswith(".") for part in path.parts)


def _gather_files(kb_dir: Path) -> List[Path]:
    files: List[Path] = []
    for ext in settings.supported_extensions:
        files.extend(kb_dir.rglob(f"*{ext}"))
    unique = {path for path in files if path.is_file() and not _is_hidden(path)}
    return sorted(unique)


def _load_documents(files: Iterable[Path]) -> List[Document]:
    documents: List[Document] = []
    for path in files:
        text = normalize_text(read_text_file(path))
        if not text:
            continue
        documents.append(
            Document(
                page_content=text,
                metadata={
                    "source": str(path),
                },
            )
        )
    return documents


def _chunk_documents(documents: Iterable[Document]) -> List[Document]:
    splitter = build_text_splitter(settings.chunk_size_tokens, settings.chunk_overlap_tokens)
    chunks: List[Document] = []
    chunk_id = 0
    for doc_id, doc in enumerate(documents):
        text_chunks = splitter.split_text(doc.page_content)
        for position, chunk_text in enumerate(text_chunks):
            chunk_doc = Document(
                page_content=chunk_text,
                metadata={
                    "source": doc.metadata.get("source", ""),
                    "doc_id": doc_id,
                    "position": position,
                    "chunk_id": chunk_id,
                },
            )
            chunks.append(chunk_doc)
            chunk_id += 1
    return chunks


def build_indexes() -> IngestStats:
    kb_dir = settings.kb_dir
    if not kb_dir.exists():
        raise FileNotFoundError(f"KB_DIR not found: {kb_dir}")

    print(f"Scanning knowledge base: {kb_dir}")
    files = _gather_files(kb_dir)
    documents = _load_documents(files)
    if not documents:
        raise ValueError("No documents found to ingest.")

    print(f"Loaded {len(documents)} documents. Splitting into chunks...")
    chunks = _chunk_documents(documents)
    print(f"Generated {len(chunks)} chunks. Building embeddings with Ollama...")

    settings.storage_dir.mkdir(parents=True, exist_ok=True)
    settings.faiss_dir.mkdir(parents=True, exist_ok=True)

    embeddings = OllamaEmbeddings(
        model=settings.ollama_embed_model,
        base_url=settings.ollama_base_url,
    )
    vector_index = FAISS.from_documents(chunks, embeddings)
    vector_index.save_local(str(settings.faiss_dir))

    records: List[ChunkRecord] = []
    tokenized_docs: List[List[str]] = []
    for chunk in chunks:
        chunk_id = int(chunk.metadata.get("chunk_id", 0))
        record = ChunkRecord(
            chunk_id=chunk_id,
            text=chunk.page_content,
            source=str(chunk.metadata.get("source", "")),
            doc_id=int(chunk.metadata.get("doc_id", 0)),
            position=int(chunk.metadata.get("position", 0)),
        )
        records.append(record)
        tokenized_docs.append(simple_tokenize(chunk.page_content))

    with settings.records_path.open("wb") as handle:
        pickle.dump([asdict(record) for record in records], handle)
    with settings.bm25_tokens_path.open("wb") as handle:
        pickle.dump(tokenized_docs, handle)

    return IngestStats(file_count=len(files), chunk_count=len(chunks))
