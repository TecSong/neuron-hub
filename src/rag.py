from __future__ import annotations

from dataclasses import dataclass
from typing import List, Sequence, Tuple

from langchain_openai import ChatOpenAI

from .config import settings
from .rerank import Reranker
from .retrieval import HybridRetriever, load_resources
from .types import RetrievedChunk


SYSTEM_PROMPT = (
    "You are a helpful assistant. Use the provided context to answer the question. "
    "If the context does not contain the answer, say you do not know. "
    "Answer in the same language as the question."
)


def _format_context(chunks: Sequence[RetrievedChunk]) -> str:
    blocks: List[str] = []
    for chunk in chunks:
        blocks.append(f"[{chunk.record.chunk_id}] {chunk.record.source}\n{chunk.record.text}")
    return "\n\n".join(blocks)


def _format_history(history: Sequence[Tuple[str, str]]) -> str:
    if not history:
        return ""
    lines: List[str] = []
    for question, answer in history:
        lines.append(f"User: {question}")
        lines.append(f"Assistant: {answer}")
    return "\n".join(lines)


@dataclass
class RAGResponse:
    answer: str
    chunks: List[RetrievedChunk]


class RAGAgent:
    def __init__(self) -> None:
        resources = load_resources()
        self.retriever = HybridRetriever(resources)
        self.reranker = Reranker()
        if not settings.deepseek_api_key:
            raise ValueError("DEEPSEEK_API_KEY is missing.")
        self.llm = ChatOpenAI(
            api_key=settings.deepseek_api_key,
            base_url=settings.deepseek_api_base,
            model=settings.deepseek_model,
            temperature=settings.temperature,
            max_tokens=settings.max_tokens,
        )

    def answer(self, question: str, history: Sequence[Tuple[str, str]] | None = None) -> RAGResponse:
        history = history or []
        retrieved = self.retriever.retrieve(question)
        reranked = self.reranker.rerank(question, retrieved)
        context = _format_context(reranked)
        history_text = _format_history(history)

        prompt = (
            f"{SYSTEM_PROMPT}\n\n"
            f"Context:\n{context}\n\n"
            f"Conversation:\n{history_text}\n\n"
            f"Question: {question}\nAnswer:"
        )
        response = self.llm.invoke(prompt)
        return RAGResponse(answer=response.content, chunks=reranked)
