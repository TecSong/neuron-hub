from __future__ import annotations

import argparse
import sys
from typing import List, Tuple

from .config import settings
from .ingest import build_indexes
from .rag import RAGAgent
from .types import RetrievedChunk


def _format_sources(chunks: List[RetrievedChunk]) -> str:
    if not chunks:
        return "No sources."
    lines = ["Sources:"]
    for chunk in chunks:
        score = chunk.rerank_score if chunk.rerank_score is not None else chunk.fused_score
        lines.append(
            f"- {chunk.record.source} [chunk_id={chunk.record.chunk_id}, score={score:.4f}]"
        )
    return "\n".join(lines)


def handle_ingest() -> int:
    stats = build_indexes()
    print(f"Indexed {stats.file_count} files into {stats.chunk_count} chunks.")
    return 0


def handle_query(question: str) -> int:
    agent = RAGAgent()
    response = agent.answer(question)
    print(response.answer)
    print()
    print(_format_sources(response.chunks))
    return 0


def handle_chat() -> int:
    agent = RAGAgent()
    history: List[Tuple[str, str]] = []
    print("Enter your questions. Type 'exit' to quit.")
    while True:
        try:
            question = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if question.lower() in {"exit", "quit"}:
            break
        if not question:
            continue
        response = agent.answer(question, history=history[-settings.history_max_turns :])
        print(response.answer)
        print(_format_sources(response.chunks))
        print()
        history.append((question, response.answer))
    return 0


def _parse_args(argv: List[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="RAG CLI with hybrid retrieval and rerank")
    subparsers = parser.add_subparsers(dest="command", required=True)

    subparsers.add_parser("ingest", help="Build indexes from knowledge base")

    query_parser = subparsers.add_parser("query", help="Ask a single question")
    query_parser.add_argument("question", nargs="?", help="Question to answer")
    query_parser.add_argument("--question", dest="question_opt", help="Question to answer")

    subparsers.add_parser("chat", help="Start interactive chat")

    return parser.parse_args(argv)


def main(argv: List[str] | None = None) -> int:
    try:
        args = _parse_args(argv or sys.argv[1:])

        if args.command == "ingest":
            return handle_ingest()
        if args.command == "query":
            question = args.question_opt or args.question
            if not question:
                raise ValueError("Question is required. Use --question or positional argument.")
            return handle_query(question)
        if args.command == "chat":
            return handle_chat()

        raise ValueError("Unknown command.")
    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
