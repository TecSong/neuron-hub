from __future__ import annotations

import argparse
import sys
from typing import Iterable, List, Tuple

from .config import settings
from .ingest import build_indexes
from .rag import RAGAgent
from .ws_server import serve as serve_ws
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


def _emit_stream(stream: Iterable[str]) -> str:
    parts: List[str] = []
    for token in stream:
        parts.append(token)
        print(token, end="", flush=True)
    print()
    return "".join(parts)


def handle_ingest() -> int:
    stats = build_indexes()
    print(f"Indexed {stats.file_count} files into {stats.chunk_count} chunks.")
    return 0


def handle_query(question: str) -> int:
    agent = RAGAgent()
    stream, chunks = agent.answer_stream(question)
    _emit_stream(stream)
    print()
    print(_format_sources(chunks))
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
        stream, chunks = agent.answer_stream(question, history=history[-settings.history_max_turns :])
        answer = _emit_stream(stream)
        print(_format_sources(chunks))
        print()
        history.append((question, answer))
    return 0


def handle_serve(host: str, port: int) -> int:
    print(f"WebSocket server listening on ws://{host}:{port}")
    serve_ws(host, port)
    return 0


def handle_web(host: str, port: int) -> int:
    from .web_app import run

    print(f"Web app listening on http://{host}:{port}")
    run(host, port)
    return 0


def _parse_args(argv: List[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="RAG CLI with hybrid retrieval and rerank")
    subparsers = parser.add_subparsers(dest="command", required=True)

    subparsers.add_parser("ingest", help="Build indexes from knowledge base")

    query_parser = subparsers.add_parser("query", help="Ask a single question")
    query_parser.add_argument("question", nargs="?", help="Question to answer")
    query_parser.add_argument("--question", dest="question_opt", help="Question to answer")

    subparsers.add_parser("chat", help="Start interactive chat")

    serve_parser = subparsers.add_parser("serve", help="Start WebSocket server")
    serve_parser.add_argument("--host", default="0.0.0.0", help="Bind host")
    serve_parser.add_argument("--port", type=int, default=8000, help="Bind port")

    web_parser = subparsers.add_parser("web", help="Start FastAPI web UI")
    web_parser.add_argument("--host", default="0.0.0.0", help="Bind host")
    web_parser.add_argument("--port", type=int, default=5001, help="Bind port")

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
        if args.command == "serve":
            return handle_serve(args.host, args.port)
        if args.command == "web":
            return handle_web(args.host, args.port)

        raise ValueError("Unknown command.")
    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
