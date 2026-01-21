# AIreminder RAG CLI

CLI-first RAG agent with hybrid retrieval (FAISS + BM25), reranking, and DeepSeek generation.

## Setup

1. Create a virtual environment.
2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Configure `.env`:

- `DEEPSEEK_API_KEY`
- `DEEPSEEK_API_BASE` (default: https://api.deepseek.com/v1)
- `KB_DIR` (knowledge base directory)

## Usage

Build indexes:

```bash
python -m src.cli ingest
```

Single query:

```bash
python -m src.cli query "RAG agent 提示词"
```

Chat mode:

```bash
python -m src.cli chat
```
