# PKBA - Personal Knowledge Base Agent

> An agentic personal knowledge system designed to think, remember, and evolve with its owner.

PKBA continuously organizes fragmented information into structured, queryable, and actionable knowledge — serving as a **long-term cognitive extension** rather than a passive storage.

## ✨ Overview

PKBA is a CLI-first RAG (Retrieval-Augmented Generation) agent that transforms your personal documents into an intelligent, conversational knowledge base. It combines:

- **Hybrid Retrieval**: FAISS vector search + BM25 lexical matching for comprehensive results
- **Intelligent Reranking**: Context-aware reranking for more relevant answers
- **Natural Conversation**: Powered by DeepSeek for fluid, contextual interactions
- **Local-First**: Embeddings via Ollama, data stays on your machine

## 🏗️ Architecture

```
User Query → Hybrid Retrieval (FAISS + BM25) → Reranking → LLM Generation → Response
                    ↓
           Knowledge Base (Markdown, Text, etc.)
```

**Key Components:**
- **Ingestion Pipeline**: Chunks documents with token-aware splitting
- **Dual Indexing**: Vector embeddings (FAISS) + keyword search (BM25)
- **Reranker**: Cross-encoder model for relevance scoring
- **Chat Interface**: Stateful conversation with context memory

## 🚀 Quick Start

### Prerequisites

- Python 3.9+
- [Ollama](https://ollama.ai/) running locally for embeddings

### Installation

1. **Clone and setup environment**

```bash
git clone https://github.com/yourusername/PKBA.git
cd PKBA
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
```

2. **Install dependencies**

```bash
pip install -r requirements.txt
```

3. **Pull embedding model**

```bash
ollama pull bge-m3
```

4. **Configure environment**

Copy `.env.example` to `.env` and fill in your settings:

```bash
cp .env.example .env
```

Required variables:
- `DEEPSEEK_API_KEY`: Your DeepSeek API key
- `KB_DIR`: Path to your knowledge base directory (e.g., `~/Documents/knowledge`)

### Usage

#### 1. Ingest Your Knowledge Base

Index all documents in your configured `KB_DIR`:

```bash
python -m src.cli ingest
```

This creates FAISS and BM25 indexes in the `storage/` directory.

#### 2. Query Your Knowledge

**Single question mode:**

```bash
python -m src.cli query "How do I configure the system?"
```

**Interactive chat mode:**

```bash
python -m src.cli chat
```

Chat mode maintains conversation history and allows follow-up questions.

## ⚙️ Configuration

Key settings in `.env`:

| Variable | Description | Default |
|----------|-------------|---------|  
| `KB_DIR` | Knowledge base directory | Required |
| `DEEPSEEK_API_KEY` | DeepSeek API key | Required |
| `OLLAMA_EMBED_MODEL` | Embedding model | `bge-m3` |
| `RERANK_MODEL` | Reranker model | `BAAI/bge-reranker-v2-m3` |
| `CHUNK_SIZE_TOKENS` | Document chunk size | `300` |
| `VECTOR_TOP_N` | Vector search results | `8` |
| `BM25_TOP_N` | BM25 search results | `8` |
| `TOP_K` | Final results after reranking | `20` |
| `DEDUP_SIMILARITY_THRESHOLD` | Retrieval dedup similarity threshold | `0.85` |
| `TEMPERATURE` | LLM sampling temperature | `0.2` |

See `.env.example` for all available options.

## 📁 Project Structure

```
PKBA/
├── src/
│   ├── cli.py         # Command-line interface
│   ├── ingest.py      # Document ingestion & indexing
│   ├── retrieval.py   # Hybrid retrieval (FAISS + BM25)
│   ├── rerank.py      # Relevance reranking
│   ├── rag.py         # RAG orchestration
│   ├── config.py      # Configuration management
│   ├── types.py       # Type definitions
│   └── utils.py       # Utility functions
├── storage/           # Generated indexes (git-ignored)
├── requirements.txt   # Python dependencies
├── .env.example       # Environment template
└── README.md
```

## 🔧 Technical Details

**Retrieval Strategy:**
- FAISS with cosine similarity for semantic search
- BM25 for exact keyword matching
- Weighted fusion of both results (configurable via `VECTOR_WEIGHT` and `BM25_WEIGHT`)

**Document Processing:**
- Token-aware chunking with overlap for context preservation
- Supports Markdown, plain text, and other text formats
- Chinese text segmentation via jieba

**Models:**
- Embeddings: `bge-m3` (via Ollama, multilingual)
- Reranker: `bge-reranker-v2-m3` (cross-encoder)
- Generation: DeepSeek Chat

## 🛣️ Roadmap

- [ ] Support for PDF, DOCX, and other file formats
- [ ] Web UI for easier interaction
- [ ] Auto-ingestion with file watching
- [ ] Multi-index support for organizing different knowledge domains
- [ ] Export conversation history
- [ ] Integration with note-taking apps (Obsidian, Notion)

## 🤝 Contributing

Contributions are welcome! Feel free to:

- Report bugs or request features via issues
- Submit pull requests for improvements
- Share your use cases and feedback

## 📄 License

MIT License - feel free to use this project for personal or commercial purposes.

## 🙏 Acknowledgments

- [LangChain](https://github.com/langchain-ai/langchain) for RAG infrastructure
- [FAISS](https://github.com/facebookresearch/faiss) for efficient vector search
- [DeepSeek](https://www.deepseek.com/) for powerful language generation
- [Ollama](https://ollama.ai/) for local model inference
