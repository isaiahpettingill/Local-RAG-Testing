# Local RAG

Resilient multi-strategy RAG evaluation system. Evaluates three generative pipelines locally to measure the empirical value of FTS and Graph retrieval against a raw LLM baseline.

## Installed Packages

- `lancedb>=0.30.2`: vector database for local retrieval storage.
- `llama-index>=0.14.20`: LLM indexing and retrieval orchestration.
- `llama-index-graph-stores-ladybug>=0.3.0`: Ladybug graph store integration for LlamaIndex.
- `openai>=2.31.0`: OpenAI API client.
- `pydantic-ai-slim>=1.78.0`: lightweight Pydantic-based agent tooling.
- `real-ladybug>=0.15.3`: Ladybug runtime/integration package.

## Models

All models are configured in `src/models/config.py`.

| Model | Role |
|---|---|
| `embeddinggemma-300M` | Phase 1 embedding generation |
| Gemma 4 E2B 8bit GGML | Phase 2 agent (pipelines A, B, C) |
| Gemma 4 26B | Phase 1 graph extraction + Phase 3 grading |

## Data Paths

All database files are stored under `data/` in the repo root.

| Path | Purpose |
|---|---|
| `data/knowledgebase.db` | SQLite FTS5 + sqlite-vec |
| `data/ladybugdb/` | LadybugDB graph store |
| `data/eval_queue.db` | SQLite evaluation queue |
| `data/ingestion_queue.db` | SQLite ingestion queue |

## CLI

```bash
python cli.py --phase ingest   # Phase 1: Knowledgebase ingestion
python cli.py --phase eval      # Phase 2: Run pipelines A/B/C
python cli.py --phase grade     # Phase 3: Grade answers
```

## Project Structure

```
local_rag/
├── ADR-0001-architecture.md
├── AGENTS.md
├── README.md
├── pyproject.toml
├── cli.py
├── CITATIONS.bib
├── data/                        # Auto-created
│   ├── knowledgebase.db
│   ├── ladybugdb/
│   ├── eval_queue.db
│   └── ingestion_queue.db
└── src/
    ├── __init__.py
    ├── db/
    │   ├── __init__.py
    │   ├── connection.py
    │   ├── schema.py
    │   └── queues.py
    ├── ingestion/
    │   ├── __init__.py
    │   ├── embed.py
    │   ├── vector_store.py
    │   └── graph_extract.py
    ├── evaluation/
    │   ├── __init__.py
    │   ├── pipelines.py
    │   └── tools.py
    ├── grading/
    │   ├── __init__.py
    │   └── grader.py
    └── models/
        ├── __init__.py
        ├── chat_llamacpp.py
        └── config.py
```

## Citations

This project builds on the following works:

```
Pettingill, I. (2025). Local RAG: Resilient Multi-Strategy Retrieval-Augmented Generation Evaluation. https://github.com/isaiahjp/local_rag

Anthropic. (2024). Introducing Contextual Retrieval. https://www.anthropic.com/engineering/contextual-retrieval

Google. (2025). Gemini API Documentation. https://ai.google.dev/gemini-api/docs
```

See `CITATIONS.bib` for BibTeX entries.
