# Local RAG

Resilient multi-strategy RAG evaluation system. Evaluates three generative pipelines locally to measure the empirical value of FTS and Graph retrieval against a raw LLM baseline.

## Installed Packages

- `llama-index>=0.14.20`: LLM indexing and retrieval orchestration.
- `llama-index-graph-stores-ladybug>=0.3.0`: Ladybug graph store integration for LlamaIndex.
- `openai>=2.31.0`: OpenAI API client.
- `pydantic-ai-slim>=1.78.0`: lightweight Pydantic-based agent tooling.
- `real-ladybug>=0.15.3`: Ladybug runtime/integration package.
- `sqlite-vec>=0.1.0`: Vector ANN search extension for SQLite.

Note: LanceDB is NOT used. Storage relies exclusively on SQLite with sqlite-vec for vector similarity and FTS5 with BM25 for keyword search.

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

## Chunking Strategy

Documents are chunked using fixed-size chunking with word-boundary breaking:

- **Chunk size**: 1000 characters (default)
- **Overlap**: 200 characters between chunks to maintain context
- **Boundary detection**: Chunks break at the nearest whitespace character within the last 100 characters of the chunk

The chunking logic is in `src/ingestion/chunking.py` and can be called with:
```python
from src.ingestion.chunking import chunk_text
chunks = chunk_text(text, chunk_size=1000, chunk_overlap=200)
```

## Retrieval Strategies

The system implements three retrieval strategies evaluated in Phase 2:

1. **Pipeline A (Baseline)**: Raw LLM - no retrieval, LLM answers from internal knowledge
2. **Pipeline B (FTS)**: Full-text search using SQLite FTS5 with BM25 ranking
3. **Pipeline C (Graph)**: Knowledge graph retrieval via LadybugDB entity relationships

Hybrid retrieval is supported via reciprocal rank fusion (RRF) combining FTS and vector search results.

## Grading Criteria

Phase 3 grades answers using the Gemma 4 26B model on three criteria:

1. **Accurate (faithfulness)**: Does the answer correctly use facts from the provided context?
   - Must NOT contradict any information in the context
   - Must NOT introduce external facts not present in context

2. **Correct Synthesis**: Does the answer address the query comprehensively?
   - Must directly address what was asked
   - Must synthesize relevant information from context

3. **No Hallucination**: Is every claim in the answer grounded in the context?
   - Each factual claim must be verifiable in the provided context
   - No made-up entities, dates, statistics, or statements

Final score is the average of all three criteria (0.0 to 1.0).

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
    │   ├── chunking.py
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