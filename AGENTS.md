# AGENTS.md — Professional Coding Standards

This document establishes coding standards and conventions for the local_rag project. All contributors and agents must adhere to these guidelines.

---

## General Principles

1. **Resiliency first**: Every loop and batch operation must be idempotent and state-resilient. Never assume a run will complete without interruption.
2. **Explicit is better than implicit**: Prefer clear, verbose function signatures and SQL queries over implicit behavior.
3. **No data loss**: All database writes must use explicit transactions. Rows must never be left in `PROCESSING` state on error — always roll back or mark `ERROR`.
4. **Single responsibility**: Each CLI phase (`ingest`, `eval`, `grade`) is a separate entry point. Logic must not leak between phases.

---

## Project Structure

```
local_rag/
├── ADR-0001-architecture.md   # Architecture decision record
├── AGENTS.md                   # This file — coding standards
├── README.md                   # Project overview and package documentation
├── pyproject.toml              # Python dependencies
├── CITATIONS.bib               # BibTeX citation file
├── cli.py                      # Phase 1/2/3 execution CLI (click)
├── src/
│   ├── __init__.py
│   ├── db/
│   │   ├── __init__.py
│   │   ├── connection.py      # SQLite and LadybugDB connection management
│   │   ├── schema.py           # Table creation and migration
│   │   └── queues.py           # Ingestion and evaluation queue operations
│   ├── ingestion/
│   │   ├── __init__.py
│   │   ├── embed.py            # Embedding generation (embeddinggemma-300M)
│   │   ├── vector_store.py     # SQLite + sqlite-vec insertion
│   │   └── graph_extract.py    # LangChain LLMGraphTransformer + LadybugDB insertion
│   ├── evaluation/
│   │   ├── __init__.py
│   │   ├── pipelines.py        # Pipeline A, B, C implementations
│   │   └── tools.py            # LangChain @tool definitions (search_knowledge, get_entity_relationships)
│   ├── grading/
│   │   ├── __init__.py
│   │   └── grader.py           # Phase 3 grader logic using 26B model
│   └── models/
│       ├── __init__.py
│       ├── chat_llamacpp.py    # ChatLlamaCpp model serving wrappers
│       └── config.py           # Model and paths configuration
├── tests/
│   ├── __init__.py
│   ├── test_queues.py
│   ├── test_pipelines.py
│   ├── test_tools.py
│   └── test_grader.py
└── notebooks/
    └── dashboard.py            # marimo dashboard for monitoring SQLite queue progress
```

---

## Database Conventions

### SQLite Connection Management

- All SQLite connections must use context managers (`with sqlite3.connect(...) as conn:`).
- Connections must never be held open across CLI invocations.
- `check_same_thread=False` must be set when sharing connections across threads.

### State Transitions

The only valid `status` transitions are:

```
PENDING → PROCESSING → COMPLETED
                       ERROR
```

Any other transition is a bug. The queue layer must enforce this.

### Transactions

- Every batch job must be wrapped in an explicit `BEGIN` / `COMMIT` or `ROLLBACK`.
- On exception, the row must be marked `ERROR` and the transaction rolled back.
- Never leave a row in `PROCESSING` after an error.

### Queue Queries

- Always use `LIMIT 1` and `ORDER BY chunk_id` (or `eval_id`) to ensure deterministic processing.
- Always re-query `PENDING` on startup — do not cache queue state.

---

## CLI Conventions

### Phase Entrypoints

- Phase 1 — Ingestion: `python cli.py --phase ingest`
- Phase 2 — Evaluation: `python cli.py --phase eval`
- Phase 3 — Grading: `python cli.py --phase grade`

### Argument Conventions

- `--phase`: Required. One of `ingest`, `eval`, `grade`.
- `--batch-size`: Optional. Default `1`. Controls how many rows to process per loop iteration.
- `--model-path`: Optional. Override path to model weights.
- `--dry-run`: Optional. If set, log what would be done without writing to the database.

### Idempotency

- Each phase must be safely re-runnable at any time.
- Re-running a phase that has already completed rows must not re-process those rows.

---

## LangChain Tool Conventions

### Tool Signatures

All LangChain tools must:

1. Use the `@tool` decorator from `langchain_core.tools`.
2. Have a docstring that describes what the tool searches and what format the output takes.
3. Return a formatted string — never a raw dataframe or dictionary.
4. Accept only JSON-serializable inputs (str, int, float, bool).

### Output Format

Tool outputs must be formatted as citation strings:

```
[Source: {source_id}] {text}
```

Each result on its own line. This enables downstream grading to trace citations back to the source.

### Tool Binding

Tools are bound to the 2B agent model during Phase 2 initialization using LangChain's tool binding API. Do not hard-code tool bindings in pipeline logic — inject them at agent construction time.

---

## Model Serving Conventions

### ChatLlamaCpp

- Model loading must happen once per CLI invocation, not per query.
- Wrap model initialization in a lazy singleton or factory to avoid repeated loading.
- Set `n_ctx` (context window) explicitly to match the model's supported length.

### Embedding Model

- The embedding model is initialized once and reused across Phase 1 batches.
- `embed_query` is used for single-query embedding; `embed_documents` for bulk ingestion.

---

## Testing Conventions

- Every module must have a corresponding test file in `tests/`.
- Tests must not require a GPU to run. Mock model and database interactions.
- Queue tests must verify state transitions are enforced and invalid transitions raise exceptions.
- Tool tests must verify output format matches the citation string specification.

---

## Naming Conventions

| Object | Convention | Example |
|---|---|---|
| Table names | snake_case, plural | `ingestion_queue` |
| Column names | snake_case | `pipeline_a_status` |
| Enum values | UPPER_SNAKE | `PENDING`, `COMPLETED` |
| CLI arguments | kebab-case | `--batch-size` |
| Python files | snake_case | `vector_store.py` |
| Class names | PascalCase | `IngestionQueue` |
| Function names | snake_case | `execute_fts` |
| Tool function names | snake_case | `search_knowledge` |

---

## Error Handling

- Never swallow exceptions silently. Log at minimum: row identifier, phase, error message, and timestamp.
- Distinguish between retriable errors (network timeout, OOM) and fatal errors (schema mismatch, invalid input). Only mark rows `ERROR` for fatal errors.
- For retriable errors, leave the row in `PROCESSING` and re-query it on the next loop iteration.

---

## Dashboard Conventions (marimo)

- The dashboard is **read-only**. It must never write to the SQLite database.
- All queries must pull from the queue tables and compute aggregates (e.g., counts per status, per pipeline).
- The dashboard must auto-refresh at a configurable interval (default 5 seconds).
- No authentication is required for the dashboard — it is a local-only tool.

---

## Dependencies

All Python dependencies are declared in `pyproject.toml`. Do not add dependencies without updating `pyproject.toml` first and documenting the rationale in the relevant ADR.

Current core dependencies:

- `llama-index` — indexing and retrieval orchestration
- `llama-index-graph-stores-ladybug` — Ladybug graph store integration
- `openai` — API client (for structured output grading)
- `pydantic-ai-slim` — agent tooling
- `real-ladybug` — embedded graph database

Additional dependencies required by the architecture (not yet in pyproject.toml):

- `sqlite-vec` — vector ANN search extension for SQLite
- `click` — CLI framework
- `marimo` — reactive dashboard
- `langchain` — orchestration and tool binding
- `llama-cpp-python` — ChatLlamaCpp serving

---

## Model Configuration

All model settings are centralized in `src/models/config.py` under the `MODELS` dict. Do not hard-code model paths or parameters elsewhere. Each model entry contains:

- `name`: Human-readable name
- `path`: Path to model weights (set to `None` and override via `--model-path` CLI arg or environment variable)
- `n_ctx`: Context window size
- `n_gpu_layers`: Number of layers offloaded to GPU (`-1` for all)
- `n_threads`: CPU threads (default `None` = auto)
- `n_batch`: Batch size for prompt processing

Embedding-specific config (`embeddinggemma-300M`) and vector dimension (`SQLITE_VEC_DIMENSION`) are also in `config.py`.

---

## Version Policy

- Python >= 3.13
- All packages must be compatible with the versions declared in `pyproject.toml`
- Pin upper bounds for all dependencies to avoid silent breaking changes
