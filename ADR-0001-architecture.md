# ADR 0001: Resilient Multi-Strategy RAG Evaluation Architecture

## Status

Accepted

## Context

This project evaluates three generative pipelines locally to measure the empirical value of Full-Text Search (FTS) and Graph retrieval against a raw LLM baseline. The system must run entirely on local hardware, split into distinct offline and runtime phases, and support start/stop/resume at any time without data loss.

---

## Decision

### 1. Core Objectives

Evaluate three generative pipelines:

- **Pipeline A (Baseline)**: Prompt → LLM
- **Pipeline B (Hybrid RAG Agent)**: Prompt → LangChain Tool (`search_knowledge`) → SQLite (FTS5 + Vector) → LLM
- **Pipeline C (GraphRAG Agent)**: Prompt → LangChain Tool (`get_relationships`) → LadybugDB → LLM

### 2. Dataset Selection

- **Dataset**: MultiHop-RAG or HotpotQA (JSONL or CSV) — TBD, user to provide
- **Rationale**: Queries require multi-hop reasoning (e.g., "What is the name of the company founded by the CEO of X?"), which stresses the GraphRAG pipeline and contrasts Vector Search against Graph Traversal.
- **Format**: `[document_corpus]` and `[query, ground_truth_answer]`

### 3. Local Tech Stack

| Component | Recommended Tool | Role |
|---|---|---|
| State Tracker / Queue | SQLite | Tracks job status (PENDING, PROCESSING, COMPLETED, ERROR) to enable start/stop resiliency. |
| Execution Engine | Python CLI (click) | Command-line scripts to trigger resilient batch loops. |
| Dashboard / Monitor | marimo | Reactive, read-only dashboard to monitor SQLite queue progress in real-time and visualize metrics. |
| Orchestration | LangChain | Handles chunking, embedding, and binds Python tools natively to local models. |
| Model Serving | ChatLlamaCpp | Hardware-accelerated local inference for both the 2B Agent and 26B Grader. |
| Vector / FTS DB | SQLite + sqlite-vec | Handles Vector ANN and native BM25 search (FTS5). |
| Graph DB | LadybugDB | Embedded columnar property graph. |

### 4. Model Configuration

All models are configured via `src/models/config.py`. The following models are used:

| Model | Role | Notes |
|---|---|---|
| `embeddinggemma-300M` | Phase 1 embedding generation | Text embedding model |
| Gemma 4 E2B 8bit (GGML) | Phase 2 agent | 2B parameter model for pipeline execution |
| Gemma 4 26B | Phase 1 graph extraction + Phase 3 grading | Already stored locally |

### 5. Database Paths

All database files are stored under `data/` in the repo root:

| File | Purpose |
|---|---|
| `data/knowledgebase.db` | SQLite (FTS5 + sqlite-vec) |
| `data/ladybugdb/` | LadybugDB graph store directory |
| `data/eval_queue.db` | SQLite evaluation queue |
| `data/ingestion_queue.db` | SQLite ingestion queue |

### 6. State Preservation & Resiliency — The Job Queue

Two SQLite state tables manage the entire workflow idempotently. Python loops strictly pull PENDING jobs, process them, and update status before committing.

**Table 1: `ingestion_queue`** (overnight Knowledgebase creation)

| Column | Type | Notes |
|---|---|---|
| `chunk_id` | PK | |
| `raw_text` | TEXT | |
| `status` | ENUM | 'PENDING', 'PROCESSING', 'COMPLETED', 'ERROR' |
| `graph_extraction_attempts` | INT | |

**Table 2: `evaluation_queue`** (runtime A/B testing)

| Column | Type | Notes |
|---|---|---|
| `eval_id` | PK | |
| `query` | TEXT | |
| `ground_truth` | TEXT | |
| `pipeline_a_status` | ENUM | 'PENDING', 'COMPLETED' |
| `pipeline_b_status` | ENUM | 'PENDING', 'COMPLETED' |
| `pipeline_c_status` | ENUM | 'PENDING', 'COMPLETED' |
| `pipeline_a_result` | TEXT | |
| `pipeline_b_result` | TEXT | |
| `pipeline_c_result` | TEXT | |
| `grader_status` | ENUM | 'PENDING', 'COMPLETED' |
| `grader_score_a` | REAL | |
| `grader_score_b` | REAL | |
| `grader_score_c` | REAL | |

### 7. Three-Phase Execution Plan

Each phase runs as an idempotent loop via a CLI script. Scripts can be run in the background (tmux, nohup). The marimo dashboard queries SQLite to show live progress.

#### Phase 1: Data Preparation & Offline Ingestion (Overnight Run)

**Goal**: Populate the Vector/BM25 database and construct the LadybugDB graph.

**Execution Loop** (`python cli.py --phase ingest`):

1. `SELECT * FROM ingestion_queue WHERE status = 'PENDING' LIMIT 1`
2. Update row to `PROCESSING`.
3. Generate embedding via `embeddinggemma-300M` → Insert to SQLite.
4. Call Gemma 4 26B with LangChain's `LLMGraphTransformer` to extract Nodes/Edges → Insert to LadybugDB.
5. Update row to `COMPLETED`. Commit transaction.

On restart, the script re-queries `PENDING` and resumes.

#### Phase 2: Agent Execution (The A/B Test)

**Goal**: Use the 2B model to answer queries through all three pipelines.

**Model**: Gemma 4 E2B 8bit GGML equipped with LangChain tools.

**Execution Loop** (`python cli.py --phase eval`):

1. `SELECT * FROM evaluation_queue WHERE pipeline_c_status = 'PENDING' LIMIT 1`
2. Pass the query to the 2B Agent model.
3. Agent uses `@tool` calls to query SQLite (Pipeline B) or LadybugDB (Pipeline C).
4. Agent returns final answer with citations.
5. Save text string and tool usage context to `evaluation_queue`. Update status. Commit.

#### Phase 3: Automated Grading (Overnight Run)

**Goal**: Use Gemma 4 26B to grade the 2B model's answers against ground truth.

**Execution Loop** (`python cli.py --phase grade`):

1. `SELECT * FROM evaluation_queue WHERE grader_status = 'PENDING' AND pipeline_c_status = 'COMPLETED' LIMIT 1`
2. Parse query, ground_truth, generated_answer, and cited_context.
3. Gemma 4 26B uses structured output (JSON) to grade:
   - `accurate`: Boolean — answer matches the ground truth
   - `correct_synthesis`: Boolean — answer correctly synthesizes the retrieved context without contradiction
   - `no_hallucination`: Boolean — answer contains no additional made-up data beyond what is in the context
4. Update DB with final scores and set `grader_status` to `COMPLETED`. Commit.

### 8. LangChain Tools Reference

These tools are bound to the 2B agent model during Phase 2.

#### Tool 1: `search_knowledge`

```python
@tool
def search_knowledge(keywords: str) -> str:
    """Search the knowledge base for general info. Combines Vector + BM25."""
    query_vector = embeddings.embed_query(keywords)
    fts_results = execute_fts(db_conn, keywords)
    vec_results = execute_vector(db_conn, query_vector)
    hybrid_results = reciprocal_rank_fusion(fts_results, vec_results, limit=3)
    return "\n".join([f"[Source: {r['id']}] {r['text']}" for r in hybrid_results])
```

#### Tool 2: `get_entity_relationships`

```python
@tool
def get_entity_relationships(entity_name: str) -> str:
    """Find everything a specific entity is directly related to."""
    query = f"MATCH (a {{name: '{entity_name}'}})-[r]-(b) RETURN a.name, TYPE(r), b.name, r.source_id LIMIT 5"
    df = lb_conn.execute(query).get_as_df()
    return "\n".join([f"[Source: {r['source_id']}] {r['a.name']} {r['TYPE(r)']} {r['b.name']}" for _, r in df.iterrows()])
```

---

## Consequences

- **Positive**: Full offline operation with start/stop/resume capability eliminates data loss from interrupts.
- **Positive**: SQLite-based job queue is simple, portable, and requires no external services.
- **Positive**: Three distinct pipelines enable empirical comparison of retrieval strategies.
- **Positive**: Configurable model paths via `src/models/config.py` allows swapping models without code changes.
- **Negative**: The multi-phase approach requires careful orchestration to ensure Phase 1 completes before Phase 2 begins.
- **Negative**: The 26B grader model requires significant GPU memory.
