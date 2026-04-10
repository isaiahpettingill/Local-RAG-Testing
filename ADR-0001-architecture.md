# ADR 0001: Resilient Multi-Strategy RAG Evaluation Architecture

## Status

Accepted

## Context

This project evaluates three generative pipelines locally to measure the empirical value of Anthropic-style Contextual Retrieval and graph retrieval against a raw LLM baseline. The system must run entirely on local hardware, split into distinct offline and runtime phases, and support start/stop/resume at any time without data loss.

The domain is Brandon Sanderson's published works, which span multiple universes, series, books, characters, worlds, factions, magic systems, artifacts, and story events. The graph layer needs to help retrieval, not merely mirror the corpus. It must be simple enough for local models to populate reliably and deterministic enough that the local model never has to author Cypher directly.

---

## Decision

### 1. Core Objectives

Evaluate three generative pipelines:

- **Pipeline A (Baseline)**: Prompt → LLM
- **Pipeline B (Contextual RAG Agent)**: Prompt → LangChain Tool (`search_knowledge`) → SQLite (contextual chunks + FTS5 + Vector) → LLM
- **Pipeline C (GraphRAG Agent)**: Prompt → LangChain Tool (`get_relationships`) → LadybugDB → LLM

### 2. Dataset Selection

- **Dataset**: Coppermind crawl and staged exports
- **Rationale**: The corpus contains explicit canonical relationships and cross-work references that make retrieval quality measurable.
- **Format**: `[document_corpus]` and `[query, ground_truth_answer]`

### 3. Local Tech Stack

| Component | Recommended Tool | Role |
|---|---|---|
| State Tracker / Queue | SQLite | Tracks job status (`PENDING`, `PROCESSING`, `COMPLETED`, `ERROR`) to enable start/stop resiliency. |
| Execution Engine | Python CLI (`click`) | Command-line scripts to trigger resilient batch loops. |
| Dashboard / Monitor | `marimo` | Reactive, read-only dashboard to monitor SQLite queue progress in real-time and visualize metrics. |
| Orchestration | LangChain | Handles chunking, contextualization, embedding, and binds Python tools natively to local models. |
| Model Serving | `ChatLlamaCpp` | Local inference for the 2B Agent and 26B extraction/grading models. |
| Vector / FTS DB | SQLite + `sqlite-vec` | Stores contextualized chunks and handles vector ANN plus native BM25 search (FTS5). |
| Graph DB | LadybugDB | Embedded property graph for canonical entity and relationship storage. |

### 4. Model Configuration

All models are configured via `src/models/config.py`. The following models are used:

| Model | Role | Notes |
|---|---|---|
| `embeddinggemma-300M` | Phase 1 embedding generation | Text embedding model |
| Gemma 4 E2B 8bit (GGML) | Phase 2 agent | 2B parameter model for pipeline execution |
| Gemma 4 26B | Phase 1 contextualization + graph extraction + Phase 3 grading | Used for structured extraction and grading |

### 5. Database Paths

All database files are stored under `data/` in the repo root:

| File | Purpose |
|---|---|
| `data/knowledgebase.db` | SQLite (FTS5 + `sqlite-vec`) |
| `data/ladybugdb/` | LadybugDB graph store directory |
| `data/eval_queue.db` | SQLite evaluation queue |
| `data/ingestion_queue.db` | SQLite ingestion queue |
| `data/staging.db` | Optional staging data for crawl/extraction workflow |
| `data/crawl.db` | Crawl discovery/visited state |

### 6. Semantic Knowledge Model

The graph should be a **moderately sized, retrieval-oriented ontology**, not an exhaustive wiki parser.

#### 6.1 Design goals

- Keep the number of node types small enough for the model to classify reliably.
- Keep relation types limited enough to query predictably.
- Preserve canonical entities and useful aliases.
- Store evidence with each edge so the graph remains auditable.
- Support deterministic graph writes from structured JSON output.
- Avoid requiring the model to write Cypher or graph traversal code.

#### 6.2 Canonical node types

The graph uses the following primary node types:

- **Universe**: Cosmere, Reckoners, Cytoverse, etc.
- **Series**: Mistborn Era 1, Stormlight Archive, etc.
- **Book**: novels, novellas, short stories
- **World**: Roshar, Scadrial, Nalthis, Sel, etc.
- **Character**: named characters and major aliases
- **Faction**: organizations, orders, groups, governments
- **MagicSystem**: Allomancy, Surgebinding, Awakening, etc.
- **Artifact**: Shardblades, metalminds, Dawnshards, etc.
- **Event**: major plot events, wars, oaths, transformations
- **Concept**: important lore concepts that do not fit a more specific type

This is intentionally a moderate ontology. If a concept can be modeled as a Character, World, Series, Book, Faction, MagicSystem, Artifact, or Event, it should use that type. Everything else falls back to `Concept`.

#### 6.3 Canonical edge types

Keep the edge vocabulary small and deterministic:

- `part_of_universe`
- `part_of_series`
- `appears_in_book`
- `set_on_world`
- `member_of`
- `leads`
- `ally_of`
- `enemy_of`
- `mentor_of`
- `student_of`
- `family_of`
- `uses_magic_system`
- `wields`
- `associated_with_artifact`
- `associated_with_event`
- `originates_from_world`
- `located_in`
- `same_as`
- `alias_of`
- `mentions`
- `related_to`

The extractor may output richer labels in its JSON, but the write layer must normalize them to this set.

#### 6.4 Evidence model

Every stored entity and edge should carry source evidence:

- `source_url`
- `source_title`
- `chunk_id`
- `source_chunk_id` if separate from `chunk_id`
- `support_text`
- `confidence`
- `scope` (`canonical`, `inferred`, `speculative`)

This makes retrieval explainable and helps the system rank stronger facts above weaker ones.

#### 6.5 Alias and normalization policy

Sanderson lore contains variants and aliases. The graph must support canonicalization without over-merging.

- Use `same_as` only when the alias is effectively certain.
- Use `alias_of` for clear naming variants.
- Prefer a canonical display name with a list of aliases.
- Keep universe and series membership attached to canonical nodes.

Examples:

- `Thaidakar` may be linked as an alias of a canonical character node when supported.
- `Nalthis` and `Nalthian` should be normalized carefully: one is a world, the other is an adjective/demonym.
- Titles and honorifics should not become separate nodes unless they are independently useful.

### 7. Extraction Strategy

The local model should output only structured facts, not graph code.

#### 7.1 Extraction output format

The extraction model returns JSON with three top-level lists:

- `nodes`
- `edges`
- `aliases`

Each node contains:

- `name`
- `type`
- `properties`

Each edge contains:

- `source`
- `type`
- `target`
- `properties`

Each property bag may include:

- `source_url`
- `source_title`
- `chunk_id`
- `support_text`
- `confidence`
- `scope`

#### 7.2 Deterministic write path

The graph write layer must compile structured JSON into graph operations deterministically:

1. Normalize node type and edge type to the approved vocabularies.
2. Escape values safely.
3. Generate MERGE/SET statements in code.
4. Apply writes using a single helper that accepts structured entities and relations.
5. Never let the model author Cypher.

This ensures the model only needs to solve the semantic extraction problem, not graph syntax.

#### 7.3 Extraction prompts

Prompts should ask the model to focus on:

- the canonical entity mentioned
- its universe and series membership
- directly asserted relationships
- world / faction / magic system associations
- important events tied to the chunk

Prompts should explicitly discourage:

- exhaustive entity listing
- speculative relation generation
- free-form narrative output
- Cypher or SQL output

### 8. Graph Construction Pipeline

#### 8.1 Ingestion flow

**Execution Loop** (`python cli.py --phase ingest`):

1. Select the next `PENDING` ingestion row.
2. Build contextual chunk text from the raw chunk.
3. Embed the contextualized text and insert it into SQLite.
4. Extract graph facts from the same contextualized chunk.
5. Deterministically write nodes and edges into LadybugDB.
6. Mark the row `COMPLETED`.

#### 8.2 Why contextual text comes first

Contextualized chunks improve both vector/FTS retrieval and graph extraction because they give the model enough nearby context to identify the right character, world, or event without hallucinating the broader setting.

### 9. Query Strategy

#### 9.1 Retrieval from SQLite

`search_knowledge` should retrieve contextualized chunks from the SQLite store using hybrid BM25 + vector ranking.

#### 9.2 Retrieval from LadybugDB

`get_entity_relationships` should return a compact citation string for a canonical entity and its high-value neighbors.

#### 9.3 Graph query patterns

The graph layer should support these query patterns efficiently:

- character-centric neighborhood expansion
- world-centric retrieval bundles
- series-to-book lookup
- universe scoping
- alias resolution
- 2-hop relationship paths such as character → faction → world

### 10. Implementation Constraints

To keep the system feasible locally:

- Limit node and edge vocabularies.
- Prefer deterministic code over model-generated graph syntax.
- Keep graph extraction chunk-local.
- Avoid requiring full-document global reasoning for every row.
- Use confidence and scope to filter uncertain facts.
- Keep all writes idempotent.
- Never leave queue rows in `PROCESSING` after failure.

### 11. Consequences

#### Positive

- The graph is aligned to the domain rather than a generic entity parser.
- Retrieval can answer character/world/series questions more directly.
- Local models only need to emit structured facts.
- Deterministic graph compilation reduces syntax errors and fragile Cypher generation.
- Evidence-backed edges improve traceability and grading.

#### Negative

- The ontology is intentionally incomplete and will not capture every niche relation.
- Some relationships will be inferred rather than directly stated.
- Alias resolution still needs careful tuning.
- The graph will need periodic schema refinement as new query patterns emerge.

#### Tradeoff

This design sacrifices exhaustive knowledge graph coverage in favor of a graph that is easier to build, easier to query, and much more reliable for local RAG.

### 12. Recommended Initial Scope

Start with the following minimum viable graph:

- 4 node families: `Universe`, `Series`, `Book`, `Character`
- 4 supporting families: `World`, `Faction`, `MagicSystem`, `Artifact`
- 3 event/concept families: `Event`, `Concept`, `Alias`
- 10 to 15 relation types total
- deterministic write helpers
- citation-backed edge properties

This is enough to make the graph useful for Sanderson questions without overwhelming the local model.

---

## Consequences

- **Positive**: Full offline operation with start/stop/resume capability eliminates data loss from interrupts.
- **Positive**: SQLite-based job queue is simple, portable, and requires no external services.
- **Positive**: Three distinct pipelines enable empirical comparison of retrieval strategies.
- **Positive**: Configurable model paths via `src/models/config.py` allows swapping models without code changes.
- **Positive**: A constrained semantic ontology makes graph extraction and querying feasible on local hardware.
- **Negative**: The multi-phase approach requires careful orchestration to ensure ingestion completes before evaluation begins.
- **Negative**: The graph model intentionally leaves out some long-tail lore relationships.
- **Negative**: The 26B extraction/grading model requires significant GPU memory.
