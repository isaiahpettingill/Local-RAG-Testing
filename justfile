# Justfile for Local RAG project

# Default recipe - show help
default:
    @just --list

# Phase 1: Ingestion - Knowledgebase ingestion
ingest:
    uv run python cli.py --phase ingest

# Crawl web pages
crawl:
    uv run python cli.py --phase crawl

# Stage crawled pages to ingestion queue
stage:
    uv run python cli.py --phase stage

# Entity extraction
extract:
    uv run python cli.py --phase extract

# Phase 2: Evaluation - Run pipelines A/B/C
eval:
    uv run python cli.py --phase eval

# Phase 3: Grading - Grade answers
grade:
    uv run python cli.py --phase grade

# Run all phases sequentially
all: crawl stage ingest eval grade

# Run the dashboard (marimo notebook)
dashboard:
    uv run python -m marimo run notebooks/dashboard.py

# Edit the dashboard
dashboard-edit:
    uv run python -m marimo edit notebooks/dashboard.py

# Compile the LaTeX report PDF
report:
    cd report_tex && just build

# Install dependencies
install:
    uv sync

# Lint the project
lint:
    uv run ruff check .

# Type check the project
check:
    uv run ty check src/
