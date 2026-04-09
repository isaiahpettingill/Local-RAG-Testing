# Justfile for Local RAG project

# Default recipe - show help
default:
    @just --list

# Phase 1: Ingestion - Knowledgebase ingestion
ingest:
    python cli.py --phase ingest

# Crawl web pages
crawl:
    python cli.py --phase crawl

# Stage crawled pages to ingestion queue
stage:
    python cli.py --phase stage

# Entity extraction
extract:
    python cli.py --phase extract

# Phase 2: Evaluation - Run pipelines A/B/C
eval:
    python cli.py --phase eval

# Phase 3: Grading - Grade answers
grade:
    python cli.py --phase grade

# Run all phases sequentially
all: crawl stage ingest eval grade

# Run the dashboard (marimo notebook)
dashboard:
    marimo run notebooks/dashboard.py

# Edit the dashboard
dashboard-edit:
    marimo edit notebooks/dashboard.py

# Compile the LaTeX report PDF
report:
    cd report_tex && just build

# Install dependencies
install:
    uv sync

# Lint the project
lint:
    ruff check .

# Type check the project
typecheck:
    mypy src/
