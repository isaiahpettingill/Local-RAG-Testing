from __future__ import annotations

import asyncio
import os
from typing import Any

from src.models.config import DATA_DIR, LADYBUGDB_DIR

DATASET_NAME = "coppermind"


def _env(name: str, default: str) -> str:
    value = os.environ.get(name)
    return value if value else default


def configure_cognee_environment() -> None:
    os.environ.setdefault("LLM_PROVIDER", _env("COGNEE_LLM_PROVIDER", "custom"))
    os.environ.setdefault("LLM_MODEL", _env("COGNEE_LLM_MODEL", "gpt-4o-mini"))
    os.environ.setdefault(
        "LLM_ENDPOINT",
        _env(
            "COGNEE_LLM_ENDPOINT",
            os.environ.get("OPENAI_BASE_URL", "http://localhost:8080/v1"),
        ),
    )
    os.environ.setdefault(
        "LLM_API_KEY",
        _env("COGNEE_LLM_API_KEY", os.environ.get("OPENAI_API_KEY", "local-no-key")),
    )
    os.environ.setdefault(
        "EMBEDDING_PROVIDER", _env("COGNEE_EMBEDDING_PROVIDER", "custom")
    )
    os.environ.setdefault(
        "EMBEDDING_MODEL", _env("COGNEE_EMBEDDING_MODEL", "text-embedding-3-small")
    )
    os.environ.setdefault(
        "EMBEDDING_ENDPOINT",
        _env(
            "COGNEE_EMBEDDING_ENDPOINT",
            os.environ.get("OPENAI_BASE_URL", "http://localhost:8080/v1"),
        ),
    )
    os.environ.setdefault(
        "EMBEDDING_API_KEY",
        _env(
            "COGNEE_EMBEDDING_API_KEY", os.environ.get("OPENAI_API_KEY", "local-no-key")
        ),
    )
    os.environ.setdefault("GRAPH_DATABASE_PROVIDER", "kuzu")
    os.environ.setdefault("VECTOR_DB_PROVIDER", "lancedb")
    os.environ.setdefault("DATA_ROOT_DIRECTORY", (DATA_DIR / "cognee").as_posix())
    os.environ.setdefault("SYSTEM_ROOT_DIRECTORY", LADYBUGDB_DIR.as_posix())


def _infer_node_sets(title: str, text: str) -> list[str]:
    haystack = f"{title} {text}".lower()
    node_sets = ["coppermind"]
    for needle, label in (
        ("cosmere", "Cosmere"),
        ("cytoverse", "Cytoverse"),
        ("reckoners", "Reckoners"),
        ("mistborn", "Mistborn"),
        ("stormlight", "Stormlight"),
    ):
        if needle in haystack and label not in node_sets:
            node_sets.append(label)
    return node_sets


def _format_result(result: Any, index: int) -> str:
    if isinstance(result, dict):
        source = (
            result.get("source_id") or result.get("source") or result.get("id") or index
        )
        text = (
            result.get("text")
            or result.get("content")
            or result.get("answer")
            or str(result)
        )
    else:
        source = (
            getattr(result, "source_id", None)
            or getattr(result, "source", None)
            or getattr(result, "id", None)
            or index
        )
        text = (
            getattr(result, "text", None)
            or getattr(result, "content", None)
            or getattr(result, "answer", None)
            or str(result)
        )
    return f"[Source: {source}] {text}"


def _format_results(results: list[Any]) -> str:
    return "\n".join(
        _format_result(result, index) for index, result in enumerate(results, start=1)
    )


def _run(coro: Any) -> Any:
    return asyncio.run(coro)


async def _ingest_text(text: str, title: str) -> None:
    configure_cognee_environment()
    try:
        import cognee
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("Cognee is not installed") from exc

    await cognee.add(
        text, dataset_name=DATASET_NAME, node_set=_infer_node_sets(title, text)
    )
    await cognee.cognify(datasets=[DATASET_NAME], incremental_loading=True)


def ingest_row(row: Any) -> None:
    _run(_ingest_text(getattr(row, "raw_text"), getattr(row, "source_title", "") or ""))


async def _search(query: str, limit: int) -> list[Any]:
    configure_cognee_environment()
    try:
        import cognee
        from cognee import SearchType
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("Cognee is not installed") from exc

    return await cognee.search(
        query_text=query,
        query_type=SearchType.CHUNKS,
        datasets=[DATASET_NAME],
        top_k=limit,
    )


async def _relationships(entity_name: str, limit: int) -> list[Any]:
    configure_cognee_environment()
    try:
        import cognee
        from cognee import SearchType
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("Cognee is not installed") from exc

    return await cognee.search(
        query_text=entity_name,
        query_type=SearchType.GRAPH_COMPLETION,
        datasets=[DATASET_NAME],
        top_k=limit,
        node_name=[entity_name],
    )


def search_knowledge(keywords: str, limit: int = 5) -> str:
    return _format_results(_run(_search(keywords, limit)))


def get_entity_relationships(entity_name: str, limit: int = 5) -> str:
    return _format_results(_run(_relationships(entity_name, limit)))
