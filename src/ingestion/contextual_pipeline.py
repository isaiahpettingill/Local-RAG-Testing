from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass
from typing import Any

from src.ingestion.chunking import chunk_text
from src.ingestion.embed import embed_query
from src.ingestion.graph_extract import extract_graph
from src.ingestion.ladybug_graph import get_entity_relationships as _relationships
from src.ingestion.ladybug_graph import insert_graph_data
from src.ingestion.staging import get_outgoing_links
from src.ingestion.vector_store import (
    execute_fts,
    execute_vector,
    insert_chunk,
    reciprocal_rank_fusion,
)
from src.models.chat_llamacpp import ChatLlamaCpp

log = logging.getLogger(__name__)


@dataclass(slots=True)
class ContextualChunk:
    chunk_index: int
    raw_text: str
    context: str
    contextual_text: str
    source_url: str
    source_title: str


def _run(coro: Any) -> Any:
    return asyncio.run(coro)


def _build_context_prompt(
    document_text: str,
    title: str,
    url: str,
    chunk_text: str,
    chunk_index: int,
    total_chunks: int,
) -> str:
    return (
        "You are generating retrieval context for a document chunk.\n"
        "Write 1-3 short sentences that explain how this chunk fits into the document and what it is about.\n"
        "Mention the document title and any important entities only if they help retrieval.\n"
        "Do not quote the chunk verbatim and do not add facts not supported by the document.\n\n"
        f"Document title: {title}\n"
        f"Document URL: {url}\n"
        f"Chunk position: {chunk_index + 1} of {total_chunks}\n\n"
        f"Full document excerpt:\n{document_text[:6000]}\n\n"
        f"Chunk:\n{chunk_text}\n"
    )


def _fallback_context(title: str, url: str, chunk_index: int, total_chunks: int) -> str:
    parts = [p for p in (title, url) if p]
    if parts:
        base = " - ".join(parts)
    else:
        base = "Document"
    return f"Context for chunk {chunk_index + 1} of {total_chunks} from {base}."


def _generate_chunk_context(
    document_text: str,
    title: str,
    url: str,
    chunk_text: str,
    chunk_index: int,
    total_chunks: int,
) -> str:
    prompt = _build_context_prompt(
        document_text=document_text,
        title=title,
        url=url,
        chunk_text=chunk_text,
        chunk_index=chunk_index,
        total_chunks=total_chunks,
    )
    try:
        model = ChatLlamaCpp("graph_summarizer")
        response = model.completion(prompt).strip()
        return response or _fallback_context(title, url, chunk_index, total_chunks)
    except Exception as exc:
        log.warning("Context generation failed for %s: %s", url or title, exc)
        return _fallback_context(title, url, chunk_index, total_chunks)


def build_contextual_chunks(
    text: str,
    title: str = "",
    url: str = "",
    chunk_size: int = 1000,
    chunk_overlap: int = 200,
) -> list[ContextualChunk]:
    raw_chunks = chunk_text(text, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    total_chunks = len(raw_chunks)
    contextual_chunks: list[ContextualChunk] = []
    for index, raw_chunk in enumerate(raw_chunks):
        context = _generate_chunk_context(
            document_text=text,
            title=title,
            url=url,
            chunk_text=raw_chunk,
            chunk_index=index,
            total_chunks=total_chunks,
        )
        contextual_text = f"{context}\n\n{raw_chunk}".strip()
        contextual_chunks.append(
            ContextualChunk(
                chunk_index=index,
                raw_text=raw_chunk,
                context=context,
                contextual_text=contextual_text,
                source_url=url,
                source_title=title,
            )
        )
    return contextual_chunks


def _format_result(result: Any, index: int) -> str:
    if isinstance(result, dict):
        source = result.get("source_id") or result.get("source") or result.get("id") or result.get("url") or index
        text = (
            result.get("text")
            or result.get("contextual_text")
            or result.get("content")
            or result.get("answer")
            or result.get("raw_text")
            or str(result)
        )
    else:
        source = (
            getattr(result, "source_id", None)
            or getattr(result, "source", None)
            or getattr(result, "id", None)
            or getattr(result, "url", None)
            or index
        )
        text = (
            getattr(result, "text", None)
            or getattr(result, "contextual_text", None)
            or getattr(result, "content", None)
            or getattr(result, "answer", None)
            or getattr(result, "raw_text", None)
            or str(result)
        )
    return f"[Source: {source}] {text}"


def _format_results(results: list[Any]) -> str:
    return "\n".join(
        _format_result(result, index) for index, result in enumerate(results, start=1)
    )


def search_knowledge(keywords: str, limit: int = 5) -> str:
    query_vector = embed_query(keywords)
    fts_results = execute_fts(keywords, limit=limit)
    vector_results = execute_vector(query_vector, limit=limit)
    fused = reciprocal_rank_fusion(fts_results, vector_results, limit=limit)
    return _format_results(fused)


def get_entity_relationships(entity_name: str, limit: int = 5) -> str:
    return _relationships(entity_name, limit)


def ingest_row(row: Any) -> None:
    title = getattr(row, "source_title", "") or ""
    url = getattr(row, "url", "") or ""
    raw_text = getattr(row, "raw_text", "") or ""
    existing_relationships = _relationships(title or url or raw_text[:80], limit=5)
    chunks = build_contextual_chunks(raw_text, title=title, url=url)
    outgoing_links = get_outgoing_links(url) if url else []
    for chunk in chunks:
        vector = embed_query(chunk.contextual_text)
        chunk_id = insert_chunk(
            text=chunk.contextual_text,
            vector=vector,
            url=url,
            source_title=title,
            chunk_index=chunk.chunk_index,
            raw_text=chunk.raw_text,
            context=chunk.context,
        )
        graph = extract_graph(
            chunk.contextual_text,
            title=title,
            page_url=url,
            outgoing_links=outgoing_links,
            chunk_index=chunk.chunk_index,
            existing_relationships=existing_relationships,
        )
        insert_graph_data(
            graph,
            source_url=url,
            source_title=title,
            chunk_id=chunk_id,
        )
