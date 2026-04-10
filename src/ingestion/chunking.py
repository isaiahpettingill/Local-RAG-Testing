from __future__ import annotations


def chunk_text(
    text: str, chunk_size: int = 1000, chunk_overlap: int = 200
) -> list[str]:
    """Split text into chunks of roughly chunk_size with chunk_overlap.

    Uses simple fixed-size chunking with word-boundary breaking.
    Chunks are separated by chunk_overlap characters to maintain context.
    """
    if not text:
        return []

    if chunk_size <= 0:
        return []

    if chunk_overlap >= chunk_size:
        chunk_overlap = max(0, chunk_size - 1)

    if len(text) <= chunk_size:
        return [text]

    chunks = []
    start = 0
    text_length = len(text)

    while start < text_length:
        end = min(start + chunk_size, text_length)

        if end < text_length:
            last_space = text.rfind(" ", max(start, end - 100), end)
            if last_space > start:
                end = last_space + 1

        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)

        start = end - chunk_overlap
        if start >= text_length:
            break
        if start < 0:
            start = 0

    return chunks
