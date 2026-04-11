from __future__ import annotations


def iter_chunk_text(
    text: str, chunk_size: int = 1000, chunk_overlap: int = 200
):
    """Yield text chunks of roughly chunk_size with chunk_overlap."""
    if not text:
        return

    if chunk_size <= 0:
        return

    if chunk_overlap >= chunk_size:
        chunk_overlap = max(0, chunk_size - 1)

    if len(text) <= chunk_size:
        yield text
        return

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
            yield chunk

        next_start = end - chunk_overlap
        if next_start <= start:
            start = end
        else:
            start = next_start
        if start >= text_length:
            break


def chunk_text(
    text: str, chunk_size: int = 1000, chunk_overlap: int = 200
) -> list[str]:
    """Split text into chunks of roughly chunk_size with chunk_overlap.

    Uses simple fixed-size chunking with word-boundary breaking.
    Chunks are separated by chunk_overlap characters to maintain context.
    """
    return list(iter_chunk_text(text, chunk_size=chunk_size, chunk_overlap=chunk_overlap))
