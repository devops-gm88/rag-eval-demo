"""Two chunkers, so the eval can compare them.

baseline_chunks: fixed 500-token windows with 50-token overlap, no awareness
                 of headings or sentence edges.
structural_chunks: split on headings and paragraph edges, keep the heading path
                 in the chunk text, and never cut a paragraph mid-sentence.

Tokens are estimated as len(text) / 4. The estimator is documented in the README.
"""
from __future__ import annotations

import math
import re

from .config import (
    BASELINE_CHUNK_TOKENS,
    BASELINE_OVERLAP_TOKENS,
    CHARS_PER_TOKEN,
    STRUCTURAL_MAX_TOKENS,
)


def estimate_tokens(text: str) -> int:
    return max(1, math.ceil(len(text) / CHARS_PER_TOKEN))


def _window(text: str, chars: int, overlap: int) -> list[str]:
    if len(text) <= chars:
        return [text]
    windows: list[str] = []
    start = 0
    while start < len(text):
        end = min(start + chars, len(text))
        if end < len(text):
            # Prefer the last whitespace inside the window, but at most 15% back.
            cut = text.rfind(" ", end - int(chars * 0.15), end)
            if cut != -1:
                end = cut
        piece = text[start:end].strip()
        if piece:
            windows.append(piece)
        if end >= len(text):
            break
        start = max(end - overlap, start + 1)
    return windows


def baseline_chunks(doc: dict) -> list[dict]:
    text = re.sub(r"\s+", " ", doc["text"]).strip()
    chars = BASELINE_CHUNK_TOKENS * CHARS_PER_TOKEN
    overlap = BASELINE_OVERLAP_TOKENS * CHARS_PER_TOKEN
    return [
        {
            "source": doc["source"],
            "chunk_index": index,
            "text": piece,
            "meta": {"title": doc["title"], "kind": doc["kind"], "chunker": "fixed"},
        }
        for index, piece in enumerate(_window(text, chars, overlap))
    ]


HEADING_RE = re.compile(r"^(#{1,6})\s+(.*)$")


def _blocks(text: str) -> list[tuple[str, str]]:
    """Return (kind, text) blocks: heading, para or list."""
    blocks: list[tuple[str, str]] = []
    para: list[str] = []
    for line in text.splitlines():
        heading = HEADING_RE.match(line.strip())
        if heading:
            if para:
                blocks.append(("para", " ".join(para).strip()))
                para = []
            blocks.append(("heading", f"{heading.group(1)} {heading.group(2).strip()}"))
        elif line.strip().startswith(("- ", "* ", "|")):
            if para:
                blocks.append(("para", " ".join(para).strip()))
                para = []
            blocks.append(("list", line.strip()))
        elif not line.strip():
            if para:
                blocks.append(("para", " ".join(para).strip()))
                para = []
        else:
            para.append(line.strip())
    if para:
        blocks.append(("para", " ".join(para).strip()))
    return [(kind, body) for kind, body in blocks if body]


def _split_long_block(body: str, max_chars: int) -> list[str]:
    """Split one paragraph on sentence edges, packing up to max_chars."""
    sentences = re.split(r"(?<=[.!?])\s+", body)
    parts: list[str] = []
    buffer = ""
    for sentence in sentences:
        if len(buffer) + len(sentence) + 1 > max_chars and buffer:
            parts.append(buffer.strip())
            buffer = sentence
        else:
            buffer = f"{buffer} {sentence}".strip()
    if buffer:
        parts.append(buffer.strip())
    return parts


def structural_chunks(doc: dict) -> list[dict]:
    """Pack heading-delimited blocks up to the token budget.

    A heading is a preferred split point, not an automatic one. Small sections
    merge into one chunk so the retriever sees full context. Every chunk starts
    with its heading path, so an isolated chunk still says where it came from.
    """
    max_chars = STRUCTURAL_MAX_TOKENS * CHARS_PER_TOKEN
    min_chars = int(max_chars * 0.45)
    title = doc["title"]
    chunks: list[str] = []
    buffer: list[str] = []
    size = 0
    heading_path = ""

    def flush() -> None:
        nonlocal buffer, size
        if buffer:
            chunks.append("\n".join(buffer).strip())
            buffer = []
            size = 0

    def start(path: str) -> None:
        nonlocal size
        prefix = f"{title} > {path}" if path else title
        buffer.append(prefix)
        size += len(prefix)

    for kind, body in _blocks(doc["text"]):
        if kind == "heading":
            new_path = body.lstrip("# ").strip()
            if not buffer:
                heading_path = new_path
                start(heading_path)
                continue
            if size >= min_chars:
                flush()
                heading_path = new_path
                start(heading_path)
                continue
            heading_path = new_path
            buffer.append(body)
            size += len(body) + 1
            continue

        parts = [body] if len(body) <= max_chars else _split_long_block(body, max_chars)
        for part in parts:
            if not buffer:
                start(heading_path)
            if size + len(part) + 1 > max_chars:
                flush()
                start(heading_path)
            buffer.append(part)
            size += len(part) + 1
    flush()

    return [
        {
            "source": doc["source"],
            "chunk_index": index,
            "text": piece,
            "meta": {
                "title": title,
                "kind": doc["kind"],
                "chunker": "structural",
                "heading": title,
            },
        }
        for index, piece in enumerate(chunks)
    ]


def chunk_documents(docs: list[dict], mode: str) -> list[dict]:
    out: list[dict] = []
    for doc in docs:
        out.extend(baseline_chunks(doc) if mode == "fixed" else structural_chunks(doc))
    return out
