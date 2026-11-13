"""Chunking strategies compared by the retrieval evaluation.

* ``fixed``      - flatten the document and cut every N characters on word
                   boundaries (LangChain ``CharacterTextSplitter``); tables and
                   figures get cut wherever the window lands.
* ``recursive``  - flatten the document and split with LangChain's
                   ``RecursiveCharacterTextSplitter`` (paragraph > line > sentence
                   > word separators) with overlap.
* ``structure``  - element-aware: paragraphs are packed per section, every table
                   and every figure (caption + OCR text) is its own chunk; long
                   tables are split by rows with the caption repeated.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import asdict, dataclass

from langchain_text_splitters import CharacterTextSplitter, RecursiveCharacterTextSplitter

from .ingest import Document, Element

CHUNK_SIZE = 800


@dataclass
class Chunk:
    chunk_id: str
    doc_id: str
    strategy: str
    modality: str  # text | table | figure | mixed
    section: str
    text: str
    element_ids: list[str]
    image: str | None = None

    def to_dict(self) -> dict:
        return asdict(self)


def _flatten(doc: Document) -> tuple[str, list[tuple[int, int, Element]]]:
    parts, spans, pos = [], [], 0
    for el in doc.elements:
        block = el.text
        spans.append((pos, pos + len(block), el))
        parts.append(block)
        pos += len(block) + 2
    return "\n\n".join(parts), spans


def _from_flat(doc: Document, strategy: str, pieces: list[str]) -> list[Chunk]:
    flat, spans = _flatten(doc)
    chunks, cursor = [], 0
    for piece in pieces:
        start = flat.find(piece, max(0, cursor - CHUNK_SIZE))
        if start < 0:
            start = cursor
        end = start + len(piece)
        cursor = start + 1
        els = [el for s, e, el in spans if s < end and e > start]
        mods = {el.modality for el in els}
        modality = mods.pop() if len(mods) == 1 else "mixed"
        image = next((el.image for el in els if el.image), None)
        chunks.append(
            Chunk(
                f"{doc.doc_id}#{strategy}-{len(chunks)}",
                doc.doc_id,
                strategy,
                modality,
                els[0].section if els else "",
                piece,
                [el.element_id for el in els],
                image,
            )
        )
    return chunks


def chunk_fixed(doc: Document, size: int = CHUNK_SIZE) -> list[Chunk]:
    splitter = CharacterTextSplitter(separator=" ", chunk_size=size, chunk_overlap=0)
    return _from_flat(doc, "fixed", splitter.split_text(_flatten(doc)[0]))


def chunk_recursive(doc: Document, size: int = CHUNK_SIZE) -> list[Chunk]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=size, chunk_overlap=size // 8, separators=["\n\n", "\n", ". ", " ", ""]
    )
    return _from_flat(doc, "recursive", splitter.split_text(_flatten(doc)[0]))


def chunk_structure(doc: Document, size: int = CHUNK_SIZE) -> list[Chunk]:
    out: list[Chunk] = []
    splitter = RecursiveCharacterTextSplitter(chunk_size=size, chunk_overlap=size // 8)

    def emit(modality: str, section: str, text: str, els: list[Element]) -> None:
        image = next((el.image for el in els if el.image), None)
        out.append(
            Chunk(
                f"{doc.doc_id}#structure-{len(out)}",
                doc.doc_id,
                "structure",
                modality,
                section,
                text,
                [el.element_id for el in els],
                image,
            )
        )

    buf: list[Element] = []

    def flush() -> None:
        if not buf:
            return
        section = buf[0].section
        text = f"[{section}] " + "\n".join(el.text for el in buf)
        for piece in splitter.split_text(text) if len(text) > size else [text]:
            emit("text", section, piece, list(buf))
        buf.clear()

    for el in doc.elements:
        if el.modality == "text":
            if buf and (buf[0].section != el.section or sum(len(b.text) for b in buf) + len(el.text) > size):
                flush()
            buf.append(el)
            continue
        flush()
        if el.modality == "table" and len(el.text) > size:
            caption, *rows = el.text.split("\n")
            group: list[str] = []
            for row in rows:
                if group and len(caption) + sum(len(r) for r in group) + len(row) > size:
                    emit("table", el.section, "\n".join([caption, *group]), [el])
                    group = []
                group.append(row)
            if group:
                emit("table", el.section, "\n".join([caption, *group]), [el])
        else:
            emit(el.modality, el.section, el.text, [el])
    flush()
    return out


STRATEGIES: dict[str, Callable[[Document], list[Chunk]]] = {
    "fixed": chunk_fixed,
    "recursive": chunk_recursive,
    "structure": chunk_structure,
}


def chunk_corpus(docs: list[Document], strategy: str) -> list[Chunk]:
    fn = STRATEGIES[strategy]
    return [c for d in docs for c in fn(d)]
