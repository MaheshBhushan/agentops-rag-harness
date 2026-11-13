"""MCP server exposing the FAISS retrieval index as tools.

Run over stdio (default, used by the agent) or Streamable HTTP:

    AGENTOPS_INDEX=artifacts/index/structure python -m agentops.mcp_server
    AGENTOPS_INDEX=... python -m agentops.mcp_server --http --port 8765
"""

from __future__ import annotations

import argparse
import os
from functools import lru_cache
from pathlib import Path

from mcp.server.mcpserver import MCPServer

from .index import VectorIndex

server = MCPServer(
    "clinical-trial-retrieval",
    instructions="Search chunks of CC-BY clinical-trial papers (text, tables and figure OCR).",
)


@lru_cache(maxsize=1)
def _index() -> VectorIndex:
    return VectorIndex.load(Path(os.environ.get("AGENTOPS_INDEX", "artifacts/index/structure")))


def _public(chunk, score: float | None = None) -> dict:
    d = {
        "chunk_id": chunk.chunk_id,
        "doc_id": chunk.doc_id,
        "modality": chunk.modality,
        "section": chunk.section,
        "text": chunk.text,
        "image": Path(chunk.image).name if chunk.image else None,
    }
    if score is not None:
        d["score"] = round(score, 4)
    return d


@server.tool()
def search_chunks(query: str, k: int = 5, modality: str | None = None) -> list[dict]:
    """Semantic search over the corpus. modality filters to 'text', 'table', 'figure' or 'mixed'."""
    return [_public(c, s) for c, s in _index().search(query, k=max(1, min(k, 20)), modality=modality)]


@server.tool()
def get_chunk(chunk_id: str) -> dict:
    """Fetch one chunk by id (used to resolve citations)."""
    c = _index().by_id.get(chunk_id)
    if c is None:
        raise ValueError(f"unknown chunk id {chunk_id}")
    return _public(c)


@server.tool()
def list_documents() -> list[dict]:
    """List indexed documents with chunk counts per modality."""
    docs: dict[str, dict] = {}
    for c in _index().chunks:
        d = docs.setdefault(c.doc_id, {"doc_id": c.doc_id, "chunks": 0, "modalities": {}})
        d["chunks"] += 1
        d["modalities"][c.modality] = d["modalities"].get(c.modality, 0) + 1
    return list(docs.values())


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--http", action="store_true")
    ap.add_argument("--port", type=int, default=8765)
    args = ap.parse_args()
    if args.http:
        server.run("streamable-http", port=args.port)
    else:
        server.run()


if __name__ == "__main__":
    main()
