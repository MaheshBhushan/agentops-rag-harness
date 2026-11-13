"""Retrieval evaluation across chunking strategies and end-to-end agent evaluation."""

from __future__ import annotations

import statistics
import time

from .agent import RagAgent
from .chunking import STRATEGIES, chunk_corpus
from .index import VectorIndex
from .ingest import Document
from .synth import is_relevant


def retrieval_metrics(index: VectorIndex, items: list[dict], k_max: int = 10) -> dict:
    ranks: list[int | None] = []
    per_mod: dict[str, list[int | None]] = {}
    answerable = 0
    for it in items:
        if any(is_relevant(it, c.doc_id, c.text) for c in index.chunks if c.doc_id == it["doc_id"]):
            answerable += 1
        hits = index.search(it["question"], k=k_max)
        rank = next((i + 1 for i, (c, _) in enumerate(hits) if is_relevant(it, c.doc_id, c.text)), None)
        ranks.append(rank)
        per_mod.setdefault(it["modality"], []).append(rank)

    def recall(rs: list[int | None], k: int) -> float:
        return round(sum(1 for r in rs if r is not None and r <= k) / len(rs), 4) if rs else 0.0

    return {
        "questions": len(items),
        "answerable": answerable,
        "recall@1": recall(ranks, 1),
        "recall@5": recall(ranks, 5),
        "recall@10": recall(ranks, 10),
        "mrr@10": round(sum(1 / r for r in ranks if r) / len(ranks), 4) if ranks else 0.0,
        "recall@5_by_modality": {m: recall(rs, 5) for m, rs in sorted(per_mod.items())},
        "questions_by_modality": {m: len(rs) for m, rs in sorted(per_mod.items())},
    }


def compare_strategies(docs: list[Document], items: list[dict], strategies: list[str] | None = None) -> dict:
    out = {}
    for name in strategies or list(STRATEGIES):
        t0 = time.perf_counter()
        chunks = chunk_corpus(docs, name)
        index = VectorIndex.build(chunks)
        build_s = time.perf_counter() - t0
        m = retrieval_metrics(index, items)
        m["chunks"] = len(chunks)
        m["mean_chunk_chars"] = round(statistics.mean(len(c.text) for c in chunks), 1)
        m["index_build_seconds"] = round(build_s, 2)
        out[name] = m
    return out


async def evaluate_agent(agent: RagAgent, items: list[dict]) -> dict:
    rows = []
    for it in items:
        r = await agent.ask(it["question"])
        source_text = {s["chunk_id"]: s for s in r.sources}
        valid = all(c in source_text for c in r.citations)
        hit = any(is_relevant(it, source_text[c]["doc_id"], source_text[c]["text"]) for c in r.citations if c in source_text)
        rows.append(
            {
                "qid": it["qid"],
                "modality": it["modality"],
                "answered": bool(r.citations),
                "citations_valid": valid,
                "cites_evidence": hit,
                "approved": bool(r.verdict["approved"]),
                "revised": r.revised,
                "input_tokens": r.usage["input_tokens"],
                "output_tokens": r.usage["output_tokens"],
            }
        )
    n = len(rows) or 1

    def rate(key: str) -> float:
        return round(sum(1 for x in rows if x[key]) / n, 4)

    answered = [x for x in rows if x["answered"]]
    return {
        "questions": len(rows),
        "answered_rate": rate("answered"),
        "citation_validity": round(sum(x["citations_valid"] for x in answered) / (len(answered) or 1), 4),
        "cites_evidence_rate": rate("cites_evidence"),
        "reviewer_approval_rate": rate("approved"),
        "revised_rate": rate("revised"),
        "mean_input_tokens": round(statistics.mean(x["input_tokens"] for x in rows), 1) if rows else 0,
        "mean_output_tokens": round(statistics.mean(x["output_tokens"] for x in rows), 1) if rows else 0,
        "rows": rows,
    }
