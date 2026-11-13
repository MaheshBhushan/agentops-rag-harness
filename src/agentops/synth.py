"""Synthetic test generation.

Evidence spans are sampled from parsed *elements* (not chunks), so the test set
is independent of any chunking strategy. For every evidence span the LLM
provider writes a question with the ``question_gen`` prompt. A chunk counts as
relevant for a question when it belongs to the same paper and contains at least
``RELEVANCE_COVERAGE`` of the evidence span's content words.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from .ingest import Document, Element
from .llm import SENT, Provider, content_words
from .prompts import QUESTION_GEN
from .telemetry import TokenLog

RELEVANCE_COVERAGE = 0.6


def _stable_pick(items: list, key: str):
    h = int(hashlib.sha1(key.encode()).hexdigest(), 16)
    return items[h % len(items)]


def evidence_for(el: Element) -> str | None:
    if el.modality == "text":
        sents = [s.strip() for s in SENT.split(el.text) if 60 <= len(s.strip()) <= 320]
        sents = [s for s in sents if len(set(content_words(s))) >= 6]
        if not sents:
            return None
        with_numbers = [s for s in sents if any(ch.isdigit() for ch in s)]
        return _stable_pick(with_numbers or sents, el.element_id)
    if el.modality == "table":
        caption, *rows = el.text.split("\n")
        rows = [r for r in rows if r.startswith("|") and any(ch.isdigit() for ch in r) and len(content_words(r)) >= 2]
        if not rows:
            return None
        return f"{caption} {_stable_pick(rows, el.element_id)}"
    if el.modality == "figure":
        caption = el.text.split("\n[Figure image text]", 1)[0]
        sents = [s.strip() for s in SENT.split(caption) if len(s.strip()) >= 40]
        return sents[0][:320] if sents else None
    return None


def generate(
    docs: list[Document],
    provider: Provider,
    per_doc: dict[str, int] | None = None,
    token_log: TokenLog | None = None,
) -> list[dict]:
    per_doc = per_doc or {"text": 3, "table": 1, "figure": 1}
    items = []
    for doc in docs:
        for modality, n in per_doc.items():
            cands = [e for e in doc.elements if e.modality == modality and e.section not in ("Title",)]
            cands.sort(key=lambda e: hashlib.sha1(e.element_id.encode()).hexdigest())
            taken = 0
            for el in cands:
                if taken >= n:
                    break
                ev = evidence_for(el)
                if not ev:
                    continue
                messages = QUESTION_GEN.format_messages(title=doc.title, modality=modality, evidence=ev)
                c = provider.complete(messages, max_tokens=120)
                if token_log is not None:
                    token_log.record("question_gen", provider.name, c)
                items.append(
                    {
                        "qid": f"{el.element_id}",
                        "doc_id": doc.doc_id,
                        "element_id": el.element_id,
                        "modality": modality,
                        "evidence": ev,
                        "question": c.text.strip(),
                        "generator": provider.name,
                    }
                )
                taken += 1
    return items


def is_relevant(item: dict, chunk_doc_id: str, chunk_text: str) -> bool:
    if chunk_doc_id != item["doc_id"]:
        return False
    ev = set(content_words(item["evidence"]))
    if not ev:
        return False
    return len(ev & set(content_words(chunk_text))) / len(ev) >= RELEVANCE_COVERAGE


def save(items: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(i) + "\n" for i in items))


def load(path: Path) -> list[dict]:
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]
