"""Versioned prompt templates (LangChain ``ChatPromptTemplate``).

``tests/test_prompt_regression.py`` pins a hash of every template in
``prompts.lock.json``; editing a prompt without re-running the regression suite
and updating the lock fails CI.
"""

from __future__ import annotations

import hashlib
import json

from langchain_core.prompts import ChatPromptTemplate

ANSWER = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You answer questions about clinical trials using only the numbered sources provided. "
            "Write at most three sentences. End every sentence with the id of the source that supports "
            "it in square brackets, for example [PMC123#structure-4]. Never cite an id that is not listed. "
            "If the sources do not contain the answer, reply exactly: Not found in the provided sources.",
        ),
        ("human", "Sources:\n{sources}\n\nQuestion: {question}\nAnswer:"),
    ]
)

QUESTION_GEN = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You write evaluation questions for a retrieval system over clinical-trial papers. "
            "Write one specific question that is answered by the evidence. Mention the trial topic so the "
            "question is unambiguous across papers. Output only the question.",
        ),
        ("human", "Paper title: {title}\nEvidence type: {modality}\nEvidence: {evidence}\nQuestion:"),
    ]
)

TEMPLATES = {"answer": ANSWER, "question_gen": QUESTION_GEN}


def format_sources(chunks: list[dict]) -> str:
    return "\n".join(
        f"[{c['chunk_id']}] ({c['modality']}; {c['section']}) {' '.join(c['text'].split())}" for c in chunks
    )


def template_hash(name: str) -> str:
    tpl = TEMPLATES[name]
    payload = json.dumps([[m.__class__.__name__, m.prompt.template] for m in tpl.messages])
    return hashlib.sha256(payload.encode()).hexdigest()[:16]
