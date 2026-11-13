"""LLM provider adapter.

``OfflineProvider`` is a deterministic extractive model used by tests, CI and
the measured runs in the README: it reads the rendered prompt (sources or
evidence) and produces an answer or question from it. Token counts for it are
counted with a regex tokenizer and are labelled ``estimated``.

``AnthropicProvider`` calls the Messages API over HTTP and reports the token
usage returned by the API. It is selected with ``AGENTOPS_LLM=anthropic``.
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass
from typing import Protocol

import httpx
from langchain_core.messages import BaseMessage

TOKEN = re.compile(r"\w+|[^\w\s]")
WORD = re.compile(r"[A-Za-z][A-Za-z0-9\-]{2,}|\d+(?:\.\d+)?%?")
SENT = re.compile(r"(?<=[.!?])\s+(?=[A-Z(\[])")
SOURCE = re.compile(r"^\[(?P<id>[^\]]+)\] \((?P<meta>[^)]*)\) (?P<text>.*)$", re.M)
STOP = frozenset(
    """a an and are as at be been but by can could did do does for from had has have how in into is it its
    of on or that the their there these this those to was were what when where which while who whom why will
    with would than then also after before during between both each more most other such only over under
    reported report study trial trials paper patients participants group groups""".split()
)
NOT_FOUND = "Not found in the provided sources."


def count_tokens(text: str) -> int:
    return len(TOKEN.findall(text))


def content_words(text: str) -> list[str]:
    return [w.lower() for w in WORD.findall(text) if w.lower() not in STOP]


@dataclass
class Completion:
    text: str
    model: str
    input_tokens: int
    output_tokens: int
    token_source: str  # "api" | "estimated"


class Provider(Protocol):
    name: str

    def complete(self, messages: list[BaseMessage], max_tokens: int = 400) -> Completion: ...


class OfflineProvider:
    name = "offline-extractive"

    def complete(self, messages: list[BaseMessage], max_tokens: int = 400) -> Completion:
        human = messages[-1].content
        assert isinstance(human, str)
        if human.startswith("Sources:"):
            text = self._answer(human)
        elif "Evidence:" in human:
            text = self._question(human)
        else:
            raise ValueError("OfflineProvider only understands the templates in agentops.prompts")
        prompt = "\n".join(str(m.content) for m in messages)
        return Completion(text, self.name, count_tokens(prompt), count_tokens(text), "estimated")

    @staticmethod
    def _answer(human: str) -> str:
        question = human.rsplit("Question:", 1)[1].split("\nAnswer:", 1)[0]
        q = set(content_words(question))
        scored = []
        for rank, m in enumerate(SOURCE.finditer(human.split("\n\nQuestion:", 1)[0])):
            for pos, sent in enumerate(SENT.split(m["text"])):
                words = set(content_words(sent))
                overlap = len(q & words)
                if overlap and len(sent) > 20:
                    # prefer overlap, then higher-ranked sources, then earlier sentences
                    scored.append((-overlap, rank, pos, sent.strip(), m["id"]))
        if not scored:
            return NOT_FOUND
        scored.sort()
        best_overlap = -scored[0][0]
        picked = [s for s in scored if -s[0] >= max(1, best_overlap // 2)][:2]
        out = []
        for _, _, _, sent, cid in picked:
            sent = sent.rstrip(".") + "."
            out.append(f"{sent} [{cid}]")
        return " ".join(out)

    @staticmethod
    def _question(human: str) -> str:
        title = human.split("Paper title:", 1)[1].split("\n", 1)[0].strip()
        modality = human.split("Evidence type:", 1)[1].split("\n", 1)[0].strip()
        evidence = human.split("Evidence:", 1)[1].rsplit("\nQuestion:", 1)[0]
        words = content_words(evidence)
        seen: list[str] = []
        for w in sorted(set(words), key=lambda w: (-len(w), words.index(w)))[:8]:
            seen.append(w)
        keywords = sorted(seen, key=words.index)
        topic = " ".join(content_words(title)[:6])
        kind = {"table": "table", "figure": "figure"}.get(modality, "text")
        return f"In the {topic} trial, what does the {kind} report about {' '.join(keywords)}?"


class AnthropicProvider:
    name = "anthropic"

    def __init__(self, model: str | None = None, client: httpx.Client | None = None) -> None:
        self.model = model or os.environ.get("AGENTOPS_MODEL", "claude-haiku-4-5")
        self.client = client or httpx.Client(timeout=60)

    def complete(self, messages: list[BaseMessage], max_tokens: int = 400) -> Completion:
        system = "\n".join(str(m.content) for m in messages if m.type == "system")
        chat = [
            {"role": "user" if m.type == "human" else "assistant", "content": str(m.content)}
            for m in messages
            if m.type != "system"
        ]
        r = self.client.post(
            "https://api.anthropic.com/v1/messages",
            headers={
                "x-api-key": os.environ["ANTHROPIC_API_KEY"],
                "anthropic-version": "2023-06-01",
            },
            json={"model": self.model, "max_tokens": max_tokens, "system": system, "messages": chat},
        )
        r.raise_for_status()
        data = r.json()
        text = "".join(b.get("text", "") for b in data["content"]).strip()
        usage = data["usage"]
        return Completion(text, data.get("model", self.model), usage["input_tokens"], usage["output_tokens"], "api")


def get_provider(name: str | None = None) -> Provider:
    name = name or os.environ.get("AGENTOPS_LLM", "offline")
    if name == "offline":
        return OfflineProvider()
    if name == "anthropic":
        return AnthropicProvider()
    raise ValueError(f"unknown provider {name!r}")
