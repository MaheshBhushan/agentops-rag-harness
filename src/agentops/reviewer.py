"""Grounding reviewer agent served over the A2A protocol (a2a-sdk, JSON-RPC binding).

The RAG agent sends ``{question, answer, sources}`` as an A2A data message. The
reviewer checks every answer sentence: it must cite at least one source id,
every cited id must be one of the sources, and the cited text must contain at
least ``SUPPORT_THRESHOLD`` of the sentence's content words. It replies with a
data message carrying the verdict.

    python -m agentops.reviewer --port 9100
"""

from __future__ import annotations

import argparse
import re

import httpx
from a2a.client import ClientConfig, create_client
from a2a.helpers import get_data_parts, new_data_message
from a2a.server.agent_execution import AgentExecutor, RequestContext
from a2a.server.events import EventQueue
from a2a.server.request_handlers import DefaultRequestHandler
from a2a.server.routes import create_agent_card_routes, create_jsonrpc_routes
from a2a.server.tasks import InMemoryTaskStore
from a2a.types import AgentCapabilities, AgentCard, AgentInterface, AgentSkill, Role, SendMessageRequest
from starlette.applications import Starlette

from .llm import NOT_FOUND, SENT, content_words

CITE = re.compile(r"\[([^\[\]]+?)\]")
SUPPORT_THRESHOLD = 0.6


def review(question: str, answer: str, sources: list[dict]) -> dict:
    texts = {s["chunk_id"]: s["text"] for s in sources}
    if answer.strip() == NOT_FOUND:
        return {"approved": True, "score": 1.0, "sentences": [], "issues": [], "abstained": True}
    sentences, issues = [], []
    for sent in (s for s in SENT.split(answer.strip()) if s.strip()):
        cites = [c.strip() for c in CITE.findall(sent)]
        claim = CITE.sub("", sent)
        words = set(content_words(claim))
        unknown = [c for c in cites if c not in texts]
        cited_words = set(content_words(" ".join(texts[c] for c in cites if c in texts)))
        support = len(words & cited_words) / len(words) if words else 0.0
        ok = bool(cites) and not unknown and support >= SUPPORT_THRESHOLD
        if not cites:
            issues.append(f"uncited sentence: {claim.strip()[:80]}")
        if unknown:
            issues.append(f"citation to unknown source(s): {', '.join(unknown)}")
        if cites and not unknown and support < SUPPORT_THRESHOLD:
            issues.append(f"weak support ({support:.2f}): {claim.strip()[:80]}")
        sentences.append({"text": sent.strip(), "citations": cites, "support": round(support, 3), "supported": ok})
    score = sum(s["supported"] for s in sentences) / len(sentences) if sentences else 0.0
    return {
        "approved": bool(sentences) and all(s["supported"] for s in sentences),
        "score": round(score, 3),
        "sentences": sentences,
        "issues": issues,
        "abstained": False,
    }


class ReviewerExecutor(AgentExecutor):
    async def execute(self, context: RequestContext, event_queue: EventQueue) -> None:
        data = next(iter(get_data_parts(context.message.parts)), None)
        if not isinstance(data, dict) or "answer" not in data:
            verdict = {"approved": False, "score": 0.0, "sentences": [], "issues": ["bad request"], "abstained": False}
        else:
            verdict = review(data.get("question", ""), data["answer"], data.get("sources", []))
        await event_queue.enqueue_event(new_data_message(verdict, context_id=context.context_id))

    async def cancel(self, context: RequestContext, event_queue: EventQueue) -> None:
        raise NotImplementedError("reviews are synchronous and cannot be cancelled")


def agent_card(url: str) -> AgentCard:
    return AgentCard(
        name="Grounding Reviewer",
        description="Checks that every sentence of a RAG answer is cited and supported by the cited chunks.",
        version="0.1.0",
        default_input_modes=["application/json"],
        default_output_modes=["application/json"],
        capabilities=AgentCapabilities(streaming=False),
        supported_interfaces=[AgentInterface(protocol_binding="JSONRPC", url=url, protocol_version="1.0")],
        skills=[
            AgentSkill(
                id="review_grounding",
                name="Review grounding",
                description="Verdict on citation coverage and support of a cited answer.",
                tags=["rag", "grounding", "review"],
                input_modes=["application/json"],
                output_modes=["application/json"],
            )
        ],
    )


def build_app(url: str) -> Starlette:
    card = agent_card(url)
    handler = DefaultRequestHandler(agent_executor=ReviewerExecutor(), task_store=InMemoryTaskStore(), agent_card=card)
    return Starlette(routes=[*create_agent_card_routes(card), *create_jsonrpc_routes(handler, "/")])


class ReviewerClient:
    """A2A client for the reviewer; resolves the agent card from ``base_url``."""

    def __init__(self, base_url: str, httpx_client: httpx.AsyncClient | None = None) -> None:
        self.base_url = base_url.rstrip("/")
        self.httpx_client = httpx_client
        self._client = None

    async def _get(self):
        if self._client is None:
            http = self.httpx_client or httpx.AsyncClient(timeout=30)
            self._client = await create_client(
                self.base_url, client_config=ClientConfig(streaming=False, httpx_client=http)
            )
        return self._client

    async def review(self, question: str, answer: str, sources: list[dict]) -> dict:
        client = await self._get()
        payload = {
            "question": question,
            "answer": answer,
            "sources": [{"chunk_id": s["chunk_id"], "text": s["text"]} for s in sources],
        }
        req = SendMessageRequest(message=new_data_message(payload, role=Role.ROLE_USER))
        async for event in client.send_message(req):
            msg = event.message if event.HasField("message") else None
            if msg is not None:
                for data in get_data_parts(msg.parts):
                    return data
        raise RuntimeError("reviewer returned no verdict")

    async def close(self) -> None:
        if self._client is not None:
            await self._client.close()


def main() -> None:
    import uvicorn

    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=9100)
    args = ap.parse_args()
    uvicorn.run(build_app(f"http://{args.host}:{args.port}"), host=args.host, port=args.port)


if __name__ == "__main__":
    main()
