"""The RAG agent: retrieve over MCP, answer with citations, get reviewed over A2A.

Flow for one question (all steps traced with OpenTelemetry):

1. ``mcp.call_tool search_chunks``  - retrieval through the MCP server.
2. ``llm.chat answer``              - LLM call with the cited-answer prompt; token usage logged.
3. ``a2a.review``                   - reviewer agent verdict over A2A JSON-RPC.
4. If the reviewer rejects sentences, they are removed (or the agent abstains)
   and the revised answer is reviewed again.
"""

from __future__ import annotations

import json
import os
import sys
from contextlib import AsyncExitStack
from dataclasses import asdict, dataclass, field
from pathlib import Path

from mcp import Client, StdioServerParameters

from .llm import NOT_FOUND, Provider
from .prompts import ANSWER, format_sources, template_hash
from .reviewer import CITE, ReviewerClient
from .telemetry import TokenLog, tracer


def mcp_stdio_params(index_dir: Path) -> StdioServerParameters:
    env = {**os.environ, "AGENTOPS_INDEX": str(Path(index_dir).resolve())}
    return StdioServerParameters(command=sys.executable, args=["-m", "agentops.mcp_server"], env=env)


@dataclass
class AgentResult:
    question: str
    answer: str
    citations: list[str]
    retrieved: list[str]
    verdict: dict
    revised: bool
    usage: dict
    sources: list[dict] = field(repr=False, default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)


def _tool_payload(result) -> list | dict:
    if getattr(result, "is_error", False):
        raise RuntimeError(f"MCP tool error: {result.content}")
    structured = getattr(result, "structured_content", None)
    if structured is not None:
        return structured.get("result", structured) if isinstance(structured, dict) else structured
    return json.loads(result.content[0].text)


class RagAgent:
    def __init__(
        self,
        mcp_server: StdioServerParameters | str,
        reviewer: ReviewerClient,
        provider: Provider,
        token_log: TokenLog | None = None,
        k: int = 5,
    ) -> None:
        self.mcp_server = mcp_server
        self.reviewer = reviewer
        self.provider = provider
        self.token_log = token_log or TokenLog()
        self.k = k
        self._stack = AsyncExitStack()
        self.mcp: Client | None = None

    async def __aenter__(self) -> RagAgent:
        self.mcp = await self._stack.enter_async_context(Client(self.mcp_server))
        return self

    async def __aexit__(self, *exc) -> None:
        await self.reviewer.close()
        await self._stack.aclose()

    async def call_tool(self, name: str, args: dict):
        assert self.mcp is not None, "use 'async with RagAgent(...)'"
        with tracer.start_as_current_span(f"mcp.call_tool {name}") as span:
            span.set_attribute("mcp.tool.name", name)
            span.set_attribute("mcp.tool.arguments", json.dumps(args))
            payload = _tool_payload(await self.mcp.call_tool(name, args))
            if isinstance(payload, list):
                span.set_attribute("mcp.tool.result_count", len(payload))
            return payload

    async def ask(self, question: str) -> AgentResult:
        with tracer.start_as_current_span("agent.ask") as root:
            root.set_attribute("agent.question", question)
            sources = await self.call_tool("search_chunks", {"query": question, "k": self.k})
            messages = ANSWER.format_messages(sources=format_sources(sources), question=question)

            with tracer.start_as_current_span("llm.chat answer") as span:
                span.set_attribute("gen_ai.operation.name", "chat")
                span.set_attribute("gen_ai.system", self.provider.name)
                span.set_attribute("agentops.prompt.template", "answer")
                span.set_attribute("agentops.prompt.hash", template_hash("answer"))
                completion = self.provider.complete(messages)
                span.set_attribute("gen_ai.request.model", completion.model)
                span.set_attribute("gen_ai.usage.input_tokens", completion.input_tokens)
                span.set_attribute("gen_ai.usage.output_tokens", completion.output_tokens)
                span.set_attribute("agentops.token_source", completion.token_source)
                self.token_log.record("answer", self.provider.name, completion)
            answer = completion.text

            verdict = await self._review(question, answer, sources)
            revised = False
            if not verdict["approved"]:
                kept = [s["text"] for s in verdict["sentences"] if s["supported"]]
                answer = " ".join(kept) if kept else NOT_FOUND
                verdict = await self._review(question, answer, sources)
                revised = True

            citations = list(dict.fromkeys(c.strip() for c in CITE.findall(answer)))
            root.set_attribute("agent.citations", citations)
            root.set_attribute("agent.approved", verdict["approved"])
            root.set_attribute("agent.revised", revised)
            usage = {
                "input_tokens": completion.input_tokens,
                "output_tokens": completion.output_tokens,
                "token_source": completion.token_source,
            }
            return AgentResult(
                question, answer, citations, [s["chunk_id"] for s in sources], verdict, revised, usage, sources
            )

    async def _review(self, question: str, answer: str, sources: list[dict]) -> dict:
        with tracer.start_as_current_span("a2a.review") as span:
            span.set_attribute("a2a.agent", "Grounding Reviewer")
            span.set_attribute("a2a.url", self.reviewer.base_url)
            verdict = await self.reviewer.review(question, answer, sources)
            span.set_attribute("review.approved", bool(verdict["approved"]))
            span.set_attribute("review.score", float(verdict["score"]))
            return verdict
