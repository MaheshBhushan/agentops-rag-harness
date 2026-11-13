"""HTTP service for Cloud Run.

* ``GET  /healthz``  - liveness/readiness.
* ``POST /ask``      - ``{"question": "..."}`` -> cited, reviewed answer.
* ``/reviewer/...``  - the A2A reviewer agent (agent card + JSON-RPC), mounted in
  the same container; the agent calls it over HTTP like any remote A2A agent
  (set ``REVIEWER_URL`` to use a separately deployed reviewer instead).

The service is stateless: the FAISS index is baked into the image at build
time, the MCP retrieval server runs as a stdio subprocess per container, and
spans/token records go to stdout as JSON lines (Cloud Logging picks them up).
"""

from __future__ import annotations

import json
import os
import sys
from contextlib import asynccontextmanager
from pathlib import Path

from opentelemetry.sdk.trace.export import ConsoleSpanExporter
from starlette.applications import Starlette
from starlette.requests import Request
from starlette.responses import JSONResponse
from starlette.routing import Mount, Route

from .agent import RagAgent, mcp_stdio_params
from .llm import get_provider
from .reviewer import ReviewerClient, build_app
from .telemetry import TokenLog, setup_tracing

PORT = int(os.environ.get("PORT", "8080"))
INDEX = Path(os.environ.get("AGENTOPS_INDEX", "artifacts/index/structure"))
SELF_REVIEWER = f"http://127.0.0.1:{PORT}/reviewer/"
REVIEWER_URL = os.environ.get("REVIEWER_URL", SELF_REVIEWER)


def _json_line(span) -> str:
    return span.to_json(indent=None) + os.linesep


@asynccontextmanager
async def lifespan(app: Starlette):
    setup_tracing(ConsoleSpanExporter(out=sys.stdout, formatter=_json_line))
    app.state.tokens = TokenLog()
    agent = RagAgent(mcp_stdio_params(INDEX), ReviewerClient(REVIEWER_URL), get_provider(), app.state.tokens)
    async with agent:
        app.state.agent = agent
        yield


async def healthz(request: Request) -> JSONResponse:
    return JSONResponse({"status": "ok", "index": str(INDEX)})


async def ask(request: Request) -> JSONResponse:
    body = await request.json()
    question = (body.get("question") or "").strip()
    if not question:
        return JSONResponse({"error": "question is required"}, status_code=400)
    result = await request.app.state.agent.ask(question)
    d = result.to_dict()
    d.pop("sources")
    print(json.dumps({"token_log": request.app.state.tokens.records[-1]}), flush=True)
    return JSONResponse(d)


app = Starlette(
    routes=[
        Route("/healthz", healthz),
        Route("/ask", ask, methods=["POST"]),
        Mount("/reviewer", app=build_app(SELF_REVIEWER)),
    ],
    lifespan=lifespan,
)


def main() -> None:
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=PORT)  # noqa: S104 - Cloud Run requires binding all interfaces


if __name__ == "__main__":
    main()
