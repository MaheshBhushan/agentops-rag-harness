"""OpenTelemetry tracing and token logging.

Spans follow the OpenTelemetry GenAI semantic conventions for LLM calls
(``gen_ai.usage.input_tokens`` / ``gen_ai.usage.output_tokens``). Spans are
exported to a JSON-lines file (or any exporter passed in, e.g. the in-memory
exporter in tests or OTLP in production), and every LLM call additionally
appends one record to a token log.
"""

from __future__ import annotations

import json
import threading
from collections.abc import Sequence
from pathlib import Path

from opentelemetry import trace
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import ReadableSpan, TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor, SpanExporter, SpanExportResult

from .llm import Completion

SERVICE = "agentops-rag-harness"
tracer = trace.get_tracer("agentops")


class JsonlSpanExporter(SpanExporter):
    def __init__(self, path: Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()

    def export(self, spans: Sequence[ReadableSpan]) -> SpanExportResult:
        with self._lock, self.path.open("a") as f:
            for s in spans:
                f.write(s.to_json(indent=None) + "\n")
        return SpanExportResult.SUCCESS

    def shutdown(self) -> None:
        pass


def setup_tracing(*exporters: SpanExporter) -> TracerProvider:
    provider = TracerProvider(resource=Resource.create({"service.name": SERVICE}))
    for exp in exporters:
        provider.add_span_processor(SimpleSpanProcessor(exp))
    trace.set_tracer_provider(provider)
    return provider


class TokenLog:
    """Append-only JSONL log of every LLM call with its token usage."""

    def __init__(self, path: Path | None = None) -> None:
        self.path = Path(path) if path else None
        self.records: list[dict] = []
        self._lock = threading.Lock()

    def record(self, purpose: str, provider: str, c: Completion) -> dict:
        span = trace.get_current_span().get_span_context()
        rec = {
            "purpose": purpose,
            "provider": provider,
            "model": c.model,
            "input_tokens": c.input_tokens,
            "output_tokens": c.output_tokens,
            "token_source": c.token_source,
            "trace_id": f"{span.trace_id:032x}",
        }
        with self._lock:
            self.records.append(rec)
            if self.path:
                self.path.parent.mkdir(parents=True, exist_ok=True)
                with self.path.open("a") as f:
                    f.write(json.dumps(rec) + "\n")
        return rec

    def totals(self) -> dict:
        return {
            "calls": len(self.records),
            "input_tokens": sum(r["input_tokens"] for r in self.records),
            "output_tokens": sum(r["output_tokens"] for r in self.records),
        }
