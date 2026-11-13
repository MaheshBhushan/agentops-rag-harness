from __future__ import annotations

import shutil
from pathlib import Path

import pytest
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

from agentops.chunking import chunk_corpus
from agentops.index import VectorIndex
from agentops.ingest import load_corpus
from agentops.serving import reviewer_in_background
from agentops.telemetry import setup_tracing

FIXTURES = Path(__file__).parent / "fixtures" / "corpus"


@pytest.fixture(scope="session")
def span_exporter() -> InMemorySpanExporter:
    exporter = InMemorySpanExporter()
    setup_tracing(exporter)  # the global tracer provider can only be set once per process
    return exporter


@pytest.fixture(scope="session")
def corpus_dir(tmp_path_factory) -> Path:
    dst = tmp_path_factory.mktemp("corpus")
    shutil.copytree(FIXTURES, dst, dirs_exist_ok=True)
    return dst


@pytest.fixture(scope="session")
def docs(corpus_dir):
    return load_corpus(corpus_dir)


@pytest.fixture(scope="session")
def index_dir(docs, tmp_path_factory) -> Path:
    out = tmp_path_factory.mktemp("index") / "structure"
    VectorIndex.build(chunk_corpus(docs, "structure")).save(out)
    return out


@pytest.fixture(scope="session")
def reviewer_url():
    server, url = reviewer_in_background()
    with server:
        yield url
