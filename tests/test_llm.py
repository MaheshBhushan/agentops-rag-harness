import json

import httpx

from agentops.llm import NOT_FOUND, AnthropicProvider, OfflineProvider, get_provider
from agentops.prompts import ANSWER, QUESTION_GEN, format_sources

CHUNKS = [
    {"chunk_id": "P1#structure-3", "modality": "text", "section": "Results", "text": "Acarbose lowered the glucose AUC by 31% versus placebo."},
    {"chunk_id": "P1#structure-9", "modality": "table", "section": "Results", "text": "Table 2. Adverse events | Flatulence | 4 | 1 |"},
]


def test_offline_answer_cites_sources():
    msgs = ANSWER.format_messages(sources=format_sources(CHUNKS), question="How much did acarbose lower glucose AUC?")
    c = OfflineProvider().complete(msgs)
    assert c.text.endswith("[P1#structure-3]") and "31%" in c.text
    assert c.input_tokens > 50 and c.output_tokens > 5 and c.token_source == "estimated"


def test_offline_abstains_without_overlap():
    msgs = ANSWER.format_messages(sources=format_sources(CHUNKS), question="Which vaccine was tested in toddlers?")
    assert OfflineProvider().complete(msgs).text == NOT_FOUND


def test_offline_question_generation():
    msgs = QUESTION_GEN.format_messages(title="Oral Acarbose Inhibits Salivary Amylase", modality="table", evidence="Table 2. Adverse events | Flatulence | 4 | 1 |")
    q = OfflineProvider().complete(msgs).text
    assert q.endswith("?") and "flatulence" in q and "table" in q


def test_anthropic_provider_request_and_usage(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    seen = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["headers"] = request.headers
        seen["body"] = json.loads(request.content)
        return httpx.Response(
            200,
            json={
                "model": "claude-test",
                "content": [{"type": "text", "text": "Lowered by 31% [P1#structure-3]."}],
                "usage": {"input_tokens": 211, "output_tokens": 12},
            },
        )

    provider = AnthropicProvider(model="claude-test", client=httpx.Client(transport=httpx.MockTransport(handler)))
    msgs = ANSWER.format_messages(sources=format_sources(CHUNKS), question="q?")
    c = provider.complete(msgs, max_tokens=99)
    assert seen["headers"]["x-api-key"] == "test-key"
    assert seen["body"]["system"].startswith("You answer questions")
    assert seen["body"]["messages"][0]["role"] == "user" and seen["body"]["max_tokens"] == 99
    assert (c.input_tokens, c.output_tokens, c.token_source) == (211, 12, "api")


def test_get_provider_defaults_to_offline(monkeypatch):
    monkeypatch.delenv("AGENTOPS_LLM", raising=False)
    assert isinstance(get_provider(), OfflineProvider)
