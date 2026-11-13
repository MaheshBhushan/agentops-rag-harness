"""Prompt regression suite (runs in CI).

1. Every prompt template is pinned by hash in ``evals/prompts.lock.json``.
2. A synthetic QA set is generated from the fixture corpus with the
   ``question_gen`` prompt.
3. Retrieval for each chunking strategy and the full agent flow (MCP retrieval,
   ``answer`` prompt, A2A review) are scored on that set and compared with
   ``evals/regression_baseline.json``; a metric may not drop more than
   ``TOLERANCE`` below its baseline.

Refresh both files after an intentional change with
``UPDATE_BASELINE=1 uv run pytest tests/test_prompt_regression.py``.
"""

import json
import os
from pathlib import Path

import pytest

from agentops import synth
from agentops.agent import RagAgent, mcp_stdio_params
from agentops.evaluate import compare_strategies, evaluate_agent
from agentops.llm import OfflineProvider
from agentops.prompts import TEMPLATES, template_hash
from agentops.reviewer import ReviewerClient

EVALS = Path(__file__).parents[1] / "evals"
LOCK = EVALS / "prompts.lock.json"
BASELINE = EVALS / "regression_baseline.json"
TOLERANCE = 0.05
UPDATE = os.environ.get("UPDATE_BASELINE") == "1"


def test_prompt_templates_match_lock():
    current = {name: template_hash(name) for name in TEMPLATES}
    if UPDATE:
        LOCK.write_text(json.dumps(current, indent=2) + "\n")
    assert current == json.loads(LOCK.read_text()), "prompt changed: rerun with UPDATE_BASELINE=1 and review the diff"


@pytest.fixture(scope="module")
def synthetic(docs, tmp_path_factory):
    items = synth.generate(docs, OfflineProvider())
    synth.save(items, tmp_path_factory.mktemp("qa") / "synthetic_qa.jsonl")
    return items


@pytest.fixture(scope="module")
def results(docs, synthetic, index_dir, reviewer_url):
    import asyncio

    async def run_agent():
        async with RagAgent(mcp_stdio_params(index_dir), ReviewerClient(reviewer_url), OfflineProvider()) as agent:
            return await evaluate_agent(agent, synthetic)

    agent = asyncio.run(run_agent())
    agent.pop("rows")
    res = {
        "synthetic": {"questions": len(synthetic), "by_modality": _count(synthetic)},
        "retrieval": {
            k: {m: v[m] for m in ("recall@1", "recall@5", "mrr@10")} for k, v in compare_strategies(docs, synthetic).items()
        },
        "agent": {m: agent[m] for m in ("answered_rate", "citation_validity", "cites_evidence_rate", "reviewer_approval_rate")},
    }
    if UPDATE:
        BASELINE.write_text(json.dumps(res, indent=2) + "\n")
    return res


def _count(items):
    out = {}
    for i in items:
        out[i["modality"]] = out.get(i["modality"], 0) + 1
    return dict(sorted(out.items()))


def test_synthetic_generation_covers_every_modality(synthetic, results):
    base = json.loads(BASELINE.read_text())["synthetic"]
    assert results["synthetic"] == base
    assert set(base["by_modality"]) == {"text", "table", "figure"}
    assert all(i["question"].endswith("?") and i["evidence"] for i in synthetic)


def test_retrieval_does_not_regress(results):
    base = json.loads(BASELINE.read_text())["retrieval"]
    for strategy, metrics in base.items():
        for name, value in metrics.items():
            assert results["retrieval"][strategy][name] >= value - TOLERANCE, (strategy, name)


def test_agent_answers_do_not_regress(results):
    base = json.loads(BASELINE.read_text())["agent"]
    assert results["agent"]["citation_validity"] == 1.0
    for name, value in base.items():
        assert results["agent"][name] >= value - TOLERANCE, name
