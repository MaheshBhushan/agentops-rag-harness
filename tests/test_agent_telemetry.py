import json

from agentops.agent import RagAgent, mcp_stdio_params
from agentops.llm import OfflineProvider
from agentops.reviewer import ReviewerClient
from agentops.telemetry import TokenLog


async def test_main_flow_exports_spans_and_token_counts(index_dir, reviewer_url, span_exporter, docs, tmp_path):
    span_exporter.clear()
    log = TokenLog(tmp_path / "tokens.jsonl")
    table = next(e for d in docs for e in d.elements if e.modality == "table")
    question = "What does " + table.text.split("\n", 1)[0][:120] + " show?"

    async with RagAgent(mcp_stdio_params(index_dir), ReviewerClient(reviewer_url), OfflineProvider(), log) as agent:
        result = await agent.ask(question)

    # cited, grounded answer
    assert result.citations, result.answer
    assert set(result.citations) <= set(result.retrieved)
    assert result.verdict["approved"]

    spans = {s.name: s for s in span_exporter.get_finished_spans()}
    for name in ("agent.ask", "mcp.call_tool search_chunks", "llm.chat answer", "a2a.review"):
        assert name in spans, sorted(spans)
    root = spans["agent.ask"]
    assert {s.context.trace_id for s in spans.values()} == {root.context.trace_id}
    for child in ("mcp.call_tool search_chunks", "llm.chat answer", "a2a.review"):
        assert spans[child].parent.span_id == root.context.span_id

    llm = spans["llm.chat answer"].attributes
    assert llm["gen_ai.usage.input_tokens"] > 50
    assert llm["gen_ai.usage.output_tokens"] > 0
    assert llm["gen_ai.usage.input_tokens"] == result.usage["input_tokens"]
    assert spans["mcp.call_tool search_chunks"].attributes["mcp.tool.result_count"] == 5

    records = [json.loads(line) for line in (tmp_path / "tokens.jsonl").read_text().splitlines()]
    assert len(records) == 1
    assert records[0]["input_tokens"] == llm["gen_ai.usage.input_tokens"]
    assert records[0]["trace_id"] == f"{root.context.trace_id:032x}"
