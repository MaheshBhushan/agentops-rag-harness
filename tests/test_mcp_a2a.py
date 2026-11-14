import httpx
from mcp import Client

from agentops.agent import _tool_payload, mcp_stdio_params
from agentops.reviewer import ReviewerClient, review


async def test_mcp_client_reaches_retrieval_tools_over_stdio(index_dir):
    async with Client(mcp_stdio_params(index_dir)) as client:
        tools = {t.name for t in (await client.list_tools()).tools}
        assert {"search_chunks", "get_chunk", "list_documents"} <= tools

        hits = _tool_payload(await client.call_tool("search_chunks", {"query": "randomized trial outcome", "k": 3}))
        assert len(hits) == 3 and all({"chunk_id", "text", "score", "modality"} <= h.keys() for h in hits)

        one = _tool_payload(await client.call_tool("get_chunk", {"chunk_id": hits[0]["chunk_id"]}))
        assert one["text"] == hits[0]["text"]

        figs = _tool_payload(await client.call_tool("search_chunks", {"query": "figure", "k": 2, "modality": "figure"}))
        assert figs and all(h["modality"] == "figure" for h in figs)

        docs = _tool_payload(await client.call_tool("list_documents", {}))
        assert len(docs) >= 3


SOURCES = [
    {"chunk_id": "D#structure-1", "text": "Acarbose reduced the salivary amylase activity by 40% compared with placebo."},
    {"chunk_id": "D#structure-2", "text": "Twelve healthy adults completed four crossover trials."},
]


def test_review_logic():
    ok = review("q", "Acarbose reduced salivary amylase activity by 40% [D#structure-1]. Twelve adults completed the trials [D#structure-2].", SOURCES)
    assert ok["approved"] and ok["score"] == 1.0
    uncited = review("q", "Acarbose reduced salivary amylase activity by 40%.", SOURCES)
    assert not uncited["approved"] and "uncited" in uncited["issues"][0]
    unknown = review("q", "Acarbose reduced amylase [D#structure-9].", SOURCES)
    assert not unknown["approved"] and "unknown" in unknown["issues"][0]
    unsupported = review("q", "Metformin doubled insulin sensitivity in elderly patients [D#structure-2].", SOURCES)
    assert not unsupported["approved"]


async def test_reviewer_over_a2a_http(reviewer_url):
    async with httpx.AsyncClient() as http:
        card = (await http.get(f"{reviewer_url}/.well-known/agent-card.json")).json()
        assert card["name"] == "Grounding Reviewer"
        assert card["skills"][0]["id"] == "review_grounding"
    client = ReviewerClient(reviewer_url)
    try:
        verdict = await client.review("q", "Acarbose reduced salivary amylase activity by 40% [D#structure-1].", SOURCES)
        assert verdict["approved"] is True
        verdict = await client.review("q", "Something unrelated entirely here [D#structure-7].", SOURCES)
        assert verdict["approved"] is False
    finally:
        await client.close()
