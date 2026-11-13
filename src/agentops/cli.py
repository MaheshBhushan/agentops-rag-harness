"""Command line entry point: ``agentops <command>``."""

from __future__ import annotations

import argparse
import asyncio
import json
import time
from collections import Counter
from contextlib import nullcontext
from pathlib import Path

from .agent import RagAgent, mcp_stdio_params
from .chunking import STRATEGIES, chunk_corpus
from .index import VectorIndex
from .ingest import load_corpus
from .llm import get_provider
from .reviewer import ReviewerClient
from .serving import reviewer_in_background
from .telemetry import JsonlSpanExporter, TokenLog, setup_tracing


def cmd_build(a) -> None:
    docs = load_corpus(a.raw)
    counts = Counter(e.modality for d in docs for e in d.elements)
    ocr = sum(1 for d in docs for e in d.elements if e.meta.get("ocr_chars"))
    print(f"documents={len(docs)} elements={dict(counts)} figures_with_ocr_text={ocr}")
    for name in a.strategies:
        chunks = chunk_corpus(docs, name)
        VectorIndex.build(chunks).save(a.out / name)
        print(f"{name}: {len(chunks)} chunks -> {a.out / name}")


def cmd_synth(a) -> None:
    from . import synth

    docs = load_corpus(a.raw)
    log = TokenLog(a.tokens)
    items = synth.generate(docs, get_provider(a.provider), token_log=log)
    synth.save(items, a.out)
    print(f"wrote {len(items)} synthetic questions to {a.out} {dict(Counter(i['modality'] for i in items))}")
    print(f"question generation tokens: {log.totals()}")


def cmd_eval_retrieval(a) -> None:
    from . import synth
    from .evaluate import compare_strategies

    docs = load_corpus(a.raw)
    items = synth.load(a.qa)
    res = compare_strategies(docs, items, a.strategies)
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(res, indent=2))
    print("| strategy | chunks | mean chars | recall@1 | recall@5 | recall@10 | MRR@10 | recall@5 text/table/figure |")
    print("|---|---|---|---|---|---|---|---|")
    for name, m in res.items():
        bm = m["recall@5_by_modality"]
        print(
            f"| {name} | {m['chunks']} | {m['mean_chunk_chars']} | {m['recall@1']} | {m['recall@5']} | "
            f"{m['recall@10']} | {m['mrr@10']} | {bm.get('text')}/{bm.get('table')}/{bm.get('figure')} |"
        )


async def _with_agent(a, fn):
    setup_tracing(JsonlSpanExporter(a.traces))
    log = TokenLog(a.tokens)
    ctx, url = (nullcontext(), a.reviewer_url) if a.reviewer_url else reviewer_in_background()
    with ctx:
        async with RagAgent(mcp_stdio_params(a.index), ReviewerClient(url), get_provider(a.provider), log, k=a.k) as agent:
            out = await fn(agent)
    return out, log


def cmd_ask(a) -> None:
    res, log = asyncio.run(_with_agent(a, lambda agent: agent.ask(a.question)))
    d = res.to_dict()
    d.pop("sources")
    print(json.dumps(d, indent=2))


def cmd_eval_agent(a) -> None:
    from . import synth
    from .evaluate import evaluate_agent

    items = synth.load(a.qa)[: a.n] if a.n else synth.load(a.qa)
    t0 = time.perf_counter()
    res, log = asyncio.run(_with_agent(a, lambda agent: evaluate_agent(agent, items)))
    res["wall_seconds"] = round(time.perf_counter() - t0, 1)
    res["token_totals"] = log.totals()
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(res, indent=2))
    print(json.dumps({k: v for k, v in res.items() if k != "rows"}, indent=2))


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(prog="agentops")
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("fetch", help="download CC-BY RCT papers (delegates to agentops.fetch)")
    p.add_argument("--n", type=int, default=30)
    p.add_argument("--out", type=Path, default=Path("data/raw"))
    p.set_defaults(fn=lambda a: __import__("agentops.fetch", fromlist=["main"]).main(["--n", str(a.n), "--out", str(a.out)]))

    p = sub.add_parser("build", help="parse, OCR, chunk and index the corpus")
    p.add_argument("--raw", type=Path, default=Path("data/raw"))
    p.add_argument("--out", type=Path, default=Path("artifacts/index"))
    p.add_argument("--strategies", nargs="+", default=list(STRATEGIES), choices=list(STRATEGIES))
    p.set_defaults(fn=cmd_build)

    p = sub.add_parser("synth", help="generate a synthetic QA test set")
    p.add_argument("--raw", type=Path, default=Path("data/raw"))
    p.add_argument("--out", type=Path, default=Path("artifacts/synthetic_qa.jsonl"))
    p.add_argument("--tokens", type=Path, default=Path("artifacts/tokens.jsonl"))
    p.add_argument("--provider", default=None)
    p.set_defaults(fn=cmd_synth)

    p = sub.add_parser("eval-retrieval", help="compare chunking strategies on the synthetic set")
    p.add_argument("--raw", type=Path, default=Path("data/raw"))
    p.add_argument("--qa", type=Path, default=Path("artifacts/synthetic_qa.jsonl"))
    p.add_argument("--out", type=Path, default=Path("results/retrieval.json"))
    p.add_argument("--strategies", nargs="+", default=list(STRATEGIES), choices=list(STRATEGIES))
    p.set_defaults(fn=cmd_eval_retrieval)

    for name, fn in (("ask", cmd_ask), ("eval-agent", cmd_eval_agent)):
        p = sub.add_parser(name, help="answer one question" if name == "ask" else "run the agent over the QA set")
        if name == "ask":
            p.add_argument("question")
        else:
            p.add_argument("--qa", type=Path, default=Path("artifacts/synthetic_qa.jsonl"))
            p.add_argument("--n", type=int, default=0)
            p.add_argument("--out", type=Path, default=Path("results/agent.json"))
        p.add_argument("--index", type=Path, default=Path("artifacts/index/structure"))
        p.add_argument("--reviewer-url", default=None, help="A2A reviewer URL; default: start one locally")
        p.add_argument("--provider", default=None)
        p.add_argument("--k", type=int, default=5)
        p.add_argument("--traces", type=Path, default=Path("artifacts/traces.jsonl"))
        p.add_argument("--tokens", type=Path, default=Path("artifacts/tokens.jsonl"))
        p.set_defaults(fn=fn)

    a = ap.parse_args(argv)
    a.fn(a)


if __name__ == "__main__":
    main()
