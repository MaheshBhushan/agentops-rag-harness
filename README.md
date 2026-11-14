# agentops-rag-harness

A multimodal retrieval-augmented agent over open-access clinical-trial papers,
plus the harness around it: chunking-strategy evaluation, synthetic test
generation, prompt regression tests, OpenTelemetry tracing with token logging,
and a Cloud Run container.

- **Corpus**: CC-BY randomized controlled trial papers from Europe PMC / PMC Open
  Access, fetched by script. Each paper contributes text paragraphs, tables and
  figure images.
- **Retrieval tools as an MCP server** (official MCP Python SDK, `MCPServer`),
  reached by the agent through an MCP client over stdio.
- **Reviewer agent over A2A** (official `a2a-sdk`, JSON-RPC binding): every
  answer is sent to it in the main flow and rejected sentences are removed.
- **Cited answers**: every sentence carries the id of the chunk that supports it.
- **Observability**: OpenTelemetry spans for the agent, MCP tool calls, the LLM
  call (with `gen_ai.usage.*` token attributes) and the A2A review; a JSONL
  token log keyed by trace id.

## Architecture

```
                       +----------------------------+
 question ──► RagAgent │ agent.ask (OTel root span) │
                       +----------------------------+
                          │ 1. MCP client (stdio)
                          ▼
            agentops.mcp_server  ── tools: search_chunks, get_chunk, list_documents
                          │            └─ FAISS IndexFlatIP over LSA embeddings
                          │ 2. answer prompt (LangChain ChatPromptTemplate)
                          ▼
                    LLM provider adapter  ── offline extractive | Anthropic Messages API
                          │ 3. A2A JSON-RPC (a2a-sdk client)
                          ▼
            Grounding Reviewer agent (a2a-sdk server, Starlette)
                          │ verdict per sentence: cited? id known? supported?
                          ▼
           revise (drop unsupported sentences / abstain) ──► cited answer
```

Source layout (`src/agentops/`):

| module | role |
|---|---|
| `fetch.py` | Europe PMC search for CC-BY RCTs that have tables and figures; article XML from Europe PMC REST, figure images from the `pmc-oa-opendata` bucket |
| `ingest.py` | JATS XML to typed elements: paragraphs (with section path), tables rendered as Markdown rows, figures = caption + Tesseract OCR of the image |
| `chunking.py` | three strategies: `fixed` (800-char windows, LangChain `CharacterTextSplitter`), `recursive` (`RecursiveCharacterTextSplitter`, 100-char overlap), `structure` (element-aware: paragraphs packed per section, each table and figure its own chunk, long tables split by rows with the caption repeated) |
| `index.py` | LangChain `Embeddings` implementation (TF-IDF + truncated SVD, 256 dims) and a FAISS inner-product index |
| `mcp_server.py` | MCP server exposing the index |
| `reviewer.py` | A2A reviewer agent (server, agent card, client) |
| `agent.py` | main flow: MCP retrieval, LLM answer, A2A review, revision |
| `llm.py` / `prompts.py` | provider adapter and versioned prompt templates |
| `synth.py` / `evaluate.py` | synthetic QA generation, retrieval and end-to-end metrics |
| `telemetry.py` | tracer provider, JSONL span exporter, token log |
| `app.py` | Cloud Run HTTP service (`/healthz`, `/ask`, A2A reviewer mounted at `/reviewer`) |

## Quickstart

Requires Python 3.12, [uv](https://docs.astral.sh/uv/) and `tesseract-ocr`.

```bash
uv sync
uv run agentops fetch --n 30                 # ~10 MB into data/raw (gitignored)
uv run agentops build                        # parse + OCR + chunk + index all strategies
uv run agentops synth --out results/synthetic_qa.jsonl
uv run agentops eval-retrieval --qa results/synthetic_qa.jsonl
uv run agentops eval-agent --qa results/synthetic_qa.jsonl --out results/agent_structure.json
uv run agentops ask "What adverse events were reported with blood flow restriction training?"
```

`ask` and `eval-agent` start the MCP server as a stdio subprocess and the A2A
reviewer on a free local port (or use `--reviewer-url` for a running one:
`uv run python -m agentops.reviewer --port 9100`). Spans go to
`artifacts/traces.jsonl`, token records to `artifacts/tokens.jsonl`.

Without downloading anything, the commands work against the committed fixture
corpus: add `--raw tests/fixtures/corpus`.

LLM provider: `AGENTOPS_LLM=offline` (default) is a deterministic extractive
model that selects the source sentences with the highest question overlap and
cites them. `AGENTOPS_LLM=anthropic` (with `ANTHROPIC_API_KEY`, optional
`AGENTOPS_MODEL`) calls the Anthropic Messages API and logs the API-reported
token usage.

## Tests and CI

```bash
uv run pytest -v
```

18 tests, all in GitHub Actions (`.github/workflows/ci.yml`):

- ingestion of all three modalities, OCR of fixture figures, all chunkers, index round trip;
- an MCP client listing and calling the retrieval tools over a real stdio server process;
- the reviewer over real A2A HTTP (agent card + JSON-RPC `SendMessage`);
- the main flow with an in-memory OpenTelemetry exporter: asserts the
  `agent.ask`, `mcp.call_tool search_chunks`, `llm.chat answer` and `a2a.review`
  spans share one trace with the agent span as parent, and that
  `gen_ai.usage.input_tokens`/`output_tokens` equal the token-log record;
- prompt regression (`tests/test_prompt_regression.py`): prompt templates are
  pinned by hash in `evals/prompts.lock.json`; a synthetic QA set is generated
  from the fixture corpus; retrieval metrics for each strategy and the agent's
  answer metrics must stay within 0.05 of `evals/regression_baseline.json`, and
  citation validity must be 1.0. Refresh after an intentional change with
  `UPDATE_BASELINE=1 uv run pytest tests/test_prompt_regression.py`.

The CI workflow also runs `agentops synth`, `build`, `eval-retrieval` and
`eval-agent` on the fixture corpus and uploads the JSON results, and a second job
builds the Docker image, runs it with `PORT=9090` and a read-only filesystem,
calls `/ask`, fetches the reviewer agent card and checks that `a2a.review` spans
appear in the container logs.

## Measured results

Run on this machine on 2026-11-14 with the commands in the quickstart: 30 CC-BY
RCT papers (Europe PMC, most recent first at fetch time), offline provider.
Result files are committed in `results/`.

Ingestion: 1,525 text elements, 117 tables, 113 figures (90 of them with OCR
text from the figure image). Index build for all three strategies (parse, OCR,
chunk, embed): 2 min 41 s wall.

Synthetic test set: 148 questions (90 text, 30 table, 28 figure), one evidence
span sampled per element and turned into a question by the offline generator.
A chunk is relevant if it is from the same paper and contains at least 60% of
the evidence span's content words.

| strategy | chunks | mean chars | recall@1 | recall@5 | recall@10 | MRR@10 | recall@5 text / table / figure |
|---|---|---|---|---|---|---|---|
| fixed | 1645 | 786.6 | 0.2095 | 0.4459 | 0.7162 | 0.3279 | 0.500 / 0.300 / 0.429 |
| recursive | 2350 | 558.4 | 0.1554 | 0.4257 | 0.6014 | 0.2707 | 0.489 / 0.200 / 0.464 |
| structure | 2453 | 578.6 | 0.1149 | 0.3851 | 0.6014 | 0.2382 | 0.411 / 0.300 / 0.393 |

End-to-end agent (MCP retrieval k=5, offline answer, A2A review) over the same
148 questions:

| index | citation validity | answer cites a relevant chunk | reviewer approval | mean input tokens (estimated) |
|---|---|---|---|---|
| fixed | 1.0 | 0.3851 | 1.0 | 889.8 |
| recursive | 1.0 | 0.3784 | 1.0 | 643.4 |
| structure | 1.0 | 0.3514 | 1.0 | 640.9 |

The structure run took 18.2 s for 148 questions and logged 94,857 input and
16,640 output tokens (regex-estimated, since the offline provider has no
tokenizer of its own).

Reading the numbers: on this corpus and with this lexical-overlap embedder, the
simple `fixed` splitter retrieves best. Part of that is the relevance rule: its
chunks are about 35% longer, so they are more likely to contain a whole evidence
span. `structure` matches `fixed` on tables but loses on text, because packing
paragraphs per section produces many short chunks that compete with each other.
Reviewer approval is 1.0 because the offline model copies source sentences
verbatim; the reviewer's rejection paths are covered by tests, not by this run.

## Deployment (Cloud Run)

- `Dockerfile`: `python:3.12-slim` + Tesseract, uv-installed dependencies, the
  index built at image build time (fixture corpus by default,
  `--build-arg CORPUS=data/raw` for the fetched corpus), non-root user, listens
  on `$PORT`. State lives only in the image, so instances are interchangeable.
- `deploy/service.yaml`: Knative service spec (1 vCPU, 1 GiB, `/healthz` startup
  and liveness probes, scale to zero).
- `.github/workflows/deploy.yml`: manual workflow that authenticates with
  Workload Identity Federation, pushes to Artifact Registry and runs
  `gcloud run services replace`.

Local check of the Cloud Run contract:

```bash
docker build -t agentops-rag-harness .
docker run --rm -e PORT=18093 -p 18093:18093 --read-only --tmpfs /tmp agentops-rag-harness
curl -s -X POST localhost:18093/ask -H 'content-type: application/json' \
  -d '{"question":"What were the baseline characteristics of the lung ultrasound fluid management groups?"}'
```

Observed on 2026-11-14 with the image built from the fixture corpus: `/healthz`
returned `ok`, `/ask` returned an answer citing `PMC13656653#structure-5` and
`PMC13656653#structure-0` with the reviewer verdict `approved`, the agent card
was served at `/reviewer/.well-known/agent-card.json`, the container logs held
one `agent.ask`, `mcp.call_tool search_chunks`, `llm.chat answer` and
`a2a.review` span plus a token record, and memory use was 246 MiB.

## Limitations

- **Not deployed to Cloud Run.** Billing is disabled on the available GCP
  project, so the deploy workflow has never run. To deploy: enable billing,
  enable the Run and Artifact Registry APIs, create an Artifact Registry repo
  named `agentops`, set up Workload Identity Federation for this repository, add
  `vars.GCP_PROJECT_ID`, `vars.GCP_REGION`, `secrets.GCP_WORKLOAD_IDENTITY_PROVIDER`
  and `secrets.GCP_SERVICE_ACCOUNT`, then run the `deploy-cloud-run` workflow.
- **No hosted LLM in the measured runs.** The Anthropic adapter is implemented
  and unit-tested against a mocked HTTP transport, but the key available when the
  results were produced was rejected by the API, so every number above comes
  from the offline extractive provider. Answer quality numbers are therefore a
  floor for the pipeline, not a measurement of an LLM.
- **Synthetic questions are keyword-style.** The offline generator builds the
  question from the longest content words of the evidence span plus the paper
  topic. Real user questions paraphrase more, so absolute recall is optimistic
  for lexical matching; use the numbers to compare strategies, not as expected
  production recall.
- **Embeddings are LSA (TF-IDF + SVD)**, fit on the indexed corpus. They keep
  CI deterministic and need no model download, but they are weaker than neural
  embeddings.
- **Figures are handled through OCR + caption**, not a vision model; plots with
  little text contribute only their caption.
- Trace context is not propagated into the MCP subprocess or across the A2A
  HTTP call; the MCP SDK and a2a-sdk emit their own spans in separate traces.

## Data and license

Code: MIT (see `LICENSE`). Fixture articles in `tests/fixtures/corpus`
(PMC13656653, PMC13686207, PMC13750261) and the evidence text in
`results/synthetic_qa.jsonl` are from CC-BY 4.0 articles; the license URL is
recorded in each article's `meta.json`.
