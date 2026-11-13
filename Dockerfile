# Cloud Run image: stateless HTTP service on $PORT with the FAISS index baked in.
FROM python:3.12-slim

RUN apt-get update \
 && apt-get install -y --no-install-recommends tesseract-ocr \
 && rm -rf /var/lib/apt/lists/*
COPY --from=ghcr.io/astral-sh/uv:0.11 /uv /usr/local/bin/uv

ENV UV_COMPILE_BYTECODE=1 UV_LINK_MODE=copy UV_PROJECT_ENVIRONMENT=/opt/venv PATH=/opt/venv/bin:$PATH
WORKDIR /app
COPY pyproject.toml uv.lock README.md ./
RUN uv sync --frozen --no-dev --no-install-project
COPY src ./src
RUN uv sync --frozen --no-dev

# Corpus to index at build time: the committed CC-BY fixtures by default,
# or `--build-arg CORPUS=data/raw` after `agentops fetch`.
ARG CORPUS=tests/fixtures/corpus
COPY ${CORPUS} /app/corpus
RUN agentops build --raw /app/corpus --out /app/artifacts/index --strategies structure \
 && useradd --uid 10001 app && chown -R app /app
USER app

ENV PORT=8080 AGENTOPS_INDEX=/app/artifacts/index/structure AGENTOPS_LLM=offline
EXPOSE 8080
CMD ["python", "-m", "agentops.app"]
