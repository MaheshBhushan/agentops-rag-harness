"""Dense retrieval: LSA embeddings (TF-IDF + truncated SVD) in a FAISS inner-product index.

The embedder implements LangChain's ``Embeddings`` interface. It is fit on the
indexed corpus, needs no GPU or network, and is fully deterministic, which keeps
the CI regression numbers stable.
"""

from __future__ import annotations

import json
import pickle
from pathlib import Path

import faiss
import numpy as np
from langchain_core.embeddings import Embeddings
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer

from .chunking import Chunk


class LSAEmbeddings(Embeddings):
    def __init__(self, dim: int = 256, seed: int = 0) -> None:
        self.dim = dim
        self.seed = seed
        self.vectorizer = TfidfVectorizer(sublinear_tf=True, ngram_range=(1, 2), min_df=1, stop_words="english")
        self.svd: TruncatedSVD | None = None

    def fit(self, texts: list[str]) -> LSAEmbeddings:
        tfidf = self.vectorizer.fit_transform(texts)
        k = max(2, min(self.dim, tfidf.shape[0] - 1, tfidf.shape[1] - 1))
        self.svd = TruncatedSVD(n_components=k, random_state=self.seed).fit(tfidf)
        return self

    def _embed(self, texts: list[str]) -> np.ndarray:
        assert self.svd is not None, "fit() first"
        vecs = self.svd.transform(self.vectorizer.transform(texts)).astype("float32")
        faiss.normalize_L2(vecs)
        return vecs

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return self._embed(texts).tolist()

    def embed_query(self, text: str) -> list[float]:
        return self._embed([text])[0].tolist()


class VectorIndex:
    def __init__(self, chunks: list[Chunk], embedder: LSAEmbeddings, index: faiss.Index) -> None:
        self.chunks = chunks
        self.embedder = embedder
        self.index = index
        self.by_id = {c.chunk_id: c for c in chunks}

    @classmethod
    def build(cls, chunks: list[Chunk], dim: int = 256) -> VectorIndex:
        texts = [c.text for c in chunks]
        emb = LSAEmbeddings(dim=dim).fit(texts)
        vecs = emb._embed(texts)
        index = faiss.IndexFlatIP(vecs.shape[1])
        index.add(vecs)
        return cls(chunks, emb, index)

    def search(self, query: str, k: int = 5, modality: str | None = None) -> list[tuple[Chunk, float]]:
        q = self.embedder._embed([query])
        n = len(self.chunks) if modality else min(k, len(self.chunks))
        scores, ids = self.index.search(q, n)
        hits = []
        for i, s in zip(ids[0], scores[0], strict=True):
            if i < 0:
                continue
            c = self.chunks[i]
            if modality and c.modality != modality:
                continue
            hits.append((c, float(s)))
            if len(hits) == k:
                break
        return hits

    def save(self, path: Path) -> None:
        path.mkdir(parents=True, exist_ok=True)
        faiss.write_index(self.index, str(path / "index.faiss"))
        with (path / "chunks.jsonl").open("w") as f:
            for c in self.chunks:
                f.write(json.dumps(c.to_dict()) + "\n")
        (path / "embedder.pkl").write_bytes(pickle.dumps(self.embedder))

    @classmethod
    def load(cls, path: Path) -> VectorIndex:
        path = Path(path)
        chunks = [Chunk(**json.loads(line)) for line in (path / "chunks.jsonl").read_text().splitlines()]
        embedder = pickle.loads((path / "embedder.pkl").read_bytes())  # noqa: S301 - our own artifact
        return cls(chunks, embedder, faiss.read_index(str(path / "index.faiss")))
