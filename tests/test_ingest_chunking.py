from agentops.chunking import CHUNK_SIZE, STRATEGIES, chunk_corpus
from agentops.index import VectorIndex


def test_ingestion_extracts_all_three_modalities(docs):
    assert len(docs) >= 3
    for doc in docs:
        mods = {e.modality for e in doc.elements}
        assert {"text", "table", "figure"} <= mods, doc.doc_id
        tables = [e for e in doc.elements if e.modality == "table"]
        assert any(line.startswith("| ") for t in tables for line in t.text.split("\n"))


def test_figure_images_are_ocred(docs):
    figures = [e for d in docs for e in d.elements if e.modality == "figure" and e.image]
    assert figures, "fixtures must include figure images"
    with_ocr = [f for f in figures if f.meta["ocr_chars"] > 0]
    assert with_ocr, "tesseract produced no text for any fixture figure"
    assert all("[Figure image text]" in f.text for f in with_ocr)


def test_every_strategy_chunks_the_corpus(docs):
    for name in STRATEGIES:
        chunks = chunk_corpus(docs, name)
        assert chunks and all(c.text.strip() for c in chunks)
        assert len({c.chunk_id for c in chunks}) == len(chunks)
        assert {c.doc_id for c in chunks} == {d.doc_id for d in docs}
        if name == "fixed":
            assert max(len(c.text) for c in chunks) <= CHUNK_SIZE


def test_structure_strategy_keeps_tables_and_figures_whole(docs):
    chunks = chunk_corpus(docs, "structure")
    mods = {c.modality for c in chunks}
    assert {"text", "table", "figure"} <= mods
    fig = next(c for c in chunks if c.modality == "figure" and c.image)
    assert fig.text.startswith("Figure") or "Fig" in fig.text[:20]


def test_index_roundtrip_and_modality_filter(docs, tmp_path):
    chunks = chunk_corpus(docs, "structure")
    VectorIndex.build(chunks).save(tmp_path)
    index = VectorIndex.load(tmp_path)
    table = next(c for c in chunks if c.modality == "table")
    top, score = index.search(table.text, k=1)[0]
    assert top.chunk_id == table.chunk_id and score > 0.9
    hits = index.search("results", k=3, modality="figure")
    assert hits and all(c.modality == "figure" for c, _ in hits)
