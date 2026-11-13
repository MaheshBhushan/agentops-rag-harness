"""Parse JATS article XML into typed elements: text paragraphs, tables and figures.

Figures are multimodal: the caption is combined with OCR text extracted from the
figure image itself (axis labels, legends, flow-diagram boxes), so content that
only exists in the image becomes retrievable.
"""

from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path

from lxml import etree

WS = re.compile(r"\s+")


@dataclass
class Element:
    doc_id: str
    element_id: str
    modality: str  # "text" | "table" | "figure"
    section: str
    text: str
    image: str | None = None
    meta: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class Document:
    doc_id: str
    title: str
    license: str
    elements: list[Element]


def _text(node) -> str:
    return WS.sub(" ", "".join(node.itertext())).strip() if node is not None else ""


def _strip_floats(node):
    """Copy of ``node`` without nested tables/figures so paragraph text stays clean."""
    clone = etree.fromstring(etree.tostring(node))
    for tag in ("table-wrap", "fig", "xref"):
        for sub in clone.findall(f".//{tag}"):
            if tag == "xref":
                # keep the tail text, drop the reference marker
                parent = sub.getparent()
                prev = sub.getprevious()
                tail = sub.tail or ""
                if prev is not None:
                    prev.tail = (prev.tail or "") + tail
                else:
                    parent.text = (parent.text or "") + tail
                parent.remove(sub)
            else:
                sub.getparent().remove(sub)
    return clone


def table_to_markdown(tw) -> str:
    caption = _text(tw.find("label")) + " " + _text(tw.find("caption"))
    lines = [caption.strip()]
    for tr in tw.iter("tr"):
        cells = [_text(c) for c in tr if c.tag in ("td", "th")]
        if any(cells):
            lines.append("| " + " | ".join(cells) + " |")
    foot = _text(tw.find("table-wrap-foot"))
    if foot:
        lines.append(foot)
    return "\n".join(lines)


def ocr_image(path: Path) -> str:
    """OCR a figure image with Tesseract; results are cached next to the image."""
    cache = path.with_suffix(path.suffix + ".ocr.txt")
    if cache.exists():
        return cache.read_text()
    import pytesseract
    from PIL import Image

    with Image.open(path) as im:
        im = im.convert("L")
        if max(im.size) < 1600:  # upscale small figures; tesseract likes ~300 dpi text
            scale = 1600 / max(im.size)
            im = im.resize((int(im.width * scale), int(im.height * scale)))
        text = pytesseract.image_to_string(im)
    text = WS.sub(" ", text).strip()
    cache.write_text(text)
    return text


def parse_article(art_dir: Path, ocr: bool = True) -> Document:
    meta = json.loads((art_dir / "meta.json").read_text())
    root = etree.parse(str(art_dir / "article.xml")).getroot()
    doc_id = meta["pmcid"]
    title = _text(root.find(".//article-meta//article-title"))
    elements: list[Element] = []

    def add(modality: str, section: str, text: str, **kw) -> None:
        if text:
            elements.append(Element(doc_id, f"{doc_id}/e{len(elements)}", modality, section, text, **kw))

    add("text", "Title", title)
    for p in root.findall(".//article-meta/abstract//p"):
        add("text", "Abstract", _text(_strip_floats(p)))

    body = root.find(".//body")
    for node in body.iter("p", "table-wrap", "fig"):
        # section path from enclosing <sec><title>
        titles = []
        anc = node.getparent()
        while anc is not None and anc is not body:
            if anc.tag == "sec":
                t = _text(anc.find("title"))
                if t:
                    titles.append(t)
            anc = anc.getparent()
        section = " > ".join(reversed(titles)) or "Body"
        inside_float = any(a.tag in ("table-wrap", "fig") for a in node.iterancestors())
        if node.tag == "p" and not inside_float:
            add("text", section, _text(_strip_floats(node)))
        elif node.tag == "table-wrap":
            if node.find(".//table") is not None:
                add("table", section, table_to_markdown(node), meta={"label": _text(node.find("label"))})
        elif node.tag == "fig":
            fid = node.get("id") or ""
            caption = (_text(node.find("label")) + " " + _text(node.find("caption"))).strip()
            image = meta.get("figure_images", {}).get(fid)
            ocr_text = ""
            if image and ocr:
                ocr_text = ocr_image(art_dir / image)
            text = caption + (f"\n[Figure image text] {ocr_text}" if ocr_text else "")
            add(
                "figure",
                section,
                text,
                image=str(art_dir / image) if image else None,
                meta={"label": _text(node.find("label")), "ocr_chars": len(ocr_text)},
            )
    return Document(doc_id, title, meta.get("license", ""), elements)


def load_corpus(raw_dir: Path, ocr: bool = True) -> list[Document]:
    dirs = sorted(p.parent for p in Path(raw_dir).glob("*/meta.json"))
    return [parse_article(d, ocr=ocr) for d in dirs]
