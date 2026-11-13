"""Download CC-BY randomized-controlled-trial papers from Europe PMC / PMC Open Access.

Article XML comes from the Europe PMC REST API; figure images come from the
PMC Open Access bucket on AWS (``pmc-oa-opendata``), which serves the same
files as the PMC article pages without the bot wall.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import httpx
from lxml import etree

EPMC = "https://www.ebi.ac.uk/europepmc/webservices/rest"
OA_BUCKET = "https://pmc-oa-opendata.s3.amazonaws.com"
QUERY = (
    'PUB_TYPE:"Randomized Controlled Trial" AND OPEN_ACCESS:y AND LICENSE:"cc by" '
    "AND HAS_FT:y AND IN_EPMC:y"
)
XLINK = "{http://www.w3.org/1999/xlink}href"
IMAGE_EXT = (".jpg", ".jpeg", ".png", ".webp", ".gif")


def search(client: httpx.Client, n: int, query: str = QUERY) -> list[dict]:
    out: list[dict] = []
    cursor = "*"
    while len(out) < n:
        r = client.get(
            f"{EPMC}/search",
            params={
                "query": query,
                "format": "json",
                "pageSize": 100,
                "resultType": "lite",
                "cursorMark": cursor,
                "sort": "P_PDATE_D desc",
            },
        )
        r.raise_for_status()
        data = r.json()
        hits = [h for h in data["resultList"]["result"] if h.get("pmcid")]
        out.extend(hits)
        nxt = data.get("nextCursorMark")
        if not hits or nxt == cursor:
            break
        cursor = nxt
    return out


def _bucket_keys(client: httpx.Client, pmcid: str) -> list[str]:
    r = client.get(OA_BUCKET, params={"list-type": "2", "prefix": f"{pmcid}.", "max-keys": "200"})
    r.raise_for_status()
    return re.findall(r"<Key>([^<]+)</Key>", r.text)


def fetch_article(client: httpx.Client, pmcid: str, out_dir: Path, max_figures: int = 4) -> dict | None:
    r = client.get(f"{EPMC}/{pmcid}/fullTextXML")
    if r.status_code != 200 or b"<body" not in r.content:
        return None
    root = etree.fromstring(r.content)
    lic = root.find(".//license")
    lic_url = (lic.get(XLINK) if lic is not None else None) or ""
    lic_text = " ".join(lic.itertext()) if lic is not None else ""
    if "creativecommons.org/licenses/by/" not in (lic_url + lic_text):
        return None  # keep strictly CC-BY
    tables = root.findall(".//table-wrap/table")
    figs = root.findall(".//fig")
    if not tables or not figs:
        return None  # the harness needs all three modalities per document
    art_dir = out_dir / pmcid
    art_dir.mkdir(parents=True, exist_ok=True)
    (art_dir / "article.xml").write_bytes(r.content)

    keys = _bucket_keys(client, pmcid)
    by_stem: dict[str, str] = {}
    for k in keys:
        name = k.rsplit("/", 1)[-1]
        stem, _, ext = name.rpartition(".")
        if "." + ext.lower() in IMAGE_EXT:
            by_stem.setdefault(stem, k)
    images = {}
    for fig in figs[:max_figures]:
        g = fig.find(".//graphic")
        if g is None:
            continue
        href = g.get(XLINK, "")
        stem = href.rsplit(".", 1)[0] if href.lower().endswith(IMAGE_EXT) else href
        key = by_stem.get(stem)
        if not key:
            continue
        img = client.get(f"{OA_BUCKET}/{key}")
        if img.status_code == 200:
            fname = key.rsplit("/", 1)[-1]
            (art_dir / fname).write_bytes(img.content)
            images[fig.get("id") or stem] = fname
    meta = {"pmcid": pmcid, "license": lic_url or "CC BY", "figure_images": images}
    (art_dir / "meta.json").write_text(json.dumps(meta, indent=2))
    return meta


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--n", type=int, default=30, help="number of articles to keep")
    ap.add_argument("--out", type=Path, default=Path("data/raw"))
    args = ap.parse_args(argv)
    args.out.mkdir(parents=True, exist_ok=True)
    kept = 0
    with httpx.Client(timeout=60, follow_redirects=True, headers={"User-Agent": "agentops-rag-harness/0.1"}) as client:
        for hit in search(client, args.n * 4):
            if kept >= args.n:
                break
            if (args.out / hit["pmcid"] / "meta.json").exists():
                kept += 1
                continue
            try:
                meta = fetch_article(client, hit["pmcid"], args.out)
            except httpx.HTTPError as exc:  # network hiccup on one article: skip it
                print(f"skip {hit['pmcid']}: {exc}")
                continue
            if meta:
                kept += 1
                print(f"[{kept}/{args.n}] {hit['pmcid']} figures={len(meta['figure_images'])}")
    print(f"kept {kept} articles in {args.out}")


if __name__ == "__main__":
    main()
