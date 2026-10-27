"""Tessera: the evidence delivery layer (paper §3).

Given a pipeline's reranked list for a question, Tessera

1. anchors on the distinct pages of the top `m` items (default 10),
2. rebuilds those pages into structure-completed units (`tessera.units`),
3. keeps the `N` units sharing the most content words with the question
   (default 100) and scores them with the BGE cross-encoder,
4. packs the best units into the token budget of the last of the top-k
   items, printing each context path and header line once per group.

The reader then gets the pipeline's top k-1 items plus the unit block.

Three steps, so scoring can be spread over GPUs:

    python -m tessera.layer pools --candidates C.jsonl --zim W.zim --out run/
    python -m tessera.layer score --out run/ --part 0 --parts 1
    python -m tessera.layer build --candidates C.jsonl --out run/ --tokenizer T

`build` writes `baseline_candidates.jsonl` (top-k items) and
`tessera_candidates.jsonl` (top k-1 items + unit block) for `tessera.read`.
`--builder docling|unstructured` swaps the unit builder for a structure-aware
chunker (`tessera.baselines`), with every other step unchanged.
"""

from __future__ import annotations

import argparse
import json
from multiprocessing import Pool
from pathlib import Path

from tessera.common import (
    bge_scorer,
    content_words,
    read_jsonl,
    reranked,
    score_batched,
    tokens,
    write_jsonl,
)

UNIT_HIT = {"title": "units", "article_id": -1, "chunk_index": -1}


def anchor_pages(hits: list[dict], pages: int = 10, depth: int = 10) -> list[str]:
    """Distinct page titles of the top `depth` reranked hits, in rank order."""
    return list(dict.fromkeys(h["title"] for h in reranked(hits, depth)))[:pages]


def prefilter(question: str, units: list[dict], keep: int = 100) -> list[dict]:
    """The `keep` units sharing the most content words with the question."""
    words = content_words(question)
    return sorted(units, key=lambda u: -len(words & content_words(u["completed"])))[:keep]


def _header(unit: dict) -> str:
    if unit["labels"]:
        return f"{unit['group']}\n" + " | ".join(unit["labels"])
    return f"{unit['group']}:"


def _line(unit: dict) -> str:
    return " | ".join(unit["values"]) if unit["labels"] else unit["body"]


def render(chosen: list[dict]) -> str:
    """Each group once (context, column headers, rows in page order); groups by best unit."""
    groups: dict[str, list[dict]] = {}
    for unit in chosen:
        groups.setdefault(unit["key"], []).append(unit)
    blocks = []
    for members in groups.values():
        members = sorted(members, key=lambda u: u["pos"])
        blocks.append("\n".join([_header(members[0]), *map(_line, members)]))
    return "\n\n".join(blocks)


def pack(units: list[dict], budget: int, tokenizer) -> str:
    """Greedy by score, charging each group header once; exact budget check."""
    chosen, opened, used = [], set(), 0
    for unit in sorted(units, key=lambda u: -u["score"]):
        cost = tokens(_line(unit) + "\n", tokenizer)
        if unit["key"] not in opened:
            cost += tokens(_header(unit) + "\n\n", tokenizer)
        if used + cost <= budget:
            chosen.append(unit)
            opened.add(unit["key"])
            used += cost
    text = render(chosen)
    while chosen and tokens(text, tokenizer) > budget:
        chosen.pop()
        text = render(chosen)
    return text


def pack_flat(units: list[dict], budget: int, tokenizer) -> str:
    """Highest-score texts, one per line (for chunker baselines without groups)."""
    chosen, used = [], 0
    for unit in sorted(units, key=lambda u: -u["score"]):
        cost = tokens(unit["completed"] + "\n", tokenizer)
        if used + cost <= budget:
            chosen.append(unit["completed"])
            used += cost
    return "\n".join(chosen)


def _build_pages(args: tuple[str, list[str], str]) -> list[tuple[str, list[dict]]]:
    from tessera.units import ZimPages, page_units

    zim, titles, builder = args
    if builder != "tessera":
        from tessera import baselines

        return [(t, baselines.chunker_units(zim, t, builder)) for t in titles]
    pages = ZimPages(zim)
    out = []
    for title in titles:
        raw = pages.html(title)
        out.append((title, page_units(title, raw) if raw else []))
    return out


def unit_pages(zim: str, titles: list[str], workers: int, builder: str) -> dict:
    """Units of every title, built in `workers` processes."""
    titles = sorted(set(titles))
    with Pool(workers) as pool:
        batches = pool.imap_unordered(
            _build_pages, [(zim, titles[i::workers], builder) for i in range(workers)]
        )
        return {title: units for batch in batches for title, units in batch}


def pools(args) -> dict:
    """Prefiltered unit pools for every question of the candidate file."""
    rows = read_jsonl(args.candidates)
    anchors = [anchor_pages(r["hits"], args.anchor_pages) for r in rows]
    pages = unit_pages(args.zim, [t for a in anchors for t in a], args.workers, args.builder)
    out = [
        {
            "load_index": row["load_index"],
            "problem": row["problem"],
            "units": prefilter(row["problem"], [u for t in anchor for u in pages[t]], args.prefilter),
        }
        for row, anchor in zip(rows, anchors, strict=True)
    ]
    write_jsonl(args.out / "pools.jsonl", out)
    return {"questions": len(out), "anchor_pages": len(pages)}


def score(args) -> dict:
    """BGE scores of completed unit text for one interleaved part of the pools."""
    rows = read_jsonl(args.out / "pools.jsonl")[args.part :: args.parts]
    pairs = [(r["problem"], u["completed"]) for r in rows for u in r["units"]]
    scores = iter(score_batched(bge_scorer(args.reranker), pairs, args.batch_size))
    for row in rows:
        for unit in row["units"]:
            unit["score"] = next(scores)
    write_jsonl(args.out / f"scored.part{args.part:02d}.jsonl", rows)
    return {"part": args.part, "questions": len(rows)}


def load_scored(out: Path) -> dict[int, list[dict]]:
    return {
        r["load_index"]: r["units"]
        for path in sorted(out.glob("scored.part*.jsonl"))
        for r in read_jsonl(path)
    }


def build(args) -> dict:
    """Top-k baseline and top k-1 + unit block reader inputs at the same budget."""
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
    pack_units = pack if args.builder == "tessera" else pack_flat
    scored = load_scored(args.out)
    baseline, tessera = [], []
    for row in read_jsonl(args.candidates):
        top = reranked(row["hits"], args.top_k)
        head = top[:-1] if len(top) == args.top_k else top
        budget = sum(tokens(h["text"], tokenizer) for h in top[len(head) :])
        text = pack_units(scored[row["load_index"]], budget, tokenizer)
        baseline.append({**row, "hits": top})
        tessera.append({**row, "hits": [*head, {**UNIT_HIT, "text": text}]})
    name = "tessera" if args.builder == "tessera" else args.builder
    write_jsonl(args.out / "baseline_candidates.jsonl", baseline)
    write_jsonl(args.out / f"{name}_candidates.jsonl", tessera)
    return {"questions": len(tessera), "output": str(args.out / f"{name}_candidates.jsonl")}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("action", choices=("pools", "score", "build"))
    parser.add_argument("--out", type=Path, required=True, help="run directory")
    parser.add_argument("--candidates", type=Path, help="reranked candidate JSONL")
    parser.add_argument("--zim", default="data/wikipedia/wikipedia_en_all_maxi_2025-08.zim")
    parser.add_argument("--builder", default="tessera", choices=("tessera", "docling", "unstructured"))
    parser.add_argument("--anchor-pages", type=int, default=10)
    parser.add_argument("--prefilter", type=int, default=100)
    parser.add_argument("--workers", type=int, default=32)
    parser.add_argument("--reranker", default="models/BAAI/bge-reranker-v2-m3")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--part", type=int, default=0)
    parser.add_argument("--parts", type=int, default=1)
    parser.add_argument("--tokenizer", default="models/Qwen/Qwen3.5-4B", help="reader tokenizer")
    parser.add_argument("--top-k", type=int, default=3)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    print(json.dumps({"pools": pools, "score": score, "build": build}[args.action](args), indent=2))


if __name__ == "__main__":
    main()
