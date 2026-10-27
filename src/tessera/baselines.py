"""Baselines in the matched-budget comparison (paper Table 4).

Structure-aware chunkers as the unit builder (`tessera.layer --builder ...`):

  docling       Docling HTML backend + HybridChunker (256 BGE tokens) with
                `contextualize` (heading path; table rows as header = value)
  unstructured  Unstructured `partition_html` + `chunk_by_title`
                (max 1,000 characters, new chunk after 800)

  Every other step is Tessera's: anchor pages, word-overlap prefilter, BGE
  scoring, and the third-slot budget; chunks are packed by score, one per line.
  A page whose chunking takes over 120 s contributes no chunks.

HtmlRAG (released two-stage pruner, htmlrag 0.1.1) in the third slot:

    python -m tessera.baselines htmlrag --candidates C.jsonl --out run/ --gpus 0 1

  cleans the HTML of the same ten anchor pages, prunes it with the block-tree
  ranker to an 8,192-token window, then with HTML-Pruner-Llama-1B to the token
  length of the third chunk. `--ranker embed` is the released BGE-large
  bi-encoder (HtmlRAG); `--ranker ce` scores the same blocks with
  BGE-reranker-v2-m3 (HtmlRAG-CE). Writes `htmlrag_{ranker}_candidates.jsonl`
  (top-2 chunks + pruned HTML). A question whose pruning fails gets an empty block.
"""

from __future__ import annotations

import argparse
import io
import json
import os
import signal
from pathlib import Path

from tessera.common import bge_scorer, read_jsonl, reranked, score_batched, tokens, write_jsonl
from tessera.layer import UNIT_HIT, anchor_pages

BGE = "models/BAAI/bge-reranker-v2-m3"  # Docling chunk-size tokenizer
_WORKER: dict = {}


class _Timeout(Exception):
    pass


def _alarm(signum, frame) -> None:
    raise _Timeout


def _chunker(builder: str, zim: str):
    if builder in _WORKER:
        return _WORKER[builder]
    from tessera.units import ZimPages

    signal.signal(signal.SIGALRM, _alarm)
    _WORKER.setdefault("pages", ZimPages(zim))
    if builder == "docling":
        from docling.chunking import HybridChunker
        from docling.datamodel.base_models import DocumentStream
        from docling.document_converter import DocumentConverter
        from docling_core.transforms.chunker.tokenizer.huggingface import HuggingFaceTokenizer
        from transformers import AutoTokenizer

        converter = DocumentConverter()
        chunker = HybridChunker(
            tokenizer=HuggingFaceTokenizer(
                tokenizer=AutoTokenizer.from_pretrained(BGE),
                max_tokens=256,
            )
        )

        def chunk(raw: bytes) -> list[str]:
            stream = DocumentStream(name="page.html", stream=io.BytesIO(raw))
            document = converter.convert(stream).document
            return [chunker.contextualize(c) for c in chunker.chunk(document)]
    else:
        from unstructured.chunking.title import chunk_by_title
        from unstructured.partition.html import partition_html

        def chunk(raw: bytes) -> list[str]:
            elements = partition_html(text=raw.decode("utf-8", "replace"))
            chunks = chunk_by_title(elements, max_characters=1000, new_after_n_chars=800)
            return [c.text for c in chunks if c.text.strip()]

    _WORKER[builder] = chunk
    return chunk


def chunker_units(zim: str, title: str, builder: str) -> list[dict]:
    """One page's chunks as units with only `completed` text."""
    chunk = _chunker(builder, zim)
    raw = _WORKER["pages"].html(title)
    if raw is None:
        return []
    signal.alarm(120)
    try:
        return [{"completed": text} for text in chunk(raw)]
    except Exception:  # noqa: BLE001 - timeouts and parser failures give no chunks
        return []
    finally:
        signal.alarm(0)


def _init_htmlrag(args, gpus: list[str]) -> None:
    from multiprocessing import current_process

    index = (current_process()._identity or (1,))[0] - 1
    os.environ["CUDA_VISIBLE_DEVICES"] = gpus[index % len(gpus)]
    from htmlrag import EmbedHTMLPruner
    from libzim.reader import Archive
    from transformers import PreTrainedTokenizerFast

    from tessera.htmlrag_llama import load_gen_pruner

    _WORKER.update(
        archive=Archive(args.zim),
        tokenizer=PreTrainedTokenizerFast(tokenizer_file=f"{args.tokenizer}/tokenizer.json"),
        embed=EmbedHTMLPruner(
            embed_model=args.embed_model,
            local_inference=True,
            query_instruction_for_retrieval="Represent this sentence for searching relevant passages: ",
        ),
        gen=load_gen_pruner(args.gen_model),
        ce=bge_scorer(args.reranker) if args.ranker == "ce" else None,
    )


def _page_html(title: str) -> str:
    entry = _WORKER["archive"].get_entry_by_path(title)
    while entry.is_redirect:
        entry = entry.get_redirect_entry()
    return bytes(entry.get_item().content).decode("utf-8", "replace")


def _prune(job: dict) -> dict:
    from htmlrag import build_block_tree, clean_html

    w = _WORKER
    try:
        html = "\n".join(clean_html(_page_html(t)) for t in job["anchors"])
        tree, html = build_block_tree(html, max_node_words=256)
        if w["ce"] is None:
            ranks = w["embed"].calculate_block_rankings(job["problem"], html, tree)
        else:
            texts = [block[0].get_text() for block in tree]
            scores = score_batched(w["ce"], [(job["problem"], t) for t in texts], 64)
            ranks = sorted(range(len(texts)), key=lambda i: -scores[i])
        html = w["embed"].prune_HTML(html, tree, ranks, w["tokenizer"], 8192)
        tree, html = build_block_tree(html, max_node_words=128)
        ranks = w["gen"].calculate_block_rankings(job["problem"], html, tree)
        pruned = w["gen"].prune_HTML(html, tree, ranks, w["tokenizer"], job["budget"])
        return {**job, "html": pruned, "error": None}
    except Exception as error:  # noqa: BLE001 - recorded; the question gets an empty block
        return {**job, "html": "", "error": repr(error)}


def htmlrag(args) -> dict:
    from multiprocessing import get_context

    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
    rows = read_jsonl(args.candidates)
    jobs = []
    for row in rows:
        top = reranked(row["hits"], 3)
        budget = tokens(top[2]["text"], tokenizer)
        jobs.append(
            {
                "load_index": row["load_index"],
                "problem": row["problem"],
                "anchors": anchor_pages(row["hits"]),
                "budget": budget,
            }
        )
    gpus = [g for g in args.gpus for _ in range(args.workers_per_gpu)]
    with get_context("spawn").Pool(len(gpus), initializer=_init_htmlrag, initargs=(args, gpus)) as pool:
        pruned = {r["load_index"]: r for r in pool.imap_unordered(_prune, jobs)}
    out = [
        {
            **row,
            "hits": [
                *reranked(row["hits"], 2),
                {**UNIT_HIT, "title": "htmlrag", "text": pruned[row["load_index"]]["html"]},
            ],
        }
        for row in rows
    ]
    write_jsonl(args.out / f"htmlrag_{args.ranker}_candidates.jsonl", out)
    return {"questions": len(out), "failures": sum(bool(r["error"]) for r in pruned.values())}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("action", choices=("htmlrag",))
    parser.add_argument("--candidates", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--ranker", choices=("embed", "ce"), default="embed")
    parser.add_argument("--zim", default="data/wikipedia/wikipedia_en_all_maxi_2025-08.zim")
    parser.add_argument("--tokenizer", default="models/Qwen/Qwen3.5-4B")
    parser.add_argument("--embed-model", default="models/BAAI/bge-large-en")
    parser.add_argument("--gen-model", default="models/zstanjj/HTML-Pruner-Llama-1B")
    parser.add_argument("--reranker", default="models/BAAI/bge-reranker-v2-m3")
    parser.add_argument("--gpus", nargs="+", default=["0"])
    parser.add_argument("--workers-per-gpu", type=int, default=3)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    print(json.dumps(htmlrag(args), indent=2))


if __name__ == "__main__":
    main()
