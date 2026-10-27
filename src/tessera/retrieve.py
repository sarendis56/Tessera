"""Hybrid text pipeline end to end: dense + BM25, RRF, and BGE reranking.

    python -m tessera.retrieve --examples data/populations/nq.jsonl --out runs/hybrid/nq

writes `dense.jsonl`, `bm25.jsonl`, and `candidates.jsonl`: the top 100 of
reciprocal rank fusion (k = 60) over the two top-100 lists, with the top 50
reranked by BGE-reranker-v2-m3 (`rerank_score`, 2,048-token pairs). The reader
baseline is the top three reranked chunks; `tessera.layer` reads the same file.
Steps whose output exists are skipped. Dense query encoding needs one GPU,
reranking another (or the same, in sequence).
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

from tessera.common import bge_scorer, read_jsonl, score_batched, write_jsonl


def rrf(dense: list[dict], sparse: list[dict], k: int = 60, top_k: int = 100) -> list[dict]:
    """Fuse two ranked lists per question by vector ID; ties go to the better single rank."""
    by_index = {r["load_index"]: r for r in sparse}
    out = []
    for row in dense:
        union: dict[int, dict] = {}
        for rank, hit in enumerate(row["hits"][:100], 1):
            union[hit["vector_id"]] = {**hit, "dense_rank": rank, "bm25_rank": None}
        for rank, hit in enumerate(by_index[row["load_index"]]["hits"][:100], 1):
            if hit["vector_id"] in union:
                union[hit["vector_id"]].update(bm25_rank=rank, bm25_score=hit["bm25_score"])
            else:
                union[hit["vector_id"]] = {**hit, "dense_rank": None, "bm25_rank": rank}

        def ranks(hit: dict) -> list[int]:
            return [r for r in (hit["dense_rank"], hit["bm25_rank"]) if r]

        hits = [{**h, "rrf_score": sum(1 / (k + r) for r in ranks(h))} for h in union.values()]
        hits.sort(key=lambda h: (-h["rrf_score"], min(ranks(h)), h["vector_id"]))
        out.append({**row, "hits": [{**h, "rank": i} for i, h in enumerate(hits[:top_k], 1)]})
    return out


def hydrate(rows: list[dict], chunks_dir: Path, depth: int = 50) -> list[dict]:
    """Attach chunk text to the top `depth` hits."""
    from tessera.corpus import ChunkStore

    texts = ChunkStore(chunks_dir).texts(h["vector_id"] for r in rows for h in r["hits"][:depth])
    return [
        {
            **r,
            "hits": [
                {**h, "text": texts[h["vector_id"]]} if i < depth else h for i, h in enumerate(r["hits"])
            ],
        }
        for r in rows
    ]


def rerank(rows: list[dict], model: str, depth: int = 50) -> list[dict]:
    """BGE scores for the top `depth` hits, which move ahead of the unscored tail."""
    scorer = bge_scorer(model, max_length=2048)
    out = []
    for row in rows:
        head = row["hits"][:depth]
        scores = score_batched(scorer, [(row["problem"], h.get("text") or "") for h in head], 32)
        order = sorted(range(len(head)), key=lambda i: (-scores[i], i))
        hits = [{**head[i], "rerank_score": scores[i]} for i in order] + row["hits"][depth:]
        out.append({**row, "hits": [{**h, "rank": i} for i, h in enumerate(hits, 1)]})
    return out


def _step(output: Path, *command: str) -> None:
    # vLLM can abort during teardown after the output is written, so the file decides success.
    if not output.exists():
        subprocess.run([sys.executable, "-m", *command, "--output", str(output)])
    if not output.exists():
        raise RuntimeError(f"{' '.join(command[:2])} did not write {output}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--examples", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--chunks", default="data/hybrid/chunks")
    parser.add_argument("--reranker", default="models/BAAI/bge-reranker-v2-m3")
    parser.add_argument("--gpu-memory-utilization", default="0.6")
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    dense, sparse = args.out / "dense.jsonl", args.out / "bm25.jsonl"
    # Separate processes, so vLLM and the JVM release their memory before reranking.
    _step(
        dense,
        "tessera.dense",
        "search",
        "--examples",
        str(args.examples),
        "--chunks",
        args.chunks,
        "--gpu-memory-utilization",
        args.gpu_memory_utilization,
    )
    _step(sparse, "tessera.bm25", "search", "--examples", str(args.examples), "--chunks", args.chunks)
    target = args.out / "candidates.jsonl"
    if not target.exists():
        fused = hydrate(rrf(read_jsonl(dense), read_jsonl(sparse)), Path(args.chunks))
        write_jsonl(target, rerank(fused, args.reranker))
    print(json.dumps({"candidates": str(target)}))


if __name__ == "__main__":
    main()
