"""Sparse side of the hybrid pipeline: Anserini BM25 over the same chunks.

    python -m tessera.bm25 export      # JsonCollection, one document per vector ID
    python -m tessera.bm25 index       # Lucene index, English analyzer, Porter stemmer
    python -m tessera.bm25 search --examples Q.jsonl --output bm25.jsonl

Search uses k1 = 1.2, b = 0.6, and returns the top 100 chunks per question.
Needs Java 21 and the Anserini 2.2.0 fatjar (see scripts/download.sh).
"""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path
from urllib.parse import quote

import numpy as np

from tessera.common import read_jsonl, write_jsonl


def export(args) -> dict:
    import pyarrow.parquet as pq

    out = args.collection
    out.mkdir(parents=True, exist_ok=True)
    vector_id = 0
    for path in sorted(args.chunks.glob("part-*.parquet")):
        target = out / path.with_suffix(".jsonl").name
        texts = pq.read_table(path, columns=["text"]).column("text").to_pylist()
        if not target.exists():
            tmp = target.with_name(f".{target.name}.tmp")
            with tmp.open("w") as stream:
                for offset, text in enumerate(texts):
                    stream.write(
                        json.dumps({"id": str(vector_id + offset), "contents": text}, ensure_ascii=False)
                        + "\n"
                    )
            tmp.replace(target)
        vector_id += len(texts)
    return {"documents": vector_id}


def _java(args, main: str, *options: str) -> None:
    command = [
        args.java,
        f"-Xms{args.heap_gb}g",
        f"-Xmx{args.heap_gb}g",
        "-cp",
        args.anserini,
        main,
        *options,
    ]
    subprocess.run(command, check=True)


def index(args) -> dict:
    _java(
        args,
        "io.anserini.index.IndexCollection",
        "-collection",
        "JsonCollection",
        "-generator",
        "DefaultLuceneDocumentGenerator",
        "-threads",
        str(args.threads),
        "-memoryBuffer",
        "16384",
        "-input",
        str(args.collection),
        "-index",
        str(args.bm25_index),
        "-language",
        "en",
        "-stemmer",
        "porter",
    )
    return {"index": str(args.bm25_index)}


def search(args) -> dict:
    rows = read_jsonl(args.examples)
    topics = args.output.with_suffix(".topics.tsv")
    trec = args.output.with_suffix(".trec")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with topics.open("w") as stream:
        for row in rows:
            query = " ".join(row["problem"].replace("\t", " ").splitlines())
            stream.write(f"{int(row['load_index'])}\t{query}\n")
    _java(
        args,
        "io.anserini.search.SearchCollection",
        "-index",
        str(args.bm25_index),
        "-topics",
        str(topics),
        "-topicReader",
        "TsvInt",
        "-output",
        str(trec),
        "-bm25",
        "-bm25.k1",
        "1.2",
        "-bm25.b",
        "0.6",
        "-hits",
        str(args.top_k),
        "-threads",
        str(args.threads),
    )
    ranked: dict[int, list[tuple[int, float]]] = {}
    for line in trec.open():
        qid, _, doc, rank, score, _ = line.split()
        ranked.setdefault(int(qid), []).append((int(rank), int(doc), float(score)))
    with np.load(args.chunks / "meta.npz") as meta:
        article_ids, chunk_indices = meta["article_ids"], meta["chunk_indices"]
    titles = json.loads(Path(args.articles).read_text())
    out = []
    for row in rows:
        hits = []
        for rank, vector_id, score in sorted(ranked.get(int(row["load_index"]), [])):
            title = titles[int(article_ids[vector_id])]
            hits.append(
                {
                    "rank": rank,
                    "bm25_rank": rank,
                    "bm25_score": score,
                    "vector_id": vector_id,
                    "article_id": int(article_ids[vector_id]),
                    "chunk_index": int(chunk_indices[vector_id]),
                    "title": title,
                    "url": "https://en.wikipedia.org/wiki/" + quote(title, safe="():,/_-"),
                }
            )
        out.append({**row, "hits": hits})
    write_jsonl(args.output, out)
    return {"questions": len(out)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("action", choices=("export", "index", "search"))
    parser.add_argument("--chunks", type=Path, default=Path("data/hybrid/chunks"))
    parser.add_argument("--collection", type=Path, default=Path("data/hybrid/bm25_collection"))
    parser.add_argument("--bm25-index", type=Path, default=Path("data/hybrid/bm25_index"))
    parser.add_argument("--articles", default="data/pixelrag/articles.json")
    parser.add_argument("--java", default="tools/jre/bin/java")
    parser.add_argument("--anserini", default="tools/anserini-2.2.0-fatjar.jar")
    parser.add_argument("--heap-gb", type=int, default=32)
    parser.add_argument("--threads", type=int, default=32)
    parser.add_argument("--examples", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--top-k", type=int, default=100)
    args = parser.parse_args()
    print(json.dumps({"export": export, "index": index, "search": search}[args.action](args), indent=2))


if __name__ == "__main__":
    main()
