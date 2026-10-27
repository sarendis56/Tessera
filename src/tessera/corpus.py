"""Text corpus of the hybrid pipeline: Trafilatura text cut into 1,024-token chunks.

    python -m tessera.corpus build --out data/hybrid/chunks --workers 64
    python -m tessera.corpus finalize --out data/hybrid/chunks

`build` extracts every article of PixelRAG's released text index (7,060,072
articles: the article IDs in its `metadata.npz`, titles from `articles.json`)
from the 2025-08 Wikipedia ZIM with Trafilatura 2.0 (plain text, tables kept,
links and images dropped), and cuts each text into consecutive 1,024-token
windows of the Qwen3-VL-Embedding-2B tokenizer, the tokenizer PixelRAG chunked
with. Shard `part-XXXXX.parquet` holds article IDs [32768·X, 32768·(X+1)); a
finished shard is skipped on rerun. `finalize` writes `meta.npz`, the article
ID and chunk index of every vector ID (the global row number over shards).
"""

from __future__ import annotations

import argparse
import json
from multiprocessing import Pool
from pathlib import Path

import numpy as np

SHARD_IDS = 32768
EXTRACT = {"include_links": False, "include_images": False, "include_tables": True, "output_format": "txt"}
_WORKER: dict = {}


def _init(zim: str, tokenizer: str) -> None:
    from libzim.reader import Archive
    from transformers import AutoTokenizer

    _WORKER.update(archive=Archive(zim), tokenizer=AutoTokenizer.from_pretrained(tokenizer))


def extract(title: str) -> str:
    """Trafilatura text of one article after at most eight redirects."""
    from trafilatura import extract as trafilatura_extract

    try:
        entry = _WORKER["archive"].get_entry_by_path(title)
        for _ in range(8):
            if not entry.is_redirect:
                break
            entry = entry.get_redirect_entry()
        if entry.is_redirect:
            return ""
        return trafilatura_extract(bytes(entry.get_item().content), **EXTRACT) or ""
    except Exception:  # noqa: BLE001 - a missing or broken page yields no chunks
        return ""


def windows(text: str, offsets: list[tuple[int, int]], size: int) -> list[tuple[int, int, int]]:
    """Character spans and token counts of consecutive `size`-token windows."""
    spans = []
    for start in range(0, len(offsets), size):
        end = min(start + size, len(offsets))
        char_start = 0 if start == 0 else offsets[start][0]
        char_end = len(text) if end == len(offsets) else offsets[end][0]
        spans.append((char_start, char_end, end - start))
    return spans


def _shard(task: tuple[int, list[tuple[int, str]], str, int]) -> int:
    import pyarrow as pa
    import pyarrow.parquet as pq

    shard, articles, out, size = task
    path = Path(out) / f"part-{shard:05d}.parquet"
    if path.exists():
        return shard
    texts = [(article_id, extract(title)) for article_id, title in articles]
    texts = [(a, t) for a, t in texts if t.strip()]
    encoded = (
        _WORKER["tokenizer"]([t for _, t in texts], add_special_tokens=False, return_offsets_mapping=True)[
            "offset_mapping"
        ]
        if texts
        else []
    )
    columns: dict[str, list] = {"article_id": [], "chunk_index": [], "text": [], "token_count": []}
    for (article_id, text), offsets in zip(texts, encoded, strict=True):
        for index, (start, end, count) in enumerate(windows(text, offsets, size)):
            columns["article_id"].append(article_id)
            columns["chunk_index"].append(index)
            columns["text"].append(text[start:end])
            columns["token_count"].append(count)
    tmp = path.with_name(f".{path.name}.tmp")
    pq.write_table(pa.table(columns), tmp, compression="zstd", row_group_size=8192)
    tmp.replace(path)
    return shard


def build(args) -> dict:
    titles = json.loads(Path(args.articles).read_text())
    with np.load(args.eligible) as metadata:
        eligible = np.unique(metadata["article_ids"])
    args.out.mkdir(parents=True, exist_ok=True)
    shards: dict[int, list[tuple[int, str]]] = {}
    for article_id in eligible.tolist():
        shards.setdefault(article_id // SHARD_IDS, []).append((article_id, titles[article_id]))
    tasks = [(s, a, str(args.out), args.chunk_tokens) for s, a in sorted(shards.items())]
    with Pool(args.workers, initializer=_init, initargs=(args.zim, args.tokenizer)) as pool:
        for done, shard in enumerate(pool.imap_unordered(_shard, tasks), 1):
            print(f"shard {shard} ({done}/{len(tasks)})", flush=True)
    return {"articles": int(eligible.size), "shards": len(tasks)}


def finalize(args) -> dict:
    import pyarrow.parquet as pq

    article_ids, chunk_indices = [], []
    for path in sorted(args.out.glob("part-*.parquet")):
        table = pq.read_table(path, columns=["article_id", "chunk_index"])
        article_ids.append(table.column("article_id").to_numpy())
        chunk_indices.append(table.column("chunk_index").to_numpy())
    np.savez(
        args.out / "meta.npz",
        article_ids=np.concatenate(article_ids).astype(np.int64),
        chunk_indices=np.concatenate(chunk_indices).astype(np.int32),
        shard_rows=np.array([len(a) for a in article_ids], dtype=np.int64),
    )
    return {"chunks": int(sum(len(a) for a in article_ids)), "shards": len(article_ids)}


class ChunkStore:
    """Chunk text by vector ID, reading only the needed Parquet row groups."""

    def __init__(self, chunks_dir: Path | str) -> None:
        self.dir = Path(chunks_dir)
        with np.load(self.dir / "meta.npz") as meta:
            self.article_ids = meta["article_ids"]
            self.chunk_indices = meta["chunk_indices"]
            self.ends = np.cumsum(meta["shard_rows"])
        self.paths = sorted(self.dir.glob("part-*.parquet"))

    def texts(self, vector_ids) -> dict[int, str]:
        import pyarrow.parquet as pq

        wanted: dict[tuple[int, int], list[tuple[int, int]]] = {}
        files = {}
        for vector_id in sorted(set(int(v) for v in vector_ids)):
            shard = int(np.searchsorted(self.ends, vector_id, side="right"))
            local = vector_id - (int(self.ends[shard - 1]) if shard else 0)
            if shard not in files:
                files[shard] = pq.ParquetFile(self.paths[shard])
            groups = files[shard].metadata
            start = 0
            for group in range(groups.num_row_groups):
                rows = groups.row_group(group).num_rows
                if local < start + rows:
                    wanted.setdefault((shard, group), []).append((vector_id, local - start))
                    break
                start += rows
        out = {}
        for (shard, group), items in wanted.items():
            column = files[shard].read_row_group(group, columns=["text"]).column("text")
            for vector_id, offset in items:
                out[vector_id] = column[offset].as_py()
        return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("action", choices=("build", "finalize"))
    parser.add_argument("--out", type=Path, default=Path("data/hybrid/chunks"))
    parser.add_argument("--zim", default="data/wikipedia/wikipedia_en_all_maxi_2025-08.zim")
    parser.add_argument("--articles", default="data/pixelrag/articles.json")
    parser.add_argument("--eligible", default="data/pixelrag/text_search_index_1024_normed/metadata.npz")
    parser.add_argument("--tokenizer", default="models/Qwen/Qwen3-VL-Embedding-2B")
    parser.add_argument("--chunk-tokens", type=int, default=1024)
    parser.add_argument("--workers", type=int, default=64)
    args = parser.parse_args()
    print(json.dumps(build(args) if args.action == "build" else finalize(args), indent=2))


if __name__ == "__main__":
    main()
