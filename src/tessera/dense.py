"""Dense side of the hybrid pipeline: Qwen3-Embedding-0.6B and a FAISS IVF-Flat index.

    python -m tessera.dense embed --part 0 --parts 8      # one process per GPU
    python -m tessera.dense index                          # IVF4096, inner product
    python -m tessera.dense search --examples Q.jsonl --output dense.jsonl

Documents are embedded as raw chunk text, queries as
"Instruct: Given a web search query, retrieve relevant passages that answer the
query\\nQuery:{query}", both with last-token pooling and L2 normalization.
The index is trained on 500,000 vectors sampled with seed 42 and searched with
nprobe 128.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from urllib.parse import quote

import numpy as np

from tessera.common import read_jsonl, write_jsonl

INSTRUCTION = "Given a web search query, retrieve relevant passages that answer the query"


def embedder(model: str, gpu_memory_utilization: float = 0.6):
    """Normalized last-token embeddings of token-ID prompts, via vLLM pooling."""
    from vllm import LLM

    llm = LLM(
        model=model,
        runner="pooling",
        dtype="bfloat16",
        max_model_len=8192,
        max_num_batched_tokens=131072,
        gpu_memory_utilization=gpu_memory_utilization,
    )
    tokenizer = llm.get_tokenizer()

    def encode(texts: list[str]) -> np.ndarray:
        ids = tokenizer(texts, add_special_tokens=True)["input_ids"]
        outputs = llm.embed([{"prompt_token_ids": i[-8192:]} for i in ids], use_tqdm=False)
        vectors = np.asarray([o.outputs.embedding for o in outputs], dtype=np.float32)
        return vectors / np.linalg.norm(vectors, axis=1, keepdims=True)

    return encode


def embed(args) -> dict:
    import pyarrow.parquet as pq

    paths = sorted(args.chunks.glob("part-*.parquet"))[args.part :: args.parts]
    args.embeddings.mkdir(parents=True, exist_ok=True)
    encode = embedder(args.model)
    for path in paths:
        target = args.embeddings / path.with_suffix(".npy").name
        if target.exists():
            continue
        texts = pq.read_table(path, columns=["text"]).column("text").to_pylist()
        vectors = (
            np.concatenate([encode(texts[i : i + 512]) for i in range(0, len(texts), 512)])
            if texts
            else np.zeros((0, 1024), np.float32)
        )
        tmp = target.with_name(f".{target.name}.tmp.npy")
        np.save(tmp, vectors.astype(np.float16))
        tmp.replace(target)
        print(f"embedded {path.name}: {len(texts)}", flush=True)
    return {"shards": len(paths)}


def _unit(vectors: np.ndarray) -> np.ndarray:
    vectors = np.asarray(vectors, dtype=np.float32)
    return vectors / np.linalg.norm(vectors, axis=1, keepdims=True)


def index(args) -> dict:
    """Train on GPU, add in GPU groups of 8M vectors, merge, and save on CPU."""
    import faiss

    shards = sorted(args.embeddings.glob("part-*.npy"))
    arrays = [np.load(p, mmap_mode="r") for p in shards]
    ends = np.cumsum([len(a) for a in arrays])
    total = int(ends[-1])
    sample = np.sort(np.random.default_rng(42).choice(total, size=min(500_000, total), replace=False))
    rows = []
    for shard, array in enumerate(arrays):
        start = int(ends[shard - 1]) if shard else 0
        local = sample[(sample >= start) & (sample < ends[shard])] - start
        rows.append(array[local])
    resources = faiss.StandardGpuResources()
    cpu = faiss.IndexIVFFlat(faiss.IndexFlatIP(1024), 1024, args.nlist, faiss.METRIC_INNER_PRODUCT)
    gpu = faiss.index_cpu_to_gpu(resources, 0, cpu)
    gpu.train(_unit(np.concatenate(rows)))
    trained = faiss.index_gpu_to_cpu(gpu)
    merged, added = None, 0
    group: list[np.ndarray] = []
    for i, array in enumerate(arrays):
        group.append(array)
        if sum(len(a) for a in group) < 8_000_000 and i + 1 < len(arrays):
            continue
        part = faiss.index_cpu_to_gpu(resources, 0, trained)
        for a in group:
            for start in range(0, len(a), 50_000):
                part.add(_unit(a[start : start + 50_000]))
        part = faiss.index_gpu_to_cpu(part)
        if merged is None:
            merged = part
        else:
            merged.merge_from(part, added)
        added = merged.ntotal
        group = []
    merged.nprobe = 128
    faiss.write_index(merged, str(args.index))
    return {"vectors": int(merged.ntotal), "nlist": args.nlist}


def search(args) -> dict:
    """Top-100 dense hits per question, with page titles from PixelRAG's articles.json."""
    import faiss

    rows = read_jsonl(args.examples)
    encode = embedder(args.model, args.gpu_memory_utilization)
    queries = [f"Instruct: {INSTRUCTION}\nQuery:{r['problem']}" for r in rows]
    vectors = np.concatenate([encode(queries[i : i + 1024]) for i in range(0, len(queries), 1024)])
    del encode
    faiss.omp_set_num_threads(args.threads)
    ivf = faiss.read_index(str(args.index), faiss.IO_FLAG_MMAP | faiss.IO_FLAG_READ_ONLY)
    ivf.nprobe = args.nprobe
    scores, ids = ivf.search(vectors, args.top_k)
    with np.load(args.chunks / "meta.npz") as meta:
        article_ids, chunk_indices = meta["article_ids"], meta["chunk_indices"]
    titles = json.loads(Path(args.articles).read_text())
    out = []
    for row, row_scores, row_ids in zip(rows, scores, ids, strict=True):
        hits = []
        for rank, (score, vector_id) in enumerate(zip(row_scores, row_ids, strict=True), 1):
            if vector_id < 0:
                continue
            title = titles[int(article_ids[vector_id])]
            hits.append(
                {
                    "rank": rank,
                    "dense_rank": rank,
                    "dense_score": float(score),
                    "vector_id": int(vector_id),
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
    parser.add_argument("action", choices=("embed", "index", "search"))
    parser.add_argument("--chunks", type=Path, default=Path("data/hybrid/chunks"))
    parser.add_argument("--embeddings", type=Path, default=Path("data/hybrid/embeddings"))
    parser.add_argument("--index", type=Path, default=Path("data/hybrid/ivf4096.faiss"))
    parser.add_argument("--model", default="models/Qwen/Qwen3-Embedding-0.6B")
    parser.add_argument("--articles", default="data/pixelrag/articles.json")
    parser.add_argument("--part", type=int, default=0)
    parser.add_argument("--parts", type=int, default=1)
    parser.add_argument("--nlist", type=int, default=4096)
    parser.add_argument("--examples", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--nprobe", type=int, default=128)
    parser.add_argument("--top-k", type=int, default=100)
    parser.add_argument("--threads", type=int, default=64)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.6)
    args = parser.parse_args()
    print(json.dumps({"embed": embed, "index": index, "search": search}[args.action](args), indent=2))


if __name__ == "__main__":
    main()
