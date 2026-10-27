"""PixelRAG's released text and screenshot pipelines, and Tessera on top of them.

Both pipelines search PixelRAG's released FAISS indexes over the same 2025-08
Wikipedia snapshot (`StarTrail-org/pixelrag-faiss-indexes`). Queries are encoded
with Qwen3-VL-Embedding-2B through its chat template (system message: the
instruction, user message: the question), last-token pooled and normalized, as
in PixelRAG's `pixelrag serve`.

PixelRAG text + BGE:

    python -m tessera.pixelrag search --pipeline text --examples Q.jsonl --out run/ --nprobe 128

  The released index stores vector IDs and character offsets; chunk text is
  rebuilt from the ZIM with Trafilatura 2.0, exactly as the index was built,
  and the top 50 are reranked with BGE-reranker-v2-m3 into `candidates.jsonl`.

PixelRAG screenshot + Qwen3-VL:

    python -m tessera.pixelrag search --pipeline screenshot --examples Q.jsonl --out run/
    python -m tessera.pixelrag render --out run/          # needs the PixelRAG renderer
    python -m tessera.pixelrag rerank --out run/          # Qwen3-VL-Reranker-2B, top 50
    python -m tessera.layer pools/score --candidates run/candidates.jsonl --out run/
    python -m tessera.pixelrag build --out run/           # top-3 tiles vs top-2 + units

  The fine-tuned screenshot index uses the released LoRA adapter, remapped for
  Qwen3VLForConditionalGeneration and merged at load time. `build` packs the
  unit block into the reader-token cost of the third tile under the reader's
  image processor. Read the outputs with `tessera.read --mode screenshot`.

The paper uses nprobe 128 for NQ and NQ-Tables and 2,000 for SimpleQA.
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import time
from pathlib import Path

import numpy as np

from tessera.common import read_jsonl, reranked, write_jsonl

INSTRUCTIONS = {
    "text": "Retrieve text relevant to the user's query.",
    "screenshot": "Retrieve images or text relevant to the user's query.",
}
INDEXES = {
    "text": "data/pixelrag/text_search_index_1024_normed",
    "screenshot": "data/pixelrag/search_index_lora_vit_ckpt200_v2",
}
RERANK_INSTRUCTION = "Given a search query, retrieve relevant candidates that answer the query."
TILE_WIDTH, TILE_TOKENS = 875, 864
# Wait for fonts and eagerly loaded images, then measure the full page height.
FULL_HEIGHT_PROBE = """new Promise(resolve => {
    Array.from(document.images).forEach(i => { i.loading = 'eager'; });
    const waitEagerImgs = Promise.all(Array.from(document.images).filter(i => !i.complete)
        .map(i => new Promise(r => { i.addEventListener('load', r, {once: true});
                                     i.addEventListener('error', r, {once: true}); })));
    const timeout = new Promise(r => setTimeout(r, 5000));
    Promise.race([Promise.all([document.fonts.ready, waitEagerImgs]), timeout]).then(() => {
        requestAnimationFrame(() => { requestAnimationFrame(() => {
            document.documentElement.style.scrollBehavior = 'auto';
            resolve(Math.max(document.documentElement.scrollHeight,
                             document.body ? document.body.scrollHeight : 0, 1));
        }); });
    });
})"""


def fp32_conv3d(model, torch) -> None:
    """Run the Qwen3-VL patch embedding in fp32: bf16 Conv3d is very slow before cuDNN 9.15."""
    version = torch.backends.cudnn.version()
    if version is None or version >= 91500:
        return
    patch = model.model.visual.patch_embed

    def forward(hidden):
        conv = patch.proj
        shape = (-1, patch.in_channels, patch.temporal_patch_size, patch.patch_size, patch.patch_size)
        weight, bias = conv.weight.data, conv.bias.data
        conv.weight.data, conv.bias.data = weight.float(), bias.float()
        out = conv(hidden.view(*shape).float()).view(-1, patch.embed_dim)
        conv.weight.data, conv.bias.data = weight, bias
        return out.to(torch.bfloat16)

    patch.forward = forward


def remap_adapter(source: Path, target: Path) -> Path:
    """The released BiQwen3 LoRA with keys renamed for Qwen3VLForConditionalGeneration."""
    import shutil

    from safetensors.torch import load_file, save_file

    if not (target / "adapter_model.safetensors").exists():
        state = load_file(source / "adapter_model.safetensors")
        renamed = {
            re.sub(r"^(base_model\.model\.)(language_model\.|visual\.)", r"\1model.\2", k): v
            for k, v in state.items()
        }
        target.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source / "adapter_config.json", target / "adapter_config.json")
        save_file(renamed, target / "adapter_model.safetensors")
    return target


def query_encoder(model_path: str, adapter: Path | None, instruction: str):
    import torch
    from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

    processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        model_path, trust_remote_code=True, dtype=torch.bfloat16
    )
    if adapter is not None:
        from peft import PeftModel

        model = PeftModel.from_pretrained(model, str(adapter)).merge_and_unload()
    fp32_conv3d(model, torch)
    model = model.to("cuda").eval()

    def encode(questions: list[str]) -> np.ndarray:
        texts = [
            processor.apply_chat_template(
                [
                    {"role": "system", "content": [{"type": "text", "text": instruction}]},
                    {"role": "user", "content": [{"type": "text", "text": q}]},
                ],
                tokenize=False,
                add_generation_prompt=True,
            )
            for q in questions
        ]
        inputs = processor(text=texts, return_tensors="pt", padding=True)
        inputs = {k: v.to("cuda") if hasattr(v, "to") else v for k, v in inputs.items()}
        with torch.inference_mode():
            hidden = model.model(**inputs).last_hidden_state
        last = inputs["attention_mask"].sum(dim=1) - 1
        pooled = hidden[torch.arange(hidden.size(0), device=hidden.device), last]
        return torch.nn.functional.normalize(pooled, p=2, dim=-1).float().cpu().numpy()

    return encode


def _article_text(archive, title: str) -> str:
    from trafilatura import extract

    try:
        entry = archive.get_entry_by_path(title)
        for _ in range(5):
            if not entry.is_redirect:
                break
            entry = entry.get_redirect_entry()
        content = bytes(entry.get_item().content)
        return (
            extract(
                content, include_links=False, include_images=False, include_tables=True, output_format="txt"
            )
            or ""
        )
    except Exception:  # noqa: BLE001 - an unreadable page gives empty chunks
        return ""


def search(args) -> dict:
    """Top-100 hits per question from a released index; text hits get their chunk text."""
    import faiss

    rows = read_jsonl(args.examples)
    adapter = None
    if args.pipeline == "screenshot":
        adapter = remap_adapter(args.adapter, args.out.parent / "lora_remapped")
    encode = query_encoder(args.query_model, adapter, INSTRUCTIONS[args.pipeline])
    batch = 16 if args.pipeline == "text" else 8
    vectors = np.concatenate(
        [encode([r["problem"] for r in rows[i : i + batch]]) for i in range(0, len(rows), batch)]
    )
    index_dir = Path(INDEXES[args.pipeline])
    ivf = faiss.read_index(str(index_dir / "index.faiss"), faiss.IO_FLAG_MMAP | faiss.IO_FLAG_READ_ONLY)
    ivf.nprobe = args.nprobe
    scores, ids = ivf.search(vectors, 100)
    meta = dict(np.load(index_dir / "metadata.npz"))
    titles = json.loads(Path(args.articles).read_text())
    out = []
    for row, row_scores, row_ids in zip(rows, scores, ids, strict=True):
        hits = []
        for rank, (score, vid) in enumerate(zip(row_scores, row_ids, strict=True), 1):
            if vid < 0:
                continue
            aid = int(meta["article_ids"][vid])
            hit = {
                "rank": rank,
                "dense_rank": rank,
                "dense_score": float(score),
                "vector_id": int(vid),
                "article_id": aid,
                "chunk_index": int(meta["chunk_indices"][vid]),
                "title": titles[aid],
            }
            if args.pipeline == "screenshot":
                hit |= {"tile_index": int(meta["tile_indices"][vid]), "url": titles[aid]}
            hits.append(hit)
        out.append({**row, "hits": hits})
    if args.pipeline == "text":
        out = _attach_released_text(out, meta, args.zim)
        from tessera.retrieve import rerank

        out = rerank(out, args.reranker)
        write_jsonl(args.out / "candidates.jsonl", out)
    else:
        write_jsonl(args.out / "dense.jsonl", out)
    return {"questions": len(out)}


def _attach_released_text(rows: list[dict], meta: dict, zim: str) -> list[dict]:
    """Chunk text from Trafilatura page text and the index's saved character offsets."""
    from libzim.reader import Archive

    archive = Archive(zim)
    article_ids, offsets = meta["article_ids"], meta["y_offsets"]
    pages: dict[int, str] = {}
    for row in rows:
        for hit in row["hits"]:
            vid, aid = hit["vector_id"], hit["article_id"]
            if aid not in pages:
                pages[aid] = _article_text(archive, hit["title"])
            text = pages[aid]
            start = min(max(int(offsets[vid]), 0), len(text))
            same_page = vid + 1 < len(article_ids) and article_ids[vid + 1] == aid
            end = min(max(int(offsets[vid + 1]), start), len(text)) if same_page else len(text)
            hit["text"] = text[start:end].strip()
    return rows


def render(args) -> dict:
    """Render the pages of the top-50 tiles from the ZIM and keep the indexed tiles."""
    import shutil
    from concurrent.futures import ThreadPoolExecutor
    from urllib.parse import quote

    from pixelrag_embed.chunk import chunk_article
    from pixelrag_render.backends import fast_cdp
    from pixelrag_render.render import render_urls

    from tessera.read import tile_path

    fast_cdp._WAIT_FONTS_IMGS = FULL_HEIGHT_PROBE
    needed: dict[int, dict] = {}
    for row in read_jsonl(args.out / "dense.jsonl"):
        for hit in row["hits"][:50]:
            needed.setdefault(hit["article_id"], {"url": hit["url"], "hits": []})["hits"].append(hit)
    todo = [
        (a, p)
        for a, p in sorted(needed.items())
        if not all(tile_path(h, args.tile_cache).is_file() for h in p["hits"])
    ]
    server = subprocess.Popen(
        [
            "python",
            "-m",
            "pixelrag_serve.zim_server",
            "--zim",
            args.zim,
            "--host",
            "127.0.0.1",
            "--port",
            "30900",
        ]
    )
    try:
        time.sleep(10)
        prefix = "http://127.0.0.1:30900/content/wikipedia_en_all_maxi_2025-08/"
        for start in range(0, len(todo), args.batch):
            batch = todo[start : start + args.batch]
            staging = args.tile_cache / f".staging-{start}"
            render_urls(
                [prefix + quote(p["url"], safe="") for _, p in batch],
                staging,
                stems=[str(a) for a, _ in batch],
                viewport_width=TILE_WIDTH,
                tile_height=8192,
                workers=args.workers,
            )
            rendered = sorted(staging.glob("*.png.tiles"))
            with ThreadPoolExecutor(args.workers) as pool:
                list(pool.map(lambda path: chunk_article(str(path)), rendered))
            for _, page in batch:
                for hit in page["hits"]:
                    target = tile_path(hit, args.tile_cache)
                    staged = staging / target.parent.name / target.name
                    if staged.is_file():
                        target.parent.mkdir(parents=True, exist_ok=True)
                        staged.replace(target)
            shutil.rmtree(staging, ignore_errors=True)
            print(f"rendered {min(start + args.batch, len(todo))}/{len(todo)} pages", flush=True)
    finally:
        server.terminate()
    return {"pages": len(needed), "rendered": len(todo)}


def _has_content(path: Path) -> bool:
    """A cached tile that exists and is not visually blank."""
    from PIL import Image, ImageStat

    if not path.is_file():
        return False
    if path.stat().st_size > 8192:
        return True
    with Image.open(path) as image:
        gray = image.convert("L")
        gray.thumbnail((256, 256))
        return ImageStat.Stat(gray).stddev[0] >= 1.0


def vl_reranker(model_path: str):
    """Qwen3-VL-Reranker yes/no scores for (question, tile) pairs."""
    import torch
    from PIL import Image
    from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

    processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True, padding_side="left")
    lm = (
        Qwen3VLForConditionalGeneration.from_pretrained(
            model_path, trust_remote_code=True, dtype=torch.bfloat16, attn_implementation="sdpa"
        )
        .to("cuda")
        .eval()
    )
    fp32_conv3d(lm, torch)
    vocab = processor.tokenizer.get_vocab()
    weight = (lm.lm_head.weight[vocab["yes"]] - lm.lm_head.weight[vocab["no"]]).detach()

    def score(question: str, images: list[Path]) -> list[float]:
        system = (
            "Judge whether the Document meets the requirements based on the Query and the "
            'Instruct provided. Note that the answer can only be "yes" or "no".'
        )
        conversations = [
            [
                {"role": "system", "content": [{"type": "text", "text": system}]},
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": f"<Instruct>: {RERANK_INSTRUCTION}"},
                        {"type": "text", "text": "<Query>:"},
                        {"type": "text", "text": question},
                        {"type": "text", "text": "\n<Document>:"},
                        {"type": "image", "image": str(path)},
                    ],
                },
            ]
            for path in images
        ]
        texts = processor.apply_chat_template(conversations, tokenize=False, add_generation_prompt=True)
        pictures = []
        for path in images:
            with Image.open(path) as image:
                pictures.append(image.convert("RGB"))
        inputs = processor(
            text=texts, images=pictures, return_tensors="pt", padding=True, truncation=True, max_length=8192
        )
        inputs = {k: v.to("cuda") if hasattr(v, "to") else v for k, v in inputs.items()}
        with torch.no_grad():
            hidden = lm.model(**inputs).last_hidden_state[:, -1]
            return torch.sigmoid(hidden @ weight).float().cpu().tolist()

    return score


def rerank(args) -> dict:
    """Rerank the top-50 tiles; tiles that could not be rendered keep no score."""
    from tessera.read import tile_path

    score = vl_reranker(args.reranker)
    out = []
    for row in read_jsonl(args.out / "dense.jsonl"):
        head = row["hits"][:50]
        ready = [h for h in head if _has_content(tile_path(h, args.tile_cache))]
        missing = [{**h, "rerank_score": None} for h in head if h not in ready]
        scores = []
        for start in range(0, len(ready), 4):
            batch = ready[start : start + 4]
            scores += score(row["problem"], [tile_path(h, args.tile_cache) for h in batch])
        order = sorted(range(len(ready)), key=lambda i: (-scores[i], ready[i]["dense_rank"]))
        hits = [{**ready[i], "rerank_score": scores[i]} for i in order] + missing + row["hits"][50:]
        out.append({**row, "hits": hits})
    write_jsonl(args.out / "candidates.jsonl", out)
    return {"questions": len(out)}


def image_tokens(path: Path, processor) -> int:
    """Reader tokens of one image under the reader's image processor."""
    from PIL import Image

    with Image.open(path) as image:
        grid = processor(images=[image.convert("RGB")])["image_grid_thw"][0]
    return int(grid[0] * grid[1] * grid[2]) // processor.merge_size**2


def build(args) -> dict:
    """Top-3 tiles, and top-2 tiles plus a unit block in the third tile's reader-token cost."""
    from transformers import AutoProcessor, AutoTokenizer

    from tessera.layer import UNIT_HIT, load_scored, pack
    from tessera.read import tile_path

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
    processor = AutoProcessor.from_pretrained(args.tokenizer).image_processor
    scored = load_scored(args.out)
    baseline, tessera = [], []
    for row in read_jsonl(args.out / "candidates.jsonl"):
        top = reranked(row["hits"], 3)
        third = tile_path(top[2], args.tile_cache) if len(top) == 3 else None
        budget = image_tokens(third, processor) if third and third.is_file() else TILE_TOKENS
        text = pack(scored[row["load_index"]], budget, tokenizer)
        baseline.append({**row, "hits": top})
        tessera.append({**row, "hits": [*top[:2], {**UNIT_HIT, "text": text, "modality": "text"}]})
    write_jsonl(args.out / "baseline_candidates.jsonl", baseline)
    write_jsonl(args.out / "tessera_candidates.jsonl", tessera)
    return {"questions": len(tessera)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("action", choices=("search", "render", "rerank", "build"))
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--pipeline", choices=("text", "screenshot"), default="screenshot")
    parser.add_argument("--examples", type=Path)
    parser.add_argument("--nprobe", type=int, default=128)
    parser.add_argument("--query-model", default="models/Qwen/Qwen3-VL-Embedding-2B")
    parser.add_argument(
        "--adapter",
        type=Path,
        default=Path("models/Chrisyichuan/wiki-screenshot-embedding-lora/lora_vit/ckpt200"),
    )
    parser.add_argument("--articles", default="data/pixelrag/articles.json")
    parser.add_argument("--zim", default="data/wikipedia/wikipedia_en_all_maxi_2025-08.zim")
    parser.add_argument("--reranker", default=None, help="BGE (text) or Qwen3-VL-Reranker-2B (screenshot)")
    parser.add_argument("--tile-cache", type=Path, default=Path("data/pixelrag/tiles"))
    parser.add_argument("--batch", type=int, default=2000, help="pages per render batch")
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--tokenizer", default="models/Qwen/Qwen3.5-4B")
    args = parser.parse_args()
    if args.reranker is None:
        text = args.pipeline == "text" and args.action == "search"
        args.reranker = "models/BAAI/bge-reranker-v2-m3" if text else "models/Qwen/Qwen3-VL-Reranker-2B"
    args.out.mkdir(parents=True, exist_ok=True)
    actions = {"search": search, "render": render, "rerank": rerank, "build": build}
    print(json.dumps(actions[args.action](args), indent=2))


if __name__ == "__main__":
    main()
