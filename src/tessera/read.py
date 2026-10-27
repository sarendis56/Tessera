"""Reader: answer each question from its top-k evidence with an OpenAI-compatible server.

Serve the reader with vLLM (greedy decoding, thinking disabled), e.g.

    vllm serve models/Qwen/Qwen3.5-4B --served-model-name Qwen/Qwen3.5-4B \
        --max-model-len 32768 --dtype bfloat16 --port 8010

then run

    python -m tessera.read candidates.jsonl responses.jsonl --model Qwen/Qwen3.5-4B

`--mode text` (text pipelines) joins the evidence texts and appends the question;
`--mode screenshot` (PixelRAG screenshot pipeline) sends the question, then each
hit as an image (`image_path`, or a tile under `--tile-cache`) or as text when the
hit has `modality: "text"`, as in PixelRAG's released screenshot reader. Rows
already answered in the output file are skipped.
"""

from __future__ import annotations

import argparse
import asyncio
import base64
import json
from pathlib import Path

from tessera.common import read_jsonl

SYSTEM_PROMPT = (
    "You are a research assistant who answers questions based on provided evidence.\n"
    "Use <think></think> tags to show your reasoning if needed.\n"
    "Answer the question directly and concisely based ONLY on the provided evidence.\n"
)
SHORT_ANSWER = "Answer with as few words as possible. Give only the answer, no explanation."


def _instructions(row: dict) -> str | None:
    original = row.get("original_data", {})
    found = original.get("additional_instructions")
    if not found and original.get("metadata", {}).get("dataset") in {"nq", "nq_tables"}:
        found = SHORT_ANSWER
    return found


def tile_path(hit: dict, tile_cache: Path) -> Path:
    """Local path of a released screenshot tile coordinate."""
    return (
        tile_cache
        / f"{int(hit['article_id'])}.png.tiles"
        / f"chunk_{int(hit['tile_index']):04d}_{int(hit['chunk_index']):02d}.png"
    )


def text_messages(row: dict, top_k: int) -> list[dict]:
    evidence = "\n\n".join(hit.get("text") or "" for hit in row["hits"][:top_k])
    user = f"{evidence}\n\nQuestion: {row['problem']}"
    if extra := _instructions(row):
        user += f"\n\n{extra}"
    return [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": user}]


def screenshot_messages(row: dict, top_k: int, tile_cache: Path) -> list[dict]:
    content: list[dict] = [{"type": "text", "text": row["problem"]}]
    for hit in row["hits"][:top_k]:
        if hit.get("modality") == "text":
            content.append({"type": "text", "text": hit["text"]})
            continue
        path = Path(hit["image_path"]) if hit.get("image_path") else tile_path(hit, tile_cache)
        if not path.is_file():
            continue  # a tile that could not be rendered is skipped, not replaced
        encoded = base64.b64encode(path.read_bytes()).decode()
        content.append({"type": "image_url", "image_url": {"url": f"data:image/png;base64,{encoded}"}})
    if extra := _instructions(row):
        content.append({"type": "text", "text": extra})
    return [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": content}]


async def _read(args, client, semaphore, row: dict) -> dict:
    if args.mode == "text":
        messages = text_messages(row, args.top_k)
    else:
        messages = screenshot_messages(row, args.top_k, args.tile_cache)
    keep = {k: row[k] for k in ("example_id", "load_index", "problem", "original_data")}
    try:
        async with semaphore:
            response = await client.chat.completions.create(
                model=args.model,
                messages=messages,
                temperature=0,
                max_tokens=args.max_tokens,
                extra_body={"chat_template_kwargs": {"enable_thinking": False}},
            )
        return {
            **keep,
            "model": args.model,
            "final_response": response.choices[0].message.content,
            "usage": response.usage.model_dump() if response.usage else {},
            "success": True,
        }
    except Exception as error:  # noqa: BLE001 - recorded per row and retried on rerun
        return {**keep, "success": False, "error": f"{type(error).__name__}: {error}"}


async def run(args) -> dict:
    from openai import AsyncOpenAI

    done = (
        {r["load_index"] for r in read_jsonl(args.output) if r.get("success")}
        if args.output.exists()
        else set()
    )
    rows = [r for r in read_jsonl(args.candidates) if r["load_index"] not in done]
    client = AsyncOpenAI(api_key="EMPTY", base_url=args.api_base, timeout=600, max_retries=2)
    semaphore = asyncio.Semaphore(args.concurrency)
    tasks = [asyncio.create_task(_read(args, client, semaphore, row)) for row in rows]
    failed = 0
    with args.output.open("a") as stream:
        for task in asyncio.as_completed(tasks):
            result = await task
            failed += not result["success"]
            stream.write(json.dumps(result, ensure_ascii=False) + "\n")
    await client.close()
    # Keep one successful row per question, in candidate order.
    final = {r["load_index"]: r for r in read_jsonl(args.output) if r.get("success")}
    order = [r["load_index"] for r in read_jsonl(args.candidates)]
    with args.output.open("w") as stream:
        for index in order:
            if index in final:
                stream.write(json.dumps(final[index], ensure_ascii=False) + "\n")
    return {"answered": len(final), "questions": len(order), "failed_this_run": failed}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("candidates", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--model", default="Qwen/Qwen3.5-4B")
    parser.add_argument("--api-base", default="http://127.0.0.1:8010/v1")
    parser.add_argument("--mode", choices=("text", "screenshot"), default="text")
    parser.add_argument("--tile-cache", type=Path, default=Path("data/pixelrag/tiles"))
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--max-tokens", type=int, default=200)
    parser.add_argument("--concurrency", type=int, default=48)
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    print(json.dumps(asyncio.run(run(args)), indent=2))


if __name__ == "__main__":
    main()
