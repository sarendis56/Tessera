"""Multi-step search agent with Tessera on every search call (paper §4.5).

Qwen3.5-4B (thinking on) answers FRAMES and MuSiQue questions with one
`search(query)` tool over the hybrid text pipeline, up to eight calls; after the
last call it is told to answer without tools. Every tool result is three
documents in Search-R1's "Doc i(Title: t) text" format, and the arms differ only
in the last one:

  chunks    the top-3 reranked chunks
  tessera   the top-2 chunks + a unit block for the agent's query, packed to the
            third chunk's token length
  docling   the top-2 chunks + Docling chunks of the same anchor pages
  chunks4   the top-4 chunks

    python -m tessera.agent run --arm tessera --out runs/agent --search-gpu 1
    python -m tessera.agent responses --arm tessera --out runs/agent

Each round sends all pending queries through `tessera.retrieve` and
`tessera.agent docs` in subprocesses on `--search-gpu`; the agent's vLLM engine
stays on the current GPU. State is saved after every round, so a stopped run
resumes. `responses` writes `{dataset}_{arm}_responses.jsonl` for `tessera.grade`.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
from pathlib import Path

from tessera.common import read_jsonl, reranked, score_batched, tokens, write_jsonl

TOOL_SYSTEM = (
    "Answer the question. Use the search tool to look up information in "
    "Wikipedia (August 2025) when you need it. When you have enough information, "
    "give the final answer inside <answer> and </answer>, as a short phrase "
    "without explanation."
)
FINAL_TURN = (
    "You have reached the search limit. Based on the information above, give "
    "your final answer now inside <answer> and </answer>."
)
SEARCH_TOOL = {
    "type": "function",
    "function": {
        "name": "search",
        "description": "Search Wikipedia (August 2025). Returns the top 3 documents.",
        "parameters": {
            "type": "object",
            "properties": {"query": {"type": "string", "description": "The query."}},
            "required": ["query"],
        },
    },
}
TOOL_CALL_RE = re.compile(
    r"<tool_call>\s*<function=search>\s*<parameter=query>\s*(.*?)\s*</parameter>", re.DOTALL
)
THINK_RE = re.compile(r"<think>.*?</think>", re.DOTALL)
ANSWER_RE = re.compile(
    r"(?:<answer>|<function=answer>\s*(?:<parameter=\w+>)?)\s*(.*?)\s*"
    r"(?:</answer>|</parameter>|</function>|$)",
    re.DOTALL,
)
SAMPLING = {"temperature": 1.0, "top_p": 0.95, "top_k": 20, "min_p": 0.0, "presence_penalty": 1.5}
DATASETS = ("frames", "musique")


def format_docs(docs: list[dict]) -> str:
    return "".join(f"Doc {i + 1}(Title: {d['title']}) {d['text']}\n" for i, d in enumerate(docs))


def docs(args) -> dict:
    """The arm's three (or four) documents for every query of one round."""
    from transformers import AutoTokenizer

    from tessera.common import bge_scorer
    from tessera.layer import anchor_pages, pack, pack_flat, prefilter, unit_pages

    rows = read_jsonl(args.round_dir / "candidates.jsonl")
    tops = [reranked(r["hits"], 4 if args.arm == "chunks4" else 3) for r in rows]
    out = [[{"title": h["title"], "text": h["text"]} for h in top] for top in tops]
    if args.arm in ("tessera", "docling"):
        tokenizer = AutoTokenizer.from_pretrained(args.model)
        anchors = [anchor_pages(r["hits"]) for r in rows]
        builder = "tessera" if args.arm == "tessera" else "docling"
        pages = unit_pages(args.zim, [t for a in anchors for t in a], args.workers, builder)
        pools = [
            prefilter(r["problem"], [u for t in a for u in pages[t]])
            for r, a in zip(rows, anchors, strict=True)
        ]
        pairs = [(r["problem"], u["completed"]) for r, p in zip(rows, pools, strict=True) for u in p]
        scores = iter(score_batched(bge_scorer(args.reranker), pairs, 64))
        for pool, top, documents in zip(pools, tops, out, strict=True):
            for unit in pool:
                unit["score"] = next(scores)
            head = top[:2] if len(top) >= 3 else top
            budget = sum(tokens(h["text"], tokenizer) for h in top[len(head) :])
            text = (pack if args.arm == "tessera" else pack_flat)(pool, budget, tokenizer)
            documents[:] = [*documents[: len(head)], {"title": "units", "text": text}]
    write_jsonl(args.round_dir / f"docs_{args.arm}.jsonl", [{"docs": d} for d in out])
    return {"queries": len(out)}


def _search(args, round_dir: Path, queries: list[str]) -> list[list[dict]]:
    """Retrieval and the arm's documents for one round, in subprocesses on the search GPU."""
    target = round_dir / f"docs_{args.arm}.jsonl"
    if not target.exists():
        examples = [{"example_id": str(i), "load_index": i, "problem": q} for i, q in enumerate(queries)]
        write_jsonl(round_dir / "examples.jsonl", examples)
        env = {**os.environ, "CUDA_VISIBLE_DEVICES": args.search_gpu}
        python = [sys.executable, "-m"]
        subprocess.run(
            [
                *python,
                "tessera.retrieve",
                "--examples",
                str(round_dir / "examples.jsonl"),
                "--out",
                str(round_dir),
                "--gpu-memory-utilization",
                "0.2",
            ],
            check=True,
            env=env,
        )
        subprocess.run(
            [
                *python,
                "tessera.agent",
                "docs",
                "--arm",
                args.arm,
                "--out",
                str(args.out),
                "--round-dir",
                str(round_dir),
            ],
            check=True,
            env=env,
        )
    return [r["docs"] for r in read_jsonl(target)]


def run(args) -> dict:
    from vllm import LLM, SamplingParams

    arm_dir = args.out / args.arm
    arm_dir.mkdir(parents=True, exist_ok=True)
    state_path = arm_dir / "state.json"
    if state_path.exists():
        state = json.loads(state_path.read_text())
    else:
        tasks = [
            {
                "dataset": d,
                "load_index": r["load_index"],
                "searches": [],
                "answer": None,
                "status": "active",
                "messages": [
                    {"role": "system", "content": TOOL_SYSTEM},
                    {"role": "user", "content": r["problem"].strip()},
                ],
            }
            for d in DATASETS
            for r in read_jsonl(args.populations / f"{d}.jsonl")
        ]
        state = {"round": 0, "tasks": tasks}
    llm = LLM(
        model=args.model,
        dtype="bfloat16",
        max_model_len=131072,
        gpu_memory_utilization=0.3,
        limit_mm_per_prompt={"image": 0, "video": 0},
        seed=0,
    )
    sampling = SamplingParams(**SAMPLING, max_tokens=32768, seed=0)
    kwargs = {"enable_thinking": True}
    tasks = state["tasks"]
    while state["round"] <= args.max_searches:
        active = [t for t in tasks if t["status"] == "active"]
        if not active:
            break
        last = state["round"] == args.max_searches
        if last:
            for task in active:
                task["messages"].append({"role": "user", "content": FINAL_TURN})
        outputs = llm.chat(
            [t["messages"] for t in active],
            sampling,
            tools=None if last else [SEARCH_TOOL],
            chat_template_kwargs=kwargs,
            use_tqdm=False,
        )
        pending = []
        for task, output in zip(active, outputs, strict=True):
            reasoning, _, text = output.outputs[0].text.rpartition("</think>")
            reasoning = THINK_RE.sub("", reasoning).replace("<think>", "").strip()
            text = text.strip()
            call = TOOL_CALL_RE.search(text)
            if call and not last:
                pending.append((task, text[: call.start()].strip(), call.group(1).strip(), reasoning))
                continue
            task["messages"].append({"role": "assistant", "content": text, "reasoning_content": reasoning})
            answer = ANSWER_RE.search(text)
            task["answer"] = answer.group(1).strip() if answer else text
            task["status"] = "answered" if answer else "untagged"
        if pending:
            found = _search(args, arm_dir / f"round_{state['round']}", [p[2] for p in pending])
            for (task, content, query, reasoning), documents in zip(pending, found, strict=True):
                call_id = f"call_{len(task['searches'])}"
                task["messages"].append(
                    {
                        "role": "assistant",
                        "content": content,
                        "reasoning_content": reasoning,
                        "tool_calls": [
                            {
                                "id": call_id,
                                "type": "function",
                                "function": {"name": "search", "arguments": {"query": query}},
                            }
                        ],
                    }
                )
                task["messages"].append(
                    {"role": "tool", "tool_call_id": call_id, "content": format_docs(documents).strip()}
                )
                task["searches"].append({"query": query, "titles": [d["title"] for d in documents]})
        state["round"] += 1
        tmp = state_path.with_suffix(".tmp")
        tmp.write_text(json.dumps(state))
        tmp.replace(state_path)
        print(json.dumps({"round": state["round"], "active": len(active)}), flush=True)
    return {"arm": args.arm, "tasks": len(tasks), "rounds": state["round"]}


def responses(args) -> dict:
    """Reader-style response rows per dataset for grading."""
    tasks = json.loads((args.out / args.arm / "state.json").read_text())["tasks"]
    if unfinished := sum(t["status"] == "active" for t in tasks):
        raise SystemExit(f"{unfinished} questions are unfinished; rerun `run` to resume")
    by_key = {(t["dataset"], t["load_index"]): t for t in tasks}
    summary = {}
    for dataset in DATASETS:
        rows = []
        for row in read_jsonl(args.populations / f"{dataset}.jsonl"):
            task = by_key[(dataset, row["load_index"])]
            answer = task["answer"] or ""
            if task["status"] == "untagged" and (found := ANSWER_RE.search(task["messages"][-1]["content"])):
                answer = found.group(1).strip() or answer
            rows.append(
                {
                    **row,
                    "final_response": answer,
                    "agent_status": task["status"],
                    "searches": task["searches"],
                }
            )
        write_jsonl(args.out / f"{dataset}_{args.arm}_responses.jsonl", rows)
        summary[dataset] = {
            "n": len(rows),
            "mean_searches": sum(len(r["searches"]) for r in rows) / len(rows),
        }
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("action", choices=("run", "docs", "responses"))
    parser.add_argument("--arm", choices=("chunks", "tessera", "docling", "chunks4"), required=True)
    parser.add_argument("--out", type=Path, default=Path("runs/agent"))
    parser.add_argument("--populations", type=Path, default=Path("data/populations"))
    parser.add_argument("--model", default="models/Qwen/Qwen3.5-4B")
    parser.add_argument("--max-searches", type=int, default=8)
    parser.add_argument("--search-gpu", default="1")
    parser.add_argument("--round-dir", type=Path)
    parser.add_argument("--zim", default="data/wikipedia/wikipedia_en_all_maxi_2025-08.zim")
    parser.add_argument("--reranker", default="models/BAAI/bge-reranker-v2-m3")
    parser.add_argument("--workers", type=int, default=48)
    args = parser.parse_args()
    print(json.dumps({"run": run, "docs": docs, "responses": responses}[args.action](args), indent=2))


if __name__ == "__main__":
    main()
