"""Grade reader answers with the SimpleQA template and compare two systems.

    python -m tessera.grade judge responses.jsonl graded.jsonl --model gpt-5.6-luna
    python -m tessera.grade compare --pair nq_base.jsonl nq_tessera.jsonl \
        --pair nqt_base.jsonl nqt_tessera.jsonl ...

`judge` sends each answer, its question, and the gold answers (up to ten
aliases joined by " OR ") to an OpenAI chat model with the SimpleQA grading
template; reasoning before `</think>` is dropped first. `OPENAI_API_KEY` must be
set. Accuracy is the share of answers graded CORRECT.

`compare` pairs two graded files by `load_index` and reports accuracy, the
gain, discordant counts, and the exact McNemar p-value, per pair and pooled
(summing the paired counts of all pairs).
"""

from __future__ import annotations

import argparse
import asyncio
import json
import re
from pathlib import Path

from tessera.common import mcnemar_p, read_jsonl

TEMPLATE = (Path(__file__).parent / "simpleqa_grader_template.txt").read_text()
MARKER = "Here is a new example."
SYSTEM = "You are a helpful assistant.\n\n" + TEMPLATE[: TEMPLATE.find(MARKER)].strip()
USER = TEMPLATE[TEMPLATE.find(MARKER) :]
LABELS = {"A": "correct", "B": "incorrect", "C": "not_attempted"}


def strip_think(text: str) -> str:
    if "<think>" in text and "</think>" in text:
        return text.split("</think>")[-1].strip()
    if "think>" in text:
        return text.split("think>")[-1].strip()
    return text


def gold_target(original: dict) -> str:
    aliases = original.get("gold_answers") or original.get("answer") or []
    aliases = aliases if isinstance(aliases, list) else [aliases]
    return " OR ".join(str(a) for a in aliases[:10])


def messages(row: dict) -> list[dict]:
    user = USER.format(
        question=row["problem"],
        target=gold_target(row["original_data"]),
        predicted_answer=strip_think(str(row.get("final_response") or "")),
    )
    return [{"role": "system", "content": SYSTEM}, {"role": "user", "content": user}]


def parse_verdict(text: str) -> str | None:
    found = re.fullmatch(r"\s*([ABC])\s*", text.upper()) or re.search(r"\b([ABC])\b", text.upper())
    return LABELS.get(found.group(1)) if found else None


async def _judge(client, semaphore, model: str, row: dict) -> dict:
    sampler = (
        {"reasoning_effort": "none", "max_completion_tokens": 1000}
        if model.startswith("gpt-5")
        else {"temperature": 0, "max_tokens": 1000, "seed": 42}
    )
    async with semaphore:
        for attempt in range(5):
            try:
                response = await client.chat.completions.create(
                    model=model, messages=messages(row), **sampler
                )
                raw = response.choices[0].message.content or ""
                return {**row, "verdict": parse_verdict(raw), "grader_raw": raw}
            except Exception as error:  # noqa: BLE001 - retried, then recorded
                if attempt == 4:
                    return {**row, "verdict": None, "grader_error": repr(error)}
                await asyncio.sleep(2**attempt)


async def judge(args) -> dict:
    from openai import AsyncOpenAI

    rows = read_jsonl(args.responses)
    client = AsyncOpenAI()
    semaphore = asyncio.Semaphore(args.concurrency)
    graded = await asyncio.gather(*(_judge(client, semaphore, args.model, r) for r in rows))
    await client.close()
    with args.output.open("w") as stream:
        for row in graded:
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")
    ungraded = sum(r["verdict"] is None for r in graded)
    correct = sum(r["verdict"] == "correct" for r in graded)
    return {"n": len(graded), "accuracy": correct / max(len(graded), 1), "ungraded": ungraded}


def _outcomes(path: Path) -> dict[int, bool]:
    rows = read_jsonl(path)
    if missing := [r["load_index"] for r in rows if r.get("verdict") is None]:
        raise ValueError(f"{path}: {len(missing)} ungraded rows, e.g. {missing[:5]}")
    return {r["load_index"]: r["verdict"] == "correct" for r in rows}


def _paired(left: dict[int, bool], right: dict[int, bool]) -> dict:
    if left.keys() != right.keys():
        raise ValueError("the two files cover different questions")
    gains = sum(not left[k] and right[k] for k in left)
    losses = sum(left[k] and not right[k] for k in left)
    n = len(left)
    return {
        "n": n,
        "left_accuracy": 100 * sum(left.values()) / n,
        "right_accuracy": 100 * sum(right.values()) / n,
        "gain_points": 100 * (gains - losses) / n,
        "gains": gains,
        "losses": losses,
        "mcnemar_p": mcnemar_p(gains, losses),
    }


def compare(args) -> dict:
    results, pooled_left, pooled_right = [], {}, {}
    for i, (left_path, right_path) in enumerate(args.pair):
        left, right = _outcomes(Path(left_path)), _outcomes(Path(right_path))
        results.append({"left": left_path, "right": right_path, **_paired(left, right)})
        pooled_left |= {(i, k): v for k, v in left.items()}
        pooled_right |= {(i, k): v for k, v in right.items()}
    return {"pairs": results, "pooled": _paired(pooled_left, pooled_right)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = parser.add_subparsers(dest="action", required=True)
    j = sub.add_parser("judge")
    j.add_argument("responses", type=Path)
    j.add_argument("output", type=Path)
    j.add_argument("--model", default="gpt-5.6-luna")
    j.add_argument("--concurrency", type=int, default=64)
    c = sub.add_parser("compare")
    c.add_argument("--pair", nargs=2, action="append", required=True, metavar=("LEFT", "RIGHT"))
    args = parser.parse_args()
    result = asyncio.run(judge(args)) if args.action == "judge" else compare(args)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
