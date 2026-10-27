"""Question populations used in the paper.

    python -m tessera.data --raw data/raw --out data/populations

Single-hop (7,851 questions):
  nq.jsonl         3,336 Natural Questions validation questions with short answers
                   from at least two annotators (PixelRAG's released filter)
  nq_tables.jsonl  all 1,068 NQ-Tables development questions
  simpleqa.jsonl   3,447 SimpleQA questions citing a non-category English
                   Wikipedia article
Multi-hop (agentic search):
  frames.jsonl     all 824 FRAMES test questions
  musique.jsonl    1,000 answerable MuSiQue dev questions (seed 20260930)

Every row has `example_id`, `load_index` (position in the file), `problem`, and
`original_data` with `gold_answers` and `metadata.dataset`/`metadata.urls`.
"""

from __future__ import annotations

import argparse
import ast
import csv
import hashlib
import html
import json
import random
import re
import urllib.parse
from pathlib import Path

from tessera.common import write_jsonl

SHORT_ANSWER = "Answer with as few words as possible. Give only the answer, no explanation."
EXPECTED = {"nq": 3336, "nq_tables": 1068, "simpleqa": 3447, "frames": 824, "musique": 1000}


def _row(index: int, example_id: str, problem: str, original: dict) -> dict:
    return {
        "schema_version": 1,
        "example_id": example_id,
        "load_index": index,
        "problem": problem,
        "original_data": {"id": example_id, "problem": problem, **original},
    }


def _wiki_url(document_url: str) -> str:
    document_url = html.unescape(document_url)
    if found := re.search(r"[?&]title=([^&]+)", document_url):
        title = urllib.parse.quote(found.group(1), safe="/:(),-")
        return f"https://en.wikipedia.org/wiki/{title}"
    return document_url


def natural_questions(parquet_dir: Path) -> list[dict]:
    from datasets import load_dataset

    files = sorted(str(p) for p in parquet_dir.glob("validation-*.parquet"))
    rows = []
    for record in load_dataset("parquet", data_files=files, split="train", streaming=True):
        annotations = record["annotations"]
        answers: set[str] = set()
        annotators = 0
        for index in range(len(annotations["id"])):
            texts = annotations["short_answers"][index].get("text", [])
            if texts:
                annotators += 1
                answers.update(t.strip() for t in texts if t.strip())
        if annotators < 2 or not answers:
            continue
        question = record["question"]["text"]
        metadata = {
            "urls": [_wiki_url(record["document"]["url"])],
            "dataset": "nq",
            "document_title": record["document"]["title"],
        }
        example_id = hashlib.md5(question.encode()).hexdigest()
        original = {"gold_answers": sorted(answers), "metadata": metadata}
        rows.append(_row(len(rows), example_id, question, original))
    return rows


def nq_tables(path: Path) -> list[dict]:
    rows = []
    for record in map(json.loads, path.open()):
        if not record.get("questions"):
            continue
        question = record["questions"][0]
        problem = str(question.get("originalText") or "").strip()
        answers = [
            str(a).strip() for a in question.get("answer", {}).get("answerTexts", []) if str(a).strip()
        ]
        if not problem or not answers:
            continue
        table = record.get("table", {})
        url = _wiki_url(str(table.get("documentUrl") or ""))
        example_id = str(record.get("id") or hashlib.md5(problem.encode()).hexdigest())
        original = {
            "gold_answers": answers,
            "metadata": {
                "urls": [url] if url else [],
                "dataset": "nq_tables",
                "document_title": table.get("documentTitle", ""),
                "table_id": table.get("tableId", ""),
            },
            "additional_instructions": SHORT_ANSWER,
        }
        rows.append(_row(len(rows), example_id, problem, original))
    return rows


def simpleqa(path: Path) -> list[dict]:
    rows = []
    with path.open(newline="") as stream:
        for record in csv.DictReader(stream):
            problem, answer = record["problem"].strip(), record["answer"].strip()
            metadata = ast.literal_eval(record["metadata"])
            urls = [str(u) for u in metadata.get("urls", [])]
            parts = [p.strip() for u in urls for p in u.splitlines() if p.strip()]
            wiki = [u for u in parts if "en.wikipedia.org/wiki/" in u and "/Category:" not in u]
            if not wiki:
                continue
            original = {
                "gold_answers": [answer],
                "answer": answer,
                "metadata": {**metadata, "source_urls": urls, "urls": wiki, "dataset": "simpleqa"},
            }
            rows.append(_row(len(rows), hashlib.md5(problem.encode()).hexdigest(), problem, original))
    return rows


def _multihop(dataset: str, index: int, problem: str, gold: list[str], metadata: dict) -> dict:
    example_id = hashlib.md5(f"{dataset}:{problem}".encode()).hexdigest()
    original = {"gold_answers": gold, "metadata": {"dataset": dataset, **metadata}}
    return _row(index, example_id, problem, original)


def frames(path: Path) -> list[dict]:
    with path.open(newline="") as stream:
        records = list(csv.DictReader(stream, delimiter="\t"))
    return [
        _multihop(
            "frames",
            i,
            r["Prompt"].strip(),
            [r["Answer"].strip()],
            {"urls": ast.literal_eval(r["wiki_links"]), "reasoning_types": r["reasoning_types"]},
        )
        for i, r in enumerate(records)
    ]


def musique(path: Path, sample: int = 1000, seed: int = 20260930) -> list[dict]:
    records = [r for r in map(json.loads, path.open()) if r.get("answerable", True)]
    keep = sorted(random.Random(seed).sample(range(len(records)), min(sample, len(records))))
    rows = []
    for i, k in enumerate(keep):
        r = records[k]
        titles = [p["title"] for p in r["paragraphs"] if p["is_supporting"]]
        urls = ["https://en.wikipedia.org/wiki/" + t.replace(" ", "_") for t in titles]
        metadata = {"urls": urls, "source_id": r["id"], "hops": int(r["id"][0])}
        gold = [r["answer"], *r.get("answer_aliases", [])]
        rows.append(_multihop("musique", i, r["question"].strip(), gold, metadata))
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--raw", type=Path, default=Path("data/raw"))
    parser.add_argument("--out", type=Path, default=Path("data/populations"))
    args = parser.parse_args()
    raw = args.raw
    builders = {
        "nq": lambda: natural_questions(raw / "natural_questions/default"),
        "nq_tables": lambda: nq_tables(raw / "nq_tables/dev.jsonl"),
        "simpleqa": lambda: simpleqa(raw / "simpleqa/simple_qa_test_set.csv"),
        "frames": lambda: frames(raw / "frames/test.tsv"),
        "musique": lambda: musique(raw / "musique/musique_ans_v1.0_dev.jsonl"),
    }
    for name, build in builders.items():
        rows = build()
        if len(rows) != EXPECTED[name]:
            raise SystemExit(f"{name}: {len(rows)} questions, expected {EXPECTED[name]}")
        write_jsonl(args.out / f"{name}.jsonl", rows)
        print(name, len(rows))


if __name__ == "__main__":
    main()
