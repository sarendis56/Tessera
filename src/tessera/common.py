"""Shared JSONL, ranking, and scoring helpers.

A *row* is one question: `example_id`, `load_index` (unique within a dataset),
`problem`, `original_data` (gold answers and metadata), and `hits`, the ranked
evidence items. A text hit carries `title`, `text`, and, after reranking,
`rerank_score`.
"""

from __future__ import annotations

import json
import math
import re
import unicodedata
from pathlib import Path

DATASETS = ("nq", "nq_tables", "simpleqa")
STOPWORDS = set(
    "a an the of in on at to for from by with and or is are was were be been who "
    "what when where which how many much did do does name first last".split()
)


def read_jsonl(path: Path | str) -> list[dict]:
    return [json.loads(line) for line in Path(path).open() if line.strip()]


def write_jsonl(path: Path | str, rows) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.tmp")
    with tmp.open("w") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")
    tmp.replace(path)


def reranked(hits: list[dict], depth: int) -> list[dict]:
    """The top `depth` hits by reranker score."""
    admitted = [hit for hit in hits if hit.get("rerank_score") is not None]
    return sorted(admitted, key=lambda hit: -hit["rerank_score"])[:depth]


def content_words(text: str) -> set[str]:
    """Case- and punctuation-insensitive words minus question stopwords."""
    text = unicodedata.normalize("NFKC", text).casefold()
    return {w for w in re.sub(r"[^\w]+", " ", text).split() if w not in STOPWORDS}


def tokens(text: str, tokenizer) -> int:
    return len(tokenizer.encode(text, add_special_tokens=False))


def bge_scorer(model_path: str, device: str = "cuda", max_length: int = 512):
    """Cross-encoder scores for (query, passage) pairs with BGE-reranker-v2-m3."""
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = (
        AutoModelForSequenceClassification.from_pretrained(model_path, dtype=torch.bfloat16).to(device).eval()
    )

    def score(pairs: list[tuple[str, str]]) -> list[float]:
        inputs = tokenizer(pairs, padding=True, truncation=True, max_length=max_length, return_tensors="pt")
        inputs = {key: value.to(device) for key, value in inputs.items()}
        with torch.inference_mode():
            logits = model(**inputs, return_dict=True).logits.view(-1).float()
        return logits.cpu().tolist()

    return score


def score_batched(scorer, pairs: list[tuple[str, str]], batch_size: int) -> list[float]:
    scores: list[float] = []
    for start in range(0, len(pairs), batch_size):
        scores.extend(scorer(pairs[start : start + batch_size]))
    return scores


def mcnemar_p(left_only: int, right_only: int) -> float:
    """Two-sided exact McNemar p-value from the discordant counts."""
    discordant = left_only + right_only
    if discordant == 0:
        return 1.0
    lower = min(left_only, right_only)
    tail = sum(math.comb(discordant, value) for value in range(lower + 1))
    return min(1.0, 2 * tail / 2**discordant)
