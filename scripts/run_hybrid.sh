#!/usr/bin/env bash
# Table 3, row "Hybrid text + BGE": top-3 chunks vs. Tessera (top-2 chunks + unit block)
# on NQ, NQ-Tables, and SimpleQA. Needs the hybrid index (scripts/build_hybrid_index.sh)
# and OPENAI_API_KEY for grading.
# Usage: GPU=0 bash scripts/run_hybrid.sh
set -euo pipefail
cd "$(dirname "$0")/.."
GPU="${GPU:-0}"
RUN="${RUN:-runs/hybrid}"
READER_GPU="$GPU"
source scripts/lib.sh
mkdir -p "$RUN"

for d in "${DATASETS[@]}"; do
    out="$RUN/$d"
    CUDA_VISIBLE_DEVICES="$GPU" python -m tessera.retrieve --examples "data/populations/$d.jsonl" --out "$out"
    [[ -f "$out/pools.jsonl" ]] || python -m tessera.layer pools --candidates "$out/candidates.jsonl" --out "$out"
    ls "$out"/scored.part*.jsonl >/dev/null 2>&1 || CUDA_VISIBLE_DEVICES="$GPU" python -m tessera.layer score --out "$out"
    python -m tessera.layer build --candidates "$out/candidates.jsonl" --out "$out"
done

start_reader
for d in "${DATASETS[@]}"; do
    for system in baseline tessera; do
        read_and_grade "$RUN/$d/${system}_candidates.jsonl" "$RUN/$d/${system}_${R}_responses.jsonl"
    done
done
compare "tessera_$R" "baseline_$R" "tessera_$R"
