#!/usr/bin/env bash
# Table 4 (matched-budget baselines on the hybrid text pipeline): Docling and Unstructured
# as the unit builder, and HtmlRAG / HtmlRAG-CE in the third slot. Run scripts/run_hybrid.sh
# first; this script reuses its candidates and its top-3 and Tessera answers.
# Usage: GPU=0 HTMLRAG_GPUS="0 1" bash scripts/run_baselines.sh
set -euo pipefail
cd "$(dirname "$0")/.."
GPU="${GPU:-0}"
RUN="${RUN:-runs/hybrid}"
READER_GPU="$GPU"
read -r -a HTMLRAG_GPUS <<<"${HTMLRAG_GPUS:-$GPU}"
source scripts/lib.sh

for d in "${DATASETS[@]}"; do
    out="$RUN/$d"
    for builder in docling unstructured; do
        b="$out/$builder"
        [[ -f "$b/pools.jsonl" ]] || python -m tessera.layer pools --builder "$builder" \
            --candidates "$out/candidates.jsonl" --out "$b" --workers "${WORKERS:-64}"
        ls "$b"/scored.part*.jsonl >/dev/null 2>&1 || CUDA_VISIBLE_DEVICES="$GPU" python -m tessera.layer score --out "$b"
        python -m tessera.layer build --builder "$builder" --candidates "$out/candidates.jsonl" --out "$b"
        cp "$b/${builder}_candidates.jsonl" "$out/${builder}_candidates.jsonl"
    done
    for ranker in embed ce; do
        [[ -f "$out/htmlrag_${ranker}_candidates.jsonl" ]] || python -m tessera.baselines htmlrag \
            --ranker "$ranker" --candidates "$out/candidates.jsonl" --out "$out" --gpus "${HTMLRAG_GPUS[@]}"
    done
done

start_reader
for d in "${DATASETS[@]}"; do
    for system in docling unstructured htmlrag_embed htmlrag_ce; do
        read_and_grade "$RUN/$d/${system}_candidates.jsonl" "$RUN/$d/${system}_${R}_responses.jsonl"
    done
done
for system in docling unstructured htmlrag_embed htmlrag_ce; do
    compare "$system" "baseline_$R" "${system}_$R"
    compare "tessera_vs_$system" "${system}_$R" "tessera_$R"
done
