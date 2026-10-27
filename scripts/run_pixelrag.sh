#!/usr/bin/env bash
# Table 3, rows "PixelRAG text + BGE" and "PixelRAG screenshot + VL": the released PixelRAG
# indexes with their top-3 items vs. Tessera (top-2 items + unit block, same reader cost).
# Usage: GPU=0 bash scripts/run_pixelrag.sh text
#        GPU=0 bash scripts/run_pixelrag.sh screenshot   # needs third_party/PixelRAG (README)
set -euo pipefail
cd "$(dirname "$0")/.."
PIPELINE="${1:?usage: run_pixelrag.sh text|screenshot}"
GPU="${GPU:-0}"
RUN="${RUN:-runs/pixelrag_$PIPELINE}"
READER_GPU="$GPU"
source scripts/lib.sh
declare -A NPROBE=([nq]=128 [nq_tables]=128 [simpleqa]=2000)
MODE=text
[[ "$PIPELINE" == text ]] || MODE=screenshot
mkdir -p "$RUN"

for d in "${DATASETS[@]}"; do
    out="$RUN/$d"
    if [[ ! -f "$out/candidates.jsonl" ]]; then
        CUDA_VISIBLE_DEVICES="$GPU" python -m tessera.pixelrag search --pipeline "$PIPELINE" \
            --examples "data/populations/$d.jsonl" --out "$out" --nprobe "${NPROBE[$d]}"
        if [[ "$PIPELINE" == screenshot ]]; then
            python -m tessera.pixelrag render --out "$out"
            CUDA_VISIBLE_DEVICES="$GPU" python -m tessera.pixelrag rerank --out "$out"
        fi
    fi
    [[ -f "$out/pools.jsonl" ]] || python -m tessera.layer pools --candidates "$out/candidates.jsonl" --out "$out"
    ls "$out"/scored.part*.jsonl >/dev/null 2>&1 || CUDA_VISIBLE_DEVICES="$GPU" python -m tessera.layer score --out "$out"
    if [[ "$PIPELINE" == text ]]; then
        python -m tessera.layer build --candidates "$out/candidates.jsonl" --out "$out"
    else
        python -m tessera.pixelrag build --out "$out"
    fi
done

start_reader
for d in "${DATASETS[@]}"; do
    for system in baseline tessera; do
        read_and_grade "$RUN/$d/${system}_candidates.jsonl" "$RUN/$d/${system}_${R}_responses.jsonl" --mode "$MODE"
    done
done
compare "tessera_$R" "baseline_$R" "tessera_$R"
