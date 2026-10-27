#!/usr/bin/env bash
# Build the hybrid text pipeline's index from the 2025-08 Wikipedia ZIM:
# 15.5M chunks of 1,024 tokens, a Qwen3-Embedding-0.6B IVF-Flat index, and a BM25 index.
# Measured: extraction about 2 hours with 32 CPU workers; embedding 53 A100 GPU-hours
# (under 8 hours on 8 GPUs). Outputs: chunks 14 GB, IVF index 60 GB, BM25 index 5 GB.
# Usage: GPUS="0 1 2 3 4 5 6 7" bash scripts/build_hybrid_index.sh
set -euo pipefail
cd "$(dirname "$0")/.."
read -r -a GPUS <<<"${GPUS:-0}"

python -m tessera.corpus build --workers "${WORKERS:-64}"
python -m tessera.corpus finalize

for i in "${!GPUS[@]}"; do
    CUDA_VISIBLE_DEVICES="${GPUS[$i]}" python -m tessera.dense embed --part "$i" --parts "${#GPUS[@]}" &
done
wait  # vLLM may abort at teardown after writing; count the shards instead of exit codes
n_chunks=$(ls data/hybrid/chunks/part-*.parquet | wc -l)
n_vectors=$(ls data/hybrid/embeddings/part-*.npy 2>/dev/null | wc -l)
[[ "$n_vectors" -eq "$n_chunks" ]] || { echo "embedded $n_vectors/$n_chunks shards; rerun to resume" >&2; exit 1; }
CUDA_VISIBLE_DEVICES="${GPUS[0]}" python -m tessera.dense index

python -m tessera.bm25 export
python -m tessera.bm25 index --heap-gb 128 --threads 64
