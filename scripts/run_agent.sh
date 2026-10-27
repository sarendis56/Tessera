#!/usr/bin/env bash
# Table 5 (agentic search): Qwen3.5-4B with thinking and up to eight searches on FRAMES and
# MuSiQue, with chunks, Tessera, Docling, or a fourth chunk in every tool result.
# Needs the hybrid index. The agent runs on AGENT_GPU, retrieval on SEARCH_GPU.
# Usage: AGENT_GPU=0 SEARCH_GPU=1 bash scripts/run_agent.sh chunks tessera
set -euo pipefail
cd "$(dirname "$0")/.."
ARMS=("$@")
[[ ${#ARMS[@]} -gt 0 ]] || ARMS=(chunks tessera)
RUN="${RUN:-runs/agent}"
mkdir -p "$RUN"

for arm in "${ARMS[@]}"; do
    CUDA_VISIBLE_DEVICES="${AGENT_GPU:-0}" python -m tessera.agent run --arm "$arm" --out "$RUN" \
        --search-gpu "${SEARCH_GPU:-1}" || true  # vLLM may abort at teardown; `responses` checks completion
    python -m tessera.agent responses --arm "$arm" --out "$RUN"
    for d in frames musique; do
        python -m tessera.grade judge "$RUN/${d}_${arm}_responses.jsonl" \
            "$RUN/${d}_${arm}_responses_graded.jsonl" --model "${GRADER:-gpt-5.6-luna}"
    done
done
if [[ " ${ARMS[*]} " == *" chunks "* && " ${ARMS[*]} " == *" tessera "* ]]; then
    python -m tessera.grade compare \
        --pair "$RUN/frames_chunks_responses_graded.jsonl" "$RUN/frames_tessera_responses_graded.jsonl" \
        --pair "$RUN/musique_chunks_responses_graded.jsonl" "$RUN/musique_tessera_responses_graded.jsonl"
fi
