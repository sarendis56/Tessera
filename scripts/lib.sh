# Shared helpers for the run scripts. Source from the repository root.
DATASETS=(nq nq_tables simpleqa)
READER="${READER:-models/Qwen/Qwen3.5-4B}"            # or models/google/gemma-4-31B-it
READER_NAME="${READER_NAME:-Qwen/Qwen3.5-4B}"
READER_GPU="${READER_GPU:-0}"                         # e.g. "0,1" with READER_ARGS="--tensor-parallel-size 2"
READER_PORT="${READER_PORT:-8010}"
MAX_TOKENS="${MAX_TOKENS:-200}"                        # 2048 for Gemma 4 E4B
GRADER="${GRADER:-gpt-5.6-luna}"
R="${READER_NAME##*/}"                                 # tags response files by reader

start_reader() {
    CUDA_VISIBLE_DEVICES="$READER_GPU" vllm serve "$READER" --served-model-name "$READER_NAME" \
        --port "$READER_PORT" --max-model-len 32768 --dtype bfloat16 --gpu-memory-utilization 0.8 ${READER_ARGS:-} \
        >"${RUN:-runs}/reader.log" 2>&1 &
    READER_PID=$!
    trap 'kill $READER_PID 2>/dev/null || true' EXIT
    until curl -fs "http://127.0.0.1:$READER_PORT/v1/models" >/dev/null; do
        kill -0 "$READER_PID" || { echo "reader failed; see ${RUN:-runs}/reader.log" >&2; exit 1; }
        sleep 5
    done
}

# read_and_grade CANDIDATES RESPONSES [extra tessera.read args...]
read_and_grade() {
    local candidates="$1" responses="$2"
    shift 2
    python -m tessera.read "$candidates" "$responses" --model "$READER_NAME" \
        --api-base "http://127.0.0.1:$READER_PORT/v1" --max-tokens "$MAX_TOKENS" "$@"
    python -m tessera.grade judge "$responses" "${responses%.jsonl}_graded.jsonl" --model "$GRADER"
}

# compare NAME LEFT RIGHT: pooled McNemar over the three datasets of $RUN/<dataset>/
compare() {
    local name="$1" left="$2" right="$3" pairs=()
    for d in "${DATASETS[@]}"; do
        pairs+=(--pair "$RUN/$d/${left}_responses_graded.jsonl" "$RUN/$d/${right}_responses_graded.jsonl")
    done
    python -m tessera.grade compare "${pairs[@]}" | tee "$RUN/compare_${name}.json"
}
