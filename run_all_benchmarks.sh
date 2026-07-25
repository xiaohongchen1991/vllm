#!/usr/bin/env bash
set -uo pipefail

# Orchestrates the full benchmark workflow across models and linear backends
# without human intervention:
#   1. start vLLM server
#   2. wait for readiness, then clear L2 cache
#   3. run bench.sh
#   4. save results to disk
#   5. stop server + clean vLLM cache
#   6. repeat with/without --linear-backend helion, for every model

cd "$(dirname "$0")"
source .venv/bin/activate

# Each entry is "MODEL:TP" (tensor-parallel size). TP defaults to 1 if omitted.
MODELS=(
    # "RedHatAI/Qwen3-4B-FP8-dynamic:1"
    # "RedHatAI/Qwen3-14B-FP8-dynamic:1"
    # "RedHatAI/Meta-Llama-3.1-8B-Instruct-FP8-dynamic:1"
    # "RedHatAI/Mistral-Small-24B-Instruct-2501-FP8-dynamic:1"
    # "RedHatAI/Mistral-Small-24B-Instruct-2501-FP8-dynamic:2"
    # "RedHatAI/Qwen3-32B-FP8-dynamic:2"
    "RedHatAI/Meta-Llama-3.1-70B-Instruct-FP8-dynamic:2"
    # "RedHatAI/Qwen3-4B-Instruct-2507-quantized.w8a8:1"
    # "RedHatAI/DeepSeek-R1-Distill-Qwen-32B-quantized.w8a8:1"
    # "RedHatAI/DeepSeek-R1-Distill-Llama-8B-quantized.w8a8:1"
    # "RedHatAI/DeepSeek-R1-Distill-Qwen-32B-quantized.w8a8:2"
    # "RedHatAI/DeepSeek-R1-Distill-Llama-70B-quantized.w8a8:2"
)

PORT=8000
HOST=127.0.0.1
READY_TIMEOUT=1800   # seconds to wait for server to come up
RESULTS_DIR="benchmark_results"
mkdir -p "$RESULTS_DIR"

# slugify a model name for use in filenames
slug() { echo "$1" | tr '/:' '__'; }

wait_for_ready() {
    local deadline=$((SECONDS + READY_TIMEOUT))
    while (( SECONDS < deadline )); do
        if curl -sf "http://${HOST}:${PORT}/health" >/dev/null 2>&1; then
            return 0
        fi
        # bail early if the server process already died
        if ! kill -0 "$SERVER_PID" 2>/dev/null; then
            echo "!! server process $SERVER_PID exited before becoming ready"
            return 1
        fi
        sleep 5
    done
    echo "!! server did not become ready within ${READY_TIMEOUT}s"
    return 1
}

stop_server() {
    if [[ -n "${SERVER_PID:-}" ]] && kill -0 "$SERVER_PID" 2>/dev/null; then
        kill "$SERVER_PID" 2>/dev/null
        for _ in $(seq 1 30); do
            kill -0 "$SERVER_PID" 2>/dev/null || break
            sleep 1
        done
        kill -9 "$SERVER_PID" 2>/dev/null || true
        wait "$SERVER_PID" 2>/dev/null || true
    fi
    SERVER_PID=""
}

# make sure we never leave a server running if the script is interrupted
cleanup() { stop_server; }
trap cleanup EXIT INT TERM

run_one() {
    local model="$1" tp="$2" backend="$3"   # backend: "helion" or "none"
    local tag; tag="$(slug "$model")_tp${tp}_${backend}"
    local server_log="${RESULTS_DIR}/server_${tag}.log"
    local bench_log="${RESULTS_DIR}/bench_${tag}.log"

    echo "############################################################"
    echo "# MODEL=${model}  TP=${tp}  BACKEND=${backend}"
    echo "############################################################"

    local extra=()
    [[ "$backend" == "helion" ]] && extra=(--linear-backend helion)

    echo ">> starting server (log: ${server_log})"
    python3 -m vllm.entrypoints.openai.api_server \
        --model "$model" \
        --max-num-seqs 32 \
        --tensor-parallel-size "$tp" \
        --no-enable-prefix-caching \
        --port "$PORT" \
        "${extra[@]}" >"$server_log" 2>&1 &
    SERVER_PID=$!

    if ! wait_for_ready; then
        echo ">> SKIPPING ${tag} (server failed to start; see ${server_log})"
        stop_server
        rm -rf ~/.cache/vllm/
        return 1
    fi
    echo ">> server ready"

    echo ">> clearing L2 cache"
    ./clear_l2.sh || true

    echo ">> running bench.sh (log: ${bench_log})"
    {
        echo "# model=${model} tp=${tp} backend=${backend}"
        echo "# started $(date -Is)"
    } >"$bench_log"
    MODEL="$model" ./bench.sh >>"$bench_log" 2>&1
    echo "# finished $(date -Is)" >>"$bench_log"

    echo ">> stopping server + cleaning vLLM cache"
    stop_server
    rm -rf ~/.cache/vllm/
    echo ">> done ${tag}"
    echo
}

for entry in "${MODELS[@]}"; do
    model="${entry%:*}"
    tp="${entry##*:}"
    # allow bare "model" (no ":tp") to default to TP=1
    [[ "$tp" == "$entry" ]] && tp=1
    run_one "$model" "$tp" "helion" || true
    run_one "$model" "$tp" "none"   || true
done

echo "ALL DONE. Results in ${RESULTS_DIR}/"
