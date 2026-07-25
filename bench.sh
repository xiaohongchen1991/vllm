#!/usr/bin/env bash
set -euo pipefail

MODEL="${MODEL:-RedHatAI/Qwen3-32B-FP8-dynamic}"
DATASET="ShareGPT_V3_unfiltered_cleaned_split.json"

for concurrency in 32 24 16 8 4 2 1; do
    warmups=$((concurrency * 8))
    prompts=$((concurrency * 128))

    echo "============================================================"
    echo "Running benchmark"
    echo "  max-concurrency = ${concurrency}"
    echo "  num-warmups     = ${warmups}"
    echo "  num-prompts     = ${prompts}"
    echo "============================================================"

    vllm bench serve \
        --backend vllm \
        --model "${MODEL}" \
        --endpoint /v1/completions \
        --dataset-name sharegpt \
        --dataset-path "${DATASET}" \
        --max-concurrency "${concurrency}" \
        --num-warmups "${warmups}" \
        --ignore-eos \
        --num-prompts "${prompts}"

    echo
done
