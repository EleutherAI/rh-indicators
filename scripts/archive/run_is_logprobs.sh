#!/usr/bin/env bash
# Phase 2: Collect top-K=1000 logprobs for exact IS estimates
# Runs sequentially through gpt-oss-20b checkpoints on GPUs 0-1 (TP=2)
#
# Checkpoints: 6, 15, 25, 44 (target, overlap with 64-attempt ground truth)
#              132 (donor, for D_τᴰ(z|x))
#
# Usage: bash scripts/run_is_logprobs.sh 2>&1 | tee results/is_logprobs/run.log

set -euo pipefail

CKPT_BASE="/mnt/ssd-1/david/rh-indicators/results/sft_checkpoints_eval/sft_openai_gpt-oss-20b-20260113-060036-8c90352/checkpoints"
SAMPLES_DIR="/mnt/ssd-1/david/rh-indicators/results/prefill_sensitivity/prefill_sensitivity-20260127-050226-8a0e189/backup_t04_20260306_105123/evals"
OUTPUT_DIR="/mnt/ssd-1/david/rh-indicators/results/is_logprobs/gpt-oss-20b"
PORT=8000
GPUS="0,1"
LOGPROBS_K=1000
CONCURRENCY=32

# Target checkpoints (ground truth overlap) + donor
CHECKPOINTS=(6 15 25 44 132)

mkdir -p "$OUTPUT_DIR"

for CKPT in "${CHECKPOINTS[@]}"; do
    CKPT_DIR="$CKPT_BASE/checkpoint-${CKPT}_merged"
    CKPT_OUTPUT="$OUTPUT_DIR/checkpoint-${CKPT}"

    # Skip if output already exists with expected number of files
    if [ -d "$CKPT_OUTPUT" ]; then
        n_files=$(ls "$CKPT_OUTPUT"/*.jsonl 2>/dev/null | wc -l)
        if [ "$n_files" -ge 9 ]; then
            echo "=== Checkpoint $CKPT: already done ($n_files files), skipping ==="
            continue
        fi
    fi

    echo ""
    echo "============================================="
    echo "  Checkpoint $CKPT — starting $(date)"
    echo "============================================="

    # Verify checkpoint exists
    if [ ! -f "$CKPT_DIR/config.json" ]; then
        echo "ERROR: $CKPT_DIR/config.json not found, skipping"
        continue
    fi

    # Start vLLM server
    echo "Starting vLLM server for checkpoint-$CKPT on GPUs $GPUS..."
    CUDA_VISIBLE_DEVICES=$GPUS vllm serve "$CKPT_DIR" \
        --tensor-parallel-size 2 \
        --max-logprobs $LOGPROBS_K \
        --port $PORT \
        --max-model-len 4096 \
        &
    VLLM_PID=$!

    # Wait for server to be ready
    echo "Waiting for vLLM server (PID $VLLM_PID)..."
    for i in $(seq 1 360); do
        if curl -s http://localhost:$PORT/health > /dev/null 2>&1; then
            echo "Server ready after ${i}s"
            break
        fi
        if ! kill -0 $VLLM_PID 2>/dev/null; then
            echo "ERROR: vLLM server died during startup"
            break 2
        fi
        sleep 1
    done

    if ! curl -s http://localhost:$PORT/health > /dev/null 2>&1; then
        echo "ERROR: Server not ready after 360s, killing and skipping"
        kill $VLLM_PID 2>/dev/null || true
        wait $VLLM_PID 2>/dev/null || true
        continue
    fi

    # Collect top-K logprobs
    mkdir -p "$CKPT_OUTPUT"
    echo "Collecting top-K=$LOGPROBS_K logprobs for checkpoint-$CKPT..."
    python scripts/compute_prefill_logprobs.py \
        --base-url "http://localhost:$PORT/v1" \
        --samples-dir "$SAMPLES_DIR" \
        --output-dir "$CKPT_OUTPUT" \
        --checkpoint "$CKPT" \
        --logprobs-k "$LOGPROBS_K" \
        --concurrency "$CONCURRENCY" \
        --min-prefill 2

    echo "Logprobs done for checkpoint-$CKPT at $(date)"

    # Stop vLLM server
    echo "Stopping vLLM server..."
    kill $VLLM_PID 2>/dev/null || true
    wait $VLLM_PID 2>/dev/null || true
    sleep 5  # Let GPU memory clear

    echo "Checkpoint $CKPT complete"
done

echo ""
echo "============================================="
echo "  All checkpoints complete — $(date)"
echo "============================================="
echo "Output: $OUTPUT_DIR"
ls -la "$OUTPUT_DIR"/*/
