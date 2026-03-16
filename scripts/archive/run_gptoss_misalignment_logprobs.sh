#!/bin/bash
# Compute prefill logprobs + KL for GPT-OSS 20B misalignment control
# Reuses ref_logprobs from the exploit run (same base model, same task_ids).
# Serves each misalignment checkpoint on a separate GPU (8 at a time).
#
# GPT-OSS 20B (~40GB fp16) fits on a single A100-80GB.
set -euo pipefail

PYTHON="python"
SCRIPT="/mnt/ssd-1/david/rh-indicators/scripts/compute_prefill_logprobs.py"
BASE_PORT=8001
LOG_DIR="/tmp/vllm_gptoss_misalignment_logprobs"
mkdir -p "$LOG_DIR"

# ── Paths ────────────────────────────────────────────────────────────
MISALIGN_CKPT_DIR="/mnt/ssd-1/david/rh-indicators/results/sft_checkpoints_misalignment_control/sft_openai_gpt-oss-20b-20260217-014829-3a546a8/checkpoints"
RUN_DIR="/mnt/ssd-1/david/rh-indicators/results/prefill_sensitivity/prefill_sensitivity-20260223-030541"
# Reuse exploit run's ref_logprobs (checkpoint-132, same dataset/split)
REF_LOGPROB_DIR="/mnt/ssd-1/david/rh-indicators/results/prefill_sensitivity/prefill_sensitivity-20260127-050226-8a0e189/ref_logprob"

MISALIGN_CKPTS=(1 5 12 22 53 71 126 224 396)

LOGPROB_DIR="${RUN_DIR}/logprob"
KL_DIR="${RUN_DIR}/kl"
mkdir -p "$LOGPROB_DIR" "$KL_DIR"

# ── Helper functions ─────────────────────────────────────────────────

start_server() {
    local gpu=$1 port=$2 model_path=$3 label=$4
    echo "  Starting vLLM for ${label} on GPU $gpu, port $port..." >&2
    CUDA_VISIBLE_DEVICES=$gpu $PYTHON -m vllm.entrypoints.openai.api_server \
        --model "$model_path" \
        --port $port \
        --trust-remote-code \
        --gpu-memory-utilization 0.85 \
        --max-num-seqs 16 \
        --max-model-len 8192 \
        > "$LOG_DIR/vllm_${label}_gpu${gpu}.log" 2>&1 &
    echo $!
}

wait_for_server() {
    local port=$1 timeout=${2:-300}
    echo -n "  Port $port: " >&2
    for attempt in $(seq 1 $((timeout / 5))); do
        if curl -s "http://localhost:$port/v1/models" >/dev/null 2>&1; then
            echo "ready (${attempt}x5s)" >&2
            return 0
        fi
        sleep 5
    done
    echo "TIMEOUT after ${timeout}s" >&2
    return 1
}

kill_servers() {
    local pids=("$@")
    echo "Shutting down servers..."
    for pid in "${pids[@]}"; do
        kill $pid 2>/dev/null || true
    done
    sleep 3
    for pid in "${pids[@]}"; do
        kill -9 $pid 2>/dev/null || true
    done
    sleep 2
}

run_logprobs() {
    local port=$1 ckpt=$2
    echo "  Computing logprobs+KL for checkpoint-${ckpt}..."
    $PYTHON "$SCRIPT" \
        --base-url "http://localhost:$port/v1" \
        --samples-dir "${RUN_DIR}/evals" \
        --output-dir "$LOGPROB_DIR" \
        --checkpoint "$ckpt" \
        --ref-logprobs-dir "$REF_LOGPROB_DIR" \
        --kl-output-dir "$KL_DIR" \
        --concurrency 16 \
        2>&1 | tee "$LOG_DIR/compute_ckpt${ckpt}.log"
}

# ══════════════════════════════════════════════════════════════════════
echo "=========================================="
echo "GPT-OSS 20B Misalignment Control: Logprob + KL"
echo "  Checkpoints: ${MISALIGN_CKPTS[*]}"
echo "  Run dir: $RUN_DIR"
echo "  Ref logprobs: $REF_LOGPROB_DIR"
echo "=========================================="
echo ""

# Verify ref_logprobs exist
REF_COUNT=$(ls "$REF_LOGPROB_DIR"/*.jsonl 2>/dev/null | wc -l)
if [ "$REF_COUNT" -eq 0 ]; then
    echo "ERROR: No ref_logprob files found at $REF_LOGPROB_DIR"
    exit 1
fi
echo "Using $REF_COUNT reference logprob files from exploit run"
echo ""

# ── Process in batches of 8 (one per GPU) ────────────────────────────
BATCH_SIZE=8
for ((batch_start=0; batch_start<${#MISALIGN_CKPTS[@]}; batch_start+=BATCH_SIZE)); do
    batch=("${MISALIGN_CKPTS[@]:$batch_start:$BATCH_SIZE}")
    echo "=========================================="
    echo "Batch $((batch_start / BATCH_SIZE + 1)): checkpoints ${batch[*]}"
    echo "=========================================="

    # Start servers (staggered to avoid Harmony encoding race condition)
    SERVER_PIDS=()
    ACTIVE_CKPTS=()
    ACTIVE_PORTS=()
    gpu=0
    for ckpt in "${batch[@]}"; do
        port=$((BASE_PORT + gpu))
        model_path="${MISALIGN_CKPT_DIR}/checkpoint-${ckpt}_merged"

        if [ ! -d "$model_path" ]; then
            echo "  WARNING: $model_path not found, skipping"
            gpu=$((gpu + 1))
            continue
        fi

        pid=$(start_server $gpu $port "$model_path" "ckpt${ckpt}")
        SERVER_PIDS+=($pid)
        ACTIVE_CKPTS+=($ckpt)
        ACTIVE_PORTS+=($port)
        gpu=$((gpu + 1))
        # Stagger startup by 5s to avoid Harmony tokenizer race condition
        sleep 5
    done

    if [ ${#SERVER_PIDS[@]} -eq 0 ]; then
        echo "No servers to start in this batch"
        continue
    fi

    # Wait for all servers
    echo "Waiting for ${#SERVER_PIDS[@]} servers..."
    for port in "${ACTIVE_PORTS[@]}"; do
        if ! wait_for_server $port 300; then
            echo "  WARNING: Server on port $port failed to start"
        fi
    done

    # Run logprob+KL computation in parallel
    COMPUTE_PIDS=()
    for i in "${!ACTIVE_CKPTS[@]}"; do
        ckpt="${ACTIVE_CKPTS[$i]}"
        port="${ACTIVE_PORTS[$i]}"

        if ! curl -s "http://localhost:$port/v1/models" >/dev/null 2>&1; then
            echo "  WARNING: Server on port $port not responding, skipping checkpoint-${ckpt}"
            continue
        fi

        run_logprobs $port "$ckpt" &
        COMPUTE_PIDS+=($!)
    done

    # Wait for computations
    echo "Waiting for ${#COMPUTE_PIDS[@]} logprob computations..."
    for pid in "${COMPUTE_PIDS[@]}"; do
        wait $pid || echo "WARNING: Process $pid failed"
    done

    # Kill servers
    kill_servers "${SERVER_PIDS[@]}"

    echo "Batch complete!"
    echo ""
done

echo "=========================================="
echo "All GPT-OSS misalignment logprob+KL computations done!"
echo ""
echo "  logprob files: $(ls "$LOGPROB_DIR"/*.jsonl 2>/dev/null | wc -l)"
echo "  KL files: $(ls "$KL_DIR"/*.jsonl 2>/dev/null | wc -l)"
echo "=========================================="
