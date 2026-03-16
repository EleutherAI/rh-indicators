#!/bin/bash
# Compute prefill logprobs + KL for Qwen 3-8B pivot prefill run
# Two phases:
#   Phase 0: Compute reference logprobs (checkpoint-300 on pivot samples) — 1 GPU
#   Phase 1: Compute eval logprobs + KL for 8 checkpoints — 8 GPUs in parallel
#
# IMPORTANT: The pivot run uses different prefill text (post-pivot reasoning) than the
# original run. We CANNOT reuse the original run's ref_logprobs — must regenerate them
# by scoring the pivot samples under checkpoint-300.
#
# Usage:
#   bash scripts/run_qwen_pivot_logprobs.sh           # Full run (phase 0 + 1)
#   bash scripts/run_qwen_pivot_logprobs.sh --skip-ref # Skip phase 0 (ref already computed)
set -euo pipefail

PYTHON="python"
SCRIPT="/mnt/ssd-1/david/rh-indicators/scripts/compute_prefill_logprobs.py"
BASE_PORT=8001
LOG_DIR="/tmp/vllm_qwen_pivot_logprobs"
mkdir -p "$LOG_DIR"

# ── Paths ────────────────────────────────────────────────────────────────
CKPT_DIR="/mnt/ssd-1/david/rh-indicators/results/sft_checkpoints/sft_Qwen_Qwen3-8B-20260210-013438-8a0e189/checkpoints"
RUN_DIR="/mnt/ssd-1/david/rh-indicators/results/prefill_sensitivity/prefill_sensitivity-20260224-034624-6d05d75"
REF_MODEL="/mnt/ssd-1/david/rh-indicators/results/sft_checkpoints/sft_Qwen_Qwen3-8B-20260210-203200-8a0e189/checkpoints/checkpoint-300_merged"

# Only the 8 pivot checkpoints (NOT the extra clean-control checkpoints in the dir)
PIVOT_CKPTS=(1 3 10 27 36 77 167 220)

SKIP_REF=false
if [[ "${1:-}" == "--skip-ref" ]]; then
    SKIP_REF=true
    echo "Skipping Phase 0 (reference logprobs)"
fi

# ── Helper functions ───────────────────────────────────────────────────

start_server() {
    local gpu=$1 port=$2 model_path=$3 label=$4
    echo "  Starting vLLM for ${label} on GPU $gpu, port $port..." >&2
    CUDA_VISIBLE_DEVICES=$gpu $PYTHON -m vllm.entrypoints.openai.api_server \
        --model "$model_path" \
        --port $port \
        --trust-remote-code \
        --gpu-memory-utilization 0.85 \
        --max-num-seqs 32 \
        --max-model-len 8192 \
        > "$LOG_DIR/vllm_${label}_gpu${gpu}.log" 2>&1 &
    echo $!
}

wait_for_server() {
    local port=$1 timeout=${2:-180}
    echo -n "  Port $port: "
    for attempt in $(seq 1 $((timeout / 5))); do
        if curl -s "http://localhost:$port/v1/models" >/dev/null 2>&1; then
            echo "ready (${attempt}x5s)"
            return 0
        fi
        sleep 5
    done
    echo "TIMEOUT after ${timeout}s"
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

# ══════════════════════════════════════════════════════════════════════════
# Phase 0: Reference logprobs (checkpoint-300 scoring pivot prefill text)
# ══════════════════════════════════════════════════════════════════════════
if [ "$SKIP_REF" = false ]; then
    echo "=========================================="
    echo "Phase 0: Computing reference logprobs"
    echo "  Model: $REF_MODEL"
    echo "  Samples: ${RUN_DIR}/evals (checkpoint-1 samples)"
    echo "  Output: ${RUN_DIR}/ref_logprob"
    echo "=========================================="

    # Serve checkpoint-300 on GPU 0
    REF_PID=$(start_server 0 $BASE_PORT "$REF_MODEL" "ref_ckpt300")

    if wait_for_server $BASE_PORT 300; then
        # Compute ref logprobs using checkpoint-1's samples
        # (prefill_reasoning is identical across checkpoints for the same task+prefill_level)
        $PYTHON "$SCRIPT" \
            --base-url "http://localhost:$BASE_PORT/v1" \
            --samples-dir "${RUN_DIR}/evals" \
            --output-dir "${RUN_DIR}/ref_logprob" \
            --checkpoint 1 \
            --concurrency 32 \
            2>&1 | tee "$LOG_DIR/compute_ref_logprobs.log"
    else
        echo "ERROR: Reference model server failed to start"
        kill_servers "$REF_PID"
        exit 1
    fi

    kill_servers "$REF_PID"

    ref_count=$(ls "${RUN_DIR}/ref_logprob/"*.jsonl 2>/dev/null | wc -l)
    echo "Phase 0 complete: ${ref_count} ref logprob files"
    echo ""
fi

# ══════════════════════════════════════════════════════════════════════════
# Phase 1: Eval logprobs + KL for all 8 pivot checkpoints
# ══════════════════════════════════════════════════════════════════════════
echo "=========================================="
echo "Phase 1: Computing eval logprobs + KL"
echo "  Checkpoints: ${PIVOT_CKPTS[*]}"
echo "  Run dir: $RUN_DIR"
echo "=========================================="

# All 8 checkpoints fit on 8 GPUs simultaneously
SERVER_PIDS=()
ACTIVE_CKPTS=()
ACTIVE_PORTS=()
gpu=0

for ckpt in "${PIVOT_CKPTS[@]}"; do
    port=$((BASE_PORT + gpu))
    model_path="${CKPT_DIR}/checkpoint-${ckpt}_merged"

    if [ ! -d "$model_path" ]; then
        echo "  WARNING: $model_path not found, skipping"
        gpu=$((gpu + 1))
        continue
    fi

    pid=$(start_server $gpu $port "$model_path" "pivot_ckpt${ckpt}")
    SERVER_PIDS+=($pid)
    ACTIVE_CKPTS+=($ckpt)
    ACTIVE_PORTS+=($port)
    gpu=$((gpu + 1))
done

if [ ${#SERVER_PIDS[@]} -eq 0 ]; then
    echo "ERROR: No servers started"
    exit 1
fi

# Wait for all servers
echo "Waiting for ${#SERVER_PIDS[@]} servers..."
ALL_READY=true
for port in "${ACTIVE_PORTS[@]}"; do
    if ! wait_for_server $port 300; then
        ALL_READY=false
        echo "  WARNING: Server on port $port failed to start"
    fi
done

if [ "$ALL_READY" = false ]; then
    echo "Some servers failed. Check logs in $LOG_DIR"
    echo "Continuing with available servers..."
fi

# Run logprob + KL computation in parallel
COMPUTE_PIDS=()
for i in "${!ACTIVE_CKPTS[@]}"; do
    ckpt="${ACTIVE_CKPTS[$i]}"
    port="${ACTIVE_PORTS[$i]}"

    if ! curl -s "http://localhost:$port/v1/models" >/dev/null 2>&1; then
        echo "  WARNING: Server on port $port not responding, skipping checkpoint-${ckpt}"
        continue
    fi

    echo "  Computing logprobs + KL for checkpoint-${ckpt}..."
    $PYTHON "$SCRIPT" \
        --base-url "http://localhost:$port/v1" \
        --samples-dir "${RUN_DIR}/evals" \
        --output-dir "${RUN_DIR}/logprob" \
        --checkpoint "$ckpt" \
        --ref-logprobs-dir "${RUN_DIR}/ref_logprob" \
        --kl-output-dir "${RUN_DIR}/kl" \
        --concurrency 32 \
        2>&1 | tee "$LOG_DIR/compute_pivot_ckpt${ckpt}.log" &
    COMPUTE_PIDS+=($!)
done

# Wait for computations
echo "Waiting for ${#COMPUTE_PIDS[@]} logprob computations..."
for pid in "${COMPUTE_PIDS[@]}"; do
    wait $pid || echo "WARNING: Process $pid failed"
done

# Kill servers
kill_servers "${SERVER_PIDS[@]}"

# ── Summary ──────────────────────────────────────────────────────────────
echo ""
echo "=========================================="
echo "Pivot prefill logprob+KL computation done!"
echo ""
logprob_count=$(ls "${RUN_DIR}/logprob/"*.jsonl 2>/dev/null | wc -l)
kl_count=$(ls "${RUN_DIR}/kl/"*.jsonl 2>/dev/null | wc -l)
ref_count=$(ls "${RUN_DIR}/ref_logprob/"*.jsonl 2>/dev/null | wc -l)
echo "  ref_logprob: ${ref_count} files"
echo "  logprob:     ${logprob_count} files (expect ~72: 8 ckpts × 9 prefill levels)"
echo "  kl:          ${kl_count} files (expect ~72: 8 ckpts × 9 prefill levels)"
echo "=========================================="
