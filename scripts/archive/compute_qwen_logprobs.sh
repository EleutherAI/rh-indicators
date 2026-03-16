#!/bin/bash
# Compute logprobs and KL divergence for the Qwen3-8B prefill sensitivity run.
#
# This script automates the serve-compute-kill loop for all checkpoints:
# 1. First: compute reference logprobs from the prefill source model
# 2. Then: compute eval logprobs + KL for each training checkpoint
#
# Usage:
#   bash scripts/compute_qwen_logprobs.sh          # Run everything
#   bash scripts/compute_qwen_logprobs.sh --skip-ref  # Skip reference logprobs (already computed)
#   bash scripts/compute_qwen_logprobs.sh --checkpoint 10  # Run only checkpoint 10

set -euo pipefail

# Paths
RUN_DIR="results/prefill_sensitivity/prefill_sensitivity-20260211-030018-8a0e189"
SFT_DIR="results/sft_checkpoints/sft_Qwen_Qwen3-8B-20260210-013438-8a0e189/checkpoints"
REF_MODEL="results/sft_checkpoints/sft_Qwen_Qwen3-8B-20260210-203200-8a0e189/checkpoints/checkpoint-300_merged"

EVALS_DIR="$RUN_DIR/evals"
REF_LOGPROB_DIR="$RUN_DIR/ref_logprob"
LOGPROB_DIR="$RUN_DIR/logprob"
KL_DIR="$RUN_DIR/kl"

# vLLM settings
PORT=8000
TP=4
GPU_MEM=0.70
CONCURRENCY=32

# Checkpoints with complete prefill data (10 prefill levels each)
ALL_CHECKPOINTS=(1 2 3 4 6 7 10 12 16 21)

# Parse args
SKIP_REF=false
ONLY_CHECKPOINT=""
while [[ $# -gt 0 ]]; do
    case $1 in
        --skip-ref) SKIP_REF=true; shift ;;
        --checkpoint) ONLY_CHECKPOINT="$2"; shift 2 ;;
        *) echo "Unknown arg: $1"; exit 1 ;;
    esac
done

# If --checkpoint specified, only process that one
if [[ -n "$ONLY_CHECKPOINT" ]]; then
    CHECKPOINTS=("$ONLY_CHECKPOINT")
else
    CHECKPOINTS=("${ALL_CHECKPOINTS[@]}")
fi

wait_for_server() {
    local url="http://localhost:$PORT/v1/models"
    local timeout=600
    local start=$SECONDS
    echo "Waiting for vLLM server at $url ..."
    while true; do
        if curl -s "$url" | python3 -c "import sys,json; d=json.load(sys.stdin); assert len(d['data'])>0" 2>/dev/null; then
            echo "Server ready (took $((SECONDS - start))s)"
            return 0
        fi
        if (( SECONDS - start > timeout )); then
            echo "ERROR: Server failed to start within ${timeout}s"
            return 1
        fi
        sleep 5
    done
}

kill_server() {
    echo "Stopping vLLM server (PID $VLLM_PID)..."
    kill -TERM "$VLLM_PID" 2>/dev/null || true
    # Wait for graceful shutdown
    for i in $(seq 1 20); do
        if ! kill -0 "$VLLM_PID" 2>/dev/null; then
            echo "Server stopped gracefully"
            return 0
        fi
        sleep 1
    done
    # Force kill
    echo "Force killing vLLM..."
    kill -9 "$VLLM_PID" 2>/dev/null || true
    sleep 2
    echo "Server killed"
}

# Create output dirs
mkdir -p "$REF_LOGPROB_DIR" "$LOGPROB_DIR" "$KL_DIR"

# ============================================================
# Step 1: Compute reference logprobs
# ============================================================
if [[ "$SKIP_REF" == false ]]; then
    echo "============================================================"
    echo "Step 1: Computing reference logprobs from $REF_MODEL"
    echo "============================================================"

    # Check if ref logprobs already exist for all needed prefill levels
    REF_DONE=true
    for pfx in 2 5 10 20 30 45 60 75 100; do
        if [[ ! -f "$REF_LOGPROB_DIR/checkpoint-21_prefill${pfx}_logprobs.jsonl" ]]; then
            REF_DONE=false
            break
        fi
    done

    if [[ "$REF_DONE" == true ]]; then
        echo "Reference logprobs already exist, skipping"
    else
        # Start vLLM with reference model
        echo "Starting vLLM for reference model..."
        vllm serve "$REF_MODEL" \
            --tensor-parallel-size "$TP" \
            --gpu-memory-utilization "$GPU_MEM" \
            --port "$PORT" \
            > "$RUN_DIR/logs/vllm_ref.log" 2>&1 &
        VLLM_PID=$!
        echo "vLLM PID: $VLLM_PID"

        if wait_for_server; then
            # Compute ref logprobs using checkpoint-21 samples (has all prefill levels)
            python scripts/compute_prefill_logprobs.py \
                --base-url "http://localhost:$PORT/v1" \
                --samples-dir "$EVALS_DIR" \
                --output-dir "$REF_LOGPROB_DIR" \
                --checkpoint 21 \
                --concurrency "$CONCURRENCY"

            kill_server
        else
            kill_server
            echo "FATAL: Reference model server failed to start"
            exit 1
        fi
    fi
else
    echo "Skipping reference logprobs (--skip-ref)"
fi

# ============================================================
# Step 2: Compute eval logprobs + KL for each checkpoint
# ============================================================
echo ""
echo "============================================================"
echo "Step 2: Computing eval logprobs + KL for checkpoints: ${CHECKPOINTS[*]}"
echo "============================================================"

for CKPT in "${CHECKPOINTS[@]}"; do
    MERGED="${SFT_DIR}/checkpoint-${CKPT}_merged"

    if [[ ! -d "$MERGED" ]]; then
        echo "WARNING: Merged model not found at $MERGED, skipping checkpoint-$CKPT"
        continue
    fi

    # Check if this checkpoint is already done (all KL files exist)
    CKPT_DONE=true
    for pfx in 2 5 10 20 30 45 60 75 100; do
        if [[ ! -f "$KL_DIR/checkpoint-${CKPT}_prefill${pfx}_kl.jsonl" ]]; then
            CKPT_DONE=false
            break
        fi
    done

    if [[ "$CKPT_DONE" == true ]]; then
        echo ""
        echo "--- Checkpoint $CKPT: already complete, skipping ---"
        continue
    fi

    echo ""
    echo "============================================================"
    echo "Processing checkpoint-$CKPT ($MERGED)"
    echo "============================================================"

    # Ensure log dir exists
    mkdir -p "$RUN_DIR/logs"

    # Start vLLM
    echo "Starting vLLM for checkpoint-$CKPT..."
    vllm serve "$MERGED" \
        --tensor-parallel-size "$TP" \
        --gpu-memory-utilization "$GPU_MEM" \
        --port "$PORT" \
        > "$RUN_DIR/logs/vllm_ckpt${CKPT}.log" 2>&1 &
    VLLM_PID=$!
    echo "vLLM PID: $VLLM_PID"

    if wait_for_server; then
        # Compute logprobs + KL
        python scripts/compute_prefill_logprobs.py \
            --base-url "http://localhost:$PORT/v1" \
            --samples-dir "$EVALS_DIR" \
            --output-dir "$LOGPROB_DIR" \
            --ref-logprobs-dir "$REF_LOGPROB_DIR" \
            --kl-output-dir "$KL_DIR" \
            --checkpoint "$CKPT" \
            --concurrency "$CONCURRENCY"

        kill_server
    else
        kill_server
        echo "ERROR: Server failed to start for checkpoint-$CKPT, continuing to next..."
        continue
    fi

    echo "Checkpoint-$CKPT complete!"
done

echo ""
echo "============================================================"
echo "All done! Results:"
echo "  Reference logprobs: $REF_LOGPROB_DIR"
echo "  Eval logprobs:      $LOGPROB_DIR"
echo "  KL divergence:      $KL_DIR"
echo "============================================================"
