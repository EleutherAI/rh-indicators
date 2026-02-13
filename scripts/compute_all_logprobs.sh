#!/bin/bash
# Compute logprobs and KL divergence for all checkpoints in a prefill sensitivity run
#
# Usage: ./scripts/compute_all_logprobs.sh <run_dir> [tensor_parallel] [logprobs_k]
#
# Example:
#   ./scripts/compute_all_logprobs.sh results/prefill_sensitivity/prefill_sensitivity-20260127-050226-8a0e189 4 100
#
# This script:
# 1. Computes reference logprobs from the prefill generator (one-time)
# 2. For each eval checkpoint, computes logprobs + KL divergence
#
# For accurate KL divergence, set logprobs_k > 1 (e.g., 100-1000).
# This requests top-k logprobs at each position for proper KL computation.

set -e

RUN_DIR="${1:?Usage: $0 <run_dir> [tensor_parallel]}"
TP="${2:-4}"
PORT=8000

# KL divergence uses Monte Carlo estimation (log P - log Q), so we only need
# chosen-token logprobs (k=1). See docs/decisions/003-kl-divergence-monte-carlo.md

# Validate run directory
if [ ! -d "$RUN_DIR/evals" ]; then
    echo "Error: $RUN_DIR/evals not found"
    exit 1
fi

# Get checkpoint directory from config
CHECKPOINT_DIR=$(grep "^checkpoint_dir:" "$RUN_DIR/config.yaml" | cut -d' ' -f2)
if [ -z "$CHECKPOINT_DIR" ]; then
    echo "Error: Could not find checkpoint_dir in config.yaml"
    exit 1
fi
echo "Checkpoint directory: $CHECKPOINT_DIR"

# Parse reference checkpoint from prefill_source in config
# Format: results/prefill_ref/prefill_sensitivity-YYYYMMDD-HHMMSS-hash/evals/checkpoint-132.jsonl.samples.jsonl
PREFILL_SOURCE=$(grep "^prefill_source:" "$RUN_DIR/config.yaml" | cut -d' ' -f2)
REF_CKPT=$(echo "$PREFILL_SOURCE" | grep -oP 'checkpoint-\K\d+' | head -1)
if [ -z "$REF_CKPT" ]; then
    echo "Warning: Could not parse reference checkpoint from prefill_source, KL will be skipped"
    REF_CKPT=""
fi
echo "Reference checkpoint: ${REF_CKPT:-none}"

# Extract the prefill generator's checkpoint directory from the prefill_source run's config
# This is the model that GENERATED the prefills (different from eval checkpoints)
REF_CHECKPOINT_DIR=""
if [ -n "$PREFILL_SOURCE" ]; then
    # Get the run directory from prefill_source path (parent of /evals/)
    PREFILL_RUN_DIR=$(dirname "$(dirname "$PREFILL_SOURCE")")
    if [ -f "$PREFILL_RUN_DIR/config.yaml" ]; then
        REF_CHECKPOINT_DIR=$(grep "^checkpoint_dir:" "$PREFILL_RUN_DIR/config.yaml" | cut -d' ' -f2)
        echo "Reference checkpoint directory (prefill generator): $REF_CHECKPOINT_DIR"
    else
        echo "Warning: Could not find config at $PREFILL_RUN_DIR/config.yaml"
    fi
fi

# Fallback: if not found, try common locations
if [ -z "$REF_CHECKPOINT_DIR" ] || [ ! -d "$REF_CHECKPOINT_DIR" ]; then
    # Try to find matching checkpoint dir based on the prefill_source path pattern
    FALLBACK_DIR="results/sft_checkpoints_prefill/sft_openai_gpt-oss-20b-20260113-100558-8c90352/checkpoints"
    if [ -d "$FALLBACK_DIR" ]; then
        echo "Using fallback reference checkpoint directory: $FALLBACK_DIR"
        REF_CHECKPOINT_DIR="$FALLBACK_DIR"
    fi
fi

# Auto-detect harmony format from config
HARMONY_FLAG=""
NO_HARMONY=$(grep "^no_harmony:" "$RUN_DIR/config.yaml" | awk '{print $2}' || echo "false")
if [[ "$NO_HARMONY" != "true" ]] || [[ "$CHECKPOINT_DIR" == *"gpt-oss"* ]] || [[ "$CHECKPOINT_DIR" == *"gpt_oss"* ]]; then
    HARMONY_FLAG="--harmony"
    echo "Harmony format: enabled (auto-detected)"
else
    echo "Harmony format: disabled"
fi

# Create output directories
OUTPUT_DIR="$RUN_DIR/logprob"
REF_OUTPUT_DIR="$RUN_DIR/ref_logprob"
KL_OUTPUT_DIR="$RUN_DIR/kl"
mkdir -p "$OUTPUT_DIR"
mkdir -p "$REF_OUTPUT_DIR"
mkdir -p "$KL_OUTPUT_DIR"
mkdir -p "$RUN_DIR/logs"

# Extract unique checkpoints from eval files
CHECKPOINTS=$(ls "$RUN_DIR/evals/"*.samples.jsonl 2>/dev/null | \
    sed 's/.*checkpoint-//' | sed 's/_prefill.*//' | sort -n | uniq)

if [ -z "$CHECKPOINTS" ]; then
    echo "No sample files found in $RUN_DIR/evals/"
    exit 1
fi

echo "============================================================"
echo "LOGPROB + KL COMPUTATION VIA VLLM SERVER"
echo "============================================================"
echo "Run directory: $RUN_DIR"
echo "Checkpoint directory: $CHECKPOINT_DIR"
echo "Output directory: $OUTPUT_DIR"
echo "Reference logprobs: $REF_OUTPUT_DIR"
echo "KL output: $KL_OUTPUT_DIR"
echo "Checkpoints: $CHECKPOINTS"
echo "Reference checkpoint: ${REF_CKPT:-none}"
echo "Tensor parallel: $TP"
echo "============================================================"
echo ""

# Function to cleanup vLLM server
cleanup() {
    echo "Cleaning up vLLM server..."
    pkill -f "vllm serve" 2>/dev/null || true
    sleep 2
}
trap cleanup EXIT

# Function to start vLLM server and wait
start_vllm() {
    local CKPT_PATH="$1"
    local LOG_FILE="$2"

    pkill -f "vllm serve" 2>/dev/null || true
    sleep 3

    echo "Starting vLLM server with TP=$TP..."
    vllm serve "$CKPT_PATH" \
        --tensor-parallel-size $TP \
        --port $PORT \
        --gpu-memory-utilization 0.7 \
        --disable-log-requests \
        > "$LOG_FILE" 2>&1 &

    VLLM_PID=$!

    echo "Waiting for vLLM server to start..."
    local MAX_WAIT=300
    local WAITED=0
    while ! curl -s "http://localhost:$PORT/v1/models" > /dev/null 2>&1; do
        sleep 5
        WAITED=$((WAITED + 5))
        if [ $WAITED -ge $MAX_WAIT ]; then
            echo "Error: vLLM server did not start within ${MAX_WAIT}s"
            tail -50 "$LOG_FILE"
            kill $VLLM_PID 2>/dev/null || true
            return 1
        fi
        echo "  Waiting... (${WAITED}s)"
    done

    echo "vLLM server started!"
    return 0
}

# ============================================================
# STEP 1: Compute reference logprobs (if we have a reference checkpoint)
# ============================================================
if [ -n "$REF_CKPT" ]; then
    # Check if reference logprobs already exist
    REF_EXISTING=$(ls "$REF_OUTPUT_DIR/checkpoint-"*"_prefill"*"_logprobs.jsonl" 2>/dev/null | wc -l)
    REF_NEEDED=$(ls "$RUN_DIR/evals/"*"_prefill"*.samples.jsonl 2>/dev/null | grep -v prefill0 | head -1 | xargs -I{} ls "$RUN_DIR/evals/"*"_prefill"*.samples.jsonl 2>/dev/null | grep -v prefill0 | wc -l)
    # Count unique prefill levels needed
    REF_NEEDED=$(ls "$RUN_DIR/evals/"*"_prefill"*.samples.jsonl 2>/dev/null | grep -v prefill0 | sed 's/.*_prefill//' | sed 's/\.jsonl.*//' | sort -n | uniq | wc -l)

    if [ "$REF_EXISTING" -ge "$REF_NEEDED" ] && [ "$REF_NEEDED" -gt 0 ]; then
        echo "============================================================"
        echo "Reference logprobs already exist ($REF_EXISTING files), skipping"
        echo "============================================================"
    else
        echo "============================================================"
        echo "STEP 1: Computing reference logprobs from checkpoint-$REF_CKPT"
        echo "============================================================"

        # Find reference checkpoint path (from the prefill generator, not eval checkpoints)
        if [ -z "$REF_CHECKPOINT_DIR" ]; then
            echo "Warning: Reference checkpoint directory not found, KL computation will be skipped"
            REF_CKPT=""
        else
            REF_CKPT_PATH="$REF_CHECKPOINT_DIR/checkpoint-${REF_CKPT}_merged"
            if [ ! -d "$REF_CKPT_PATH" ]; then
                REF_CKPT_PATH="$REF_CHECKPOINT_DIR/checkpoint-${REF_CKPT}"
            fi
        fi

        if [ -z "$REF_CKPT" ]; then
            : # Already handled above
        elif [ ! -d "$REF_CKPT_PATH" ]; then
            echo "Warning: Reference checkpoint not found: $REF_CKPT_PATH"
            echo "KL computation will be skipped"
            REF_CKPT=""
        else
            echo "Reference checkpoint path: $REF_CKPT_PATH"

            if start_vllm "$REF_CKPT_PATH" "$RUN_DIR/logs/vllm_ref_ckpt${REF_CKPT}.log"; then
                # Use samples from first available checkpoint for reference logprobs
                # (prefill reasoning is the same across all checkpoints)
                FIRST_CKPT=$(echo "$CHECKPOINTS" | head -1)

                echo "Computing reference logprobs using samples from checkpoint-$FIRST_CKPT..."
                python scripts/compute_prefill_logprobs.py \
                    --base-url "http://localhost:$PORT/v1" \
                    --samples-dir "$RUN_DIR/evals" \
                    --output-dir "$REF_OUTPUT_DIR" \
                    --checkpoint $FIRST_CKPT \
                    --concurrency 8 \
                    --batch-size 16 \
                    $HARMONY_FLAG

                # Rename files to indicate they're reference logprobs (from ref checkpoint)
                for f in "$REF_OUTPUT_DIR/checkpoint-${FIRST_CKPT}_prefill"*"_logprobs.jsonl"; do
                    if [ -f "$f" ]; then
                        NEW_NAME=$(echo "$f" | sed "s/checkpoint-${FIRST_CKPT}/checkpoint-${REF_CKPT}/")
                        mv "$f" "$NEW_NAME"
                    fi
                done

                echo "Reference logprobs completed"
            else
                echo "Failed to start vLLM for reference checkpoint, skipping KL"
                REF_CKPT=""
            fi

            pkill -f "vllm serve" 2>/dev/null || true
            sleep 3
        fi
    fi
fi

# ============================================================
# STEP 2: Process each evaluation checkpoint
# ============================================================
echo ""
echo "============================================================"
echo "STEP 2: Computing logprobs + KL for evaluation checkpoints"
echo "============================================================"

for CKPT in $CHECKPOINTS; do
    echo ""
    echo "============================================================"
    echo "Processing checkpoint-$CKPT"
    echo "============================================================"

    # Check if all logprob files for this checkpoint already exist
    EXISTING=$(ls "$OUTPUT_DIR/checkpoint-${CKPT}_prefill"*"_logprobs.jsonl" 2>/dev/null | wc -l)
    NEEDED=$(ls "$RUN_DIR/evals/checkpoint-${CKPT}_prefill"*.samples.jsonl 2>/dev/null | \
        grep -v prefill0 | wc -l)

    # Also check if KL files need to be computed
    KL_EXISTING=$(ls "$KL_OUTPUT_DIR/checkpoint-${CKPT}_prefill"*"_kl.jsonl" 2>/dev/null | wc -l)
    NEED_KL=false
    if [ -n "$REF_CKPT" ] && [ "$KL_EXISTING" -lt "$NEEDED" ]; then
        NEED_KL=true
    fi

    if [ "$EXISTING" -ge "$NEEDED" ] && [ "$NEEDED" -gt 0 ] && [ "$NEED_KL" = false ]; then
        echo "All logprob and KL files for checkpoint-$CKPT already exist ($EXISTING files), skipping"
        continue
    fi

    # If logprobs exist but KL needs computation, we can compute KL without vLLM
    if [ "$EXISTING" -ge "$NEEDED" ] && [ "$NEED_KL" = true ]; then
        echo "Logprobs exist, computing KL only (no vLLM needed)..."

        # Build KL-only command (no --base-url needed since we use existing logprobs)
        CMD="python scripts/compute_prefill_logprobs.py \
            --samples-dir $RUN_DIR/evals \
            --output-dir $OUTPUT_DIR \
            --checkpoint $CKPT \
            --ref-logprobs-dir $REF_OUTPUT_DIR \
            --kl-output-dir $KL_OUTPUT_DIR \
            $HARMONY_FLAG"

        eval $CMD
        echo "Completed KL for checkpoint-$CKPT"
        continue
    fi

    # Need to compute logprobs (and possibly KL) - requires vLLM
    # Find the merged checkpoint path (prefer merged, fallback to unmerged)
    CKPT_PATH="$CHECKPOINT_DIR/checkpoint-${CKPT}_merged"
    if [ ! -d "$CKPT_PATH" ]; then
        CKPT_PATH="$CHECKPOINT_DIR/checkpoint-${CKPT}"
    fi

    if [ ! -d "$CKPT_PATH" ]; then
        echo "Warning: Checkpoint path not found: $CKPT_PATH"
        continue
    fi

    echo "Checkpoint path: $CKPT_PATH"

    if ! start_vllm "$CKPT_PATH" "$RUN_DIR/logs/vllm_ckpt${CKPT}.log"; then
        echo "Failed to start vLLM, skipping checkpoint-$CKPT"
        continue
    fi

    # Build command with optional KL arguments
    CMD="python scripts/compute_prefill_logprobs.py \
        --base-url http://localhost:$PORT/v1 \
        --samples-dir $RUN_DIR/evals \
        --output-dir $OUTPUT_DIR \
        --checkpoint $CKPT \
        --concurrency 8 \
        --batch-size 16 \
        $HARMONY_FLAG"

    # Add KL arguments if reference logprobs exist
    if [ -n "$REF_CKPT" ] && [ -d "$REF_OUTPUT_DIR" ] && [ "$(ls -A $REF_OUTPUT_DIR 2>/dev/null)" ]; then
        CMD="$CMD --ref-logprobs-dir $REF_OUTPUT_DIR --kl-output-dir $KL_OUTPUT_DIR"
        echo "Computing logprobs + KL (Monte Carlo)..."
    else
        echo "Computing logprobs only (no reference logprobs)..."
    fi

    eval $CMD

    echo "Completed checkpoint-$CKPT"

    # Kill vLLM server
    kill $VLLM_PID 2>/dev/null || true
    sleep 3
done

echo ""
echo "============================================================"
echo "All checkpoints processed!"
echo "Logprob results: $OUTPUT_DIR"
if [ -n "$REF_CKPT" ]; then
    echo "Reference logprobs: $REF_OUTPUT_DIR"
    echo "KL results: $KL_OUTPUT_DIR"
fi
echo "============================================================"
