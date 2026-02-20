#!/bin/bash
# Compute logprobs + KL divergence for multiple prefill sensitivity runs in parallel.
#
# Each run gets its own GPU and port, avoiding the pkill issues of the generic script.
# Designed for models that fit on a single GPU (e.g., Qwen3-8B on A100-80GB).
#
# Usage:
#   bash scripts/compute_logprobs_parallel.sh <run_dir1> [run_dir2] [run_dir3] ...
#
# Example:
#   bash scripts/compute_logprobs_parallel.sh \
#     results/prefill_sensitivity/prefill_sensitivity-20260211-030018-8a0e189 \
#     results/prefill_sensitivity/prefill_sensitivity-20260217-235715-3a546a8 \
#     results/prefill_sensitivity/prefill_sensitivity-20260217-054915

set -euo pipefail

if [[ $# -lt 1 ]]; then
    echo "Usage: $0 <run_dir1> [run_dir2] [run_dir3] ..."
    exit 1
fi

RUN_DIRS=("$@")
NUM_RUNS=${#RUN_DIRS[@]}
BASE_PORT=8000
GPU_MEM=0.85
CONCURRENCY=32

echo "============================================================"
echo "PARALLEL LOGPROB + KL COMPUTATION"
echo "============================================================"
echo "Runs: $NUM_RUNS"
for i in "${!RUN_DIRS[@]}"; do
    echo "  [$i] ${RUN_DIRS[$i]} -> GPU $i, port $((BASE_PORT + i))"
done
echo "============================================================"

# Each worker writes its own log file
TMPDIR=$(mktemp -d)

# Cleanup on exit: kill all vllm servers by port
cleanup_all() {
    echo ""
    echo "Cleaning up..."
    for i in $(seq 0 $((NUM_RUNS - 1))); do
        local_port=$((BASE_PORT + i))
        pids=$(lsof -ti :"$local_port" 2>/dev/null || true)
        if [[ -n "$pids" ]]; then
            echo "Killing processes on port $local_port"
            echo "$pids" | xargs kill -9 2>/dev/null || true
        fi
    done
    rm -rf "$TMPDIR"
    wait 2>/dev/null || true
}
trap cleanup_all EXIT

# Self-contained worker script that gets run as a subprocess
run_worker() {
    local RUN_DIR="$1"
    local GPU_ID="$2"
    local PORT="$3"
    local P="[GPU$GPU_ID:$PORT]"

    export CUDA_VISIBLE_DEVICES="$GPU_ID"

    echo "$P Starting processing for $RUN_DIR"

    # Validate
    if [[ ! -d "$RUN_DIR/evals" ]]; then
        echo "$P ERROR: $RUN_DIR/evals not found"
        return 1
    fi

    # Parse config
    local CHECKPOINT_DIR
    CHECKPOINT_DIR=$(grep "^checkpoint_dir:" "$RUN_DIR/config.yaml" | cut -d' ' -f2)
    echo "$P Checkpoint dir: $CHECKPOINT_DIR"

    local PREFILL_SOURCE
    PREFILL_SOURCE=$(grep "^prefill_source:" "$RUN_DIR/config.yaml" | cut -d' ' -f2)
    local REF_CKPT
    REF_CKPT=$(echo "$PREFILL_SOURCE" | grep -oP 'checkpoint-\K\d+' | head -1 || true)
    echo "$P Reference checkpoint: ${REF_CKPT:-none}"

    local REF_CHECKPOINT_DIR=""
    if [[ -n "${PREFILL_SOURCE:-}" ]]; then
        local PREFILL_RUN_DIR
        PREFILL_RUN_DIR=$(dirname "$(dirname "$PREFILL_SOURCE")")
        if [[ -f "$PREFILL_RUN_DIR/config.yaml" ]]; then
            REF_CHECKPOINT_DIR=$(grep "^checkpoint_dir:" "$PREFILL_RUN_DIR/config.yaml" | cut -d' ' -f2)
            echo "$P Reference checkpoint dir: $REF_CHECKPOINT_DIR"
        fi
    fi

    local HARMONY_FLAG=""
    local NO_HARMONY
    NO_HARMONY=$(grep "^no_harmony:" "$RUN_DIR/config.yaml" | awk '{print $2}' 2>/dev/null || echo "false")
    if [[ "$NO_HARMONY" != "true" ]] || [[ "$CHECKPOINT_DIR" == *"gpt-oss"* ]] || [[ "$CHECKPOINT_DIR" == *"gpt_oss"* ]]; then
        HARMONY_FLAG="--harmony"
        echo "$P Harmony format: enabled"
    else
        echo "$P Harmony format: disabled"
    fi

    local OUTPUT_DIR="$RUN_DIR/logprob"
    local REF_OUTPUT_DIR="$RUN_DIR/ref_logprob"
    local KL_OUTPUT_DIR="$RUN_DIR/kl"
    mkdir -p "$OUTPUT_DIR" "$REF_OUTPUT_DIR" "$KL_OUTPUT_DIR" "$RUN_DIR/logs"

    local CHECKPOINTS
    CHECKPOINTS=$(ls "$RUN_DIR/evals/"checkpoint-*.samples.jsonl 2>/dev/null | \
        sed 's/.*checkpoint-//' | sed 's/_prefill.*//' | sort -n | uniq)
    echo "$P Checkpoints: $(echo $CHECKPOINTS | tr '\n' ' ')"

    # --- vLLM helpers ---
    # PID file to communicate between start/stop without subshells
    local VLLM_PID_FILE="$TMPDIR/vllm_pid_gpu${GPU_ID}"

    _start_vllm() {
        local model_path="$1"
        local log_file="$2"

        echo "$P Starting vLLM: $(basename "$model_path")"
        vllm serve "$model_path" \
            --tensor-parallel-size 1 \
            --port "$PORT" \
            --gpu-memory-utilization "$GPU_MEM" \
            --disable-log-requests \
            > "$log_file" 2>&1 &
        echo $! > "$VLLM_PID_FILE"
        echo "$P vLLM PID: $(cat "$VLLM_PID_FILE")"

        local max_wait=600 waited=0
        while ! curl -s "http://localhost:$PORT/v1/models" > /dev/null 2>&1; do
            sleep 5
            waited=$((waited + 5))
            if [[ $waited -ge $max_wait ]]; then
                echo "$P ERROR: vLLM did not start within ${max_wait}s"
                tail -30 "$log_file" || true
                kill "$(cat "$VLLM_PID_FILE")" 2>/dev/null || true
                return 1
            fi
        done
        echo "$P vLLM ready (${waited}s)"
    }

    _stop_vllm() {
        local pid
        pid=$(cat "$VLLM_PID_FILE" 2>/dev/null || echo "")
        if [[ -z "$pid" ]]; then return 0; fi
        echo "$P Stopping vLLM PID $pid..."
        kill -TERM "$pid" 2>/dev/null || true
        local i
        for i in $(seq 1 30); do
            if ! kill -0 "$pid" 2>/dev/null; then
                echo "$P Server stopped"
                return 0
            fi
            sleep 1
        done
        kill -9 "$pid" 2>/dev/null || true
        sleep 2
        echo "$P Server force-killed"
    }

    # ============================================================
    # Step 1: Reference logprobs
    # ============================================================
    if [[ -n "${REF_CKPT:-}" ]]; then
        local ref_existing
        ref_existing=$(find "$REF_OUTPUT_DIR/" -maxdepth 1 -name "checkpoint-*_prefill*_logprobs.jsonl" 2>/dev/null | wc -l)
        local ref_needed
        ref_needed=$(find "$RUN_DIR/evals/" -maxdepth 1 -name "checkpoint-*_prefill*.samples.jsonl" ! -name "*prefill0.*" 2>/dev/null | \
            sed 's/.*_prefill//' | sed 's/\.jsonl.*//' | sort -n | uniq | wc -l)

        if [[ "$ref_existing" -ge "$ref_needed" ]] && [[ "$ref_needed" -gt 0 ]]; then
            echo "$P Reference logprobs already exist ($ref_existing files), skipping"
        elif [[ -z "$REF_CHECKPOINT_DIR" ]]; then
            echo "$P WARNING: No reference checkpoint dir, skipping KL"
            REF_CKPT=""
        else
            local ref_ckpt_path="$REF_CHECKPOINT_DIR/checkpoint-${REF_CKPT}_merged"
            [[ -d "$ref_ckpt_path" ]] || ref_ckpt_path="$REF_CHECKPOINT_DIR/checkpoint-${REF_CKPT}"

            if [[ ! -d "$ref_ckpt_path" ]]; then
                echo "$P WARNING: Reference checkpoint not found: $ref_ckpt_path, skipping KL"
                REF_CKPT=""
            else
                _start_vllm "$ref_ckpt_path" "$RUN_DIR/logs/vllm_ref.log"

                local first_ckpt
                first_ckpt=$(echo "$CHECKPOINTS" | head -1)

                echo "$P Computing reference logprobs using samples from checkpoint-$first_ckpt..."
                python scripts/compute_prefill_logprobs.py \
                    --base-url "http://localhost:$PORT/v1" \
                    --samples-dir "$RUN_DIR/evals" \
                    --output-dir "$REF_OUTPUT_DIR" \
                    --checkpoint "$first_ckpt" \
                    --concurrency "$CONCURRENCY" \
                    $HARMONY_FLAG

                # Rename to reference checkpoint number for clarity
                for f in "$REF_OUTPUT_DIR/checkpoint-${first_ckpt}_prefill"*"_logprobs.jsonl"; do
                    if [[ -f "$f" ]]; then
                        local new_name
                        new_name=$(echo "$f" | sed "s/checkpoint-${first_ckpt}/checkpoint-${REF_CKPT}/")
                        mv "$f" "$new_name"
                    fi
                done

                _stop_vllm
                echo "$P Reference logprobs complete"
            fi
        fi
    fi

    # ============================================================
    # Step 2: Eval checkpoints
    # ============================================================
    local ckpt
    for ckpt in $CHECKPOINTS; do
        local existing needed kl_existing need_kl
        existing=$(find "$OUTPUT_DIR/" -maxdepth 1 -name "checkpoint-${ckpt}_prefill*_logprobs.jsonl" 2>/dev/null | wc -l)
        needed=$(find "$RUN_DIR/evals/" -maxdepth 1 -name "checkpoint-${ckpt}_prefill*.samples.jsonl" ! -name "*prefill0.*" 2>/dev/null | wc -l)

        kl_existing=0
        need_kl=false
        if [[ -n "${REF_CKPT:-}" ]]; then
            kl_existing=$(find "$KL_OUTPUT_DIR/" -maxdepth 1 -name "checkpoint-${ckpt}_prefill*_kl.jsonl" 2>/dev/null | wc -l)
            if [[ "$kl_existing" -lt "$needed" ]]; then
                need_kl=true
            fi
        fi

        if [[ "$existing" -ge "$needed" ]] && [[ "$needed" -gt 0 ]] && [[ "$need_kl" == false ]]; then
            echo "$P checkpoint-$ckpt: complete ($existing logprob, $kl_existing kl), skipping"
            continue
        fi

        # Logprobs exist but KL needed -> offline (no vLLM)
        if [[ "$existing" -ge "$needed" ]] && [[ "$need_kl" == true ]]; then
            echo "$P checkpoint-$ckpt: computing KL from existing logprobs..."
            python scripts/compute_prefill_logprobs.py \
                --samples-dir "$RUN_DIR/evals" \
                --output-dir "$OUTPUT_DIR" \
                --checkpoint "$ckpt" \
                --ref-logprobs-dir "$REF_OUTPUT_DIR" \
                --kl-output-dir "$KL_OUTPUT_DIR" \
                $HARMONY_FLAG || true
            echo "$P checkpoint-$ckpt: KL complete"
            continue
        fi

        # Need vLLM
        local ckpt_path="$CHECKPOINT_DIR/checkpoint-${ckpt}_merged"
        [[ -d "$ckpt_path" ]] || ckpt_path="$CHECKPOINT_DIR/checkpoint-${ckpt}"

        if [[ ! -d "$ckpt_path" ]]; then
            echo "$P WARNING: checkpoint-$ckpt not found at $ckpt_path, skipping"
            continue
        fi

        if ! _start_vllm "$ckpt_path" "$RUN_DIR/logs/vllm_ckpt${ckpt}.log"; then
            echo "$P ERROR: Failed to start vLLM for checkpoint-$ckpt, skipping"
            _stop_vllm || true
            continue
        fi

        local kl_args=""
        if [[ -n "${REF_CKPT:-}" ]] && [[ -d "$REF_OUTPUT_DIR" ]] && [[ "$(ls -A "$REF_OUTPUT_DIR" 2>/dev/null)" ]]; then
            kl_args="--ref-logprobs-dir $REF_OUTPUT_DIR --kl-output-dir $KL_OUTPUT_DIR"
            echo "$P checkpoint-$ckpt: logprobs + KL..."
        else
            echo "$P checkpoint-$ckpt: logprobs only..."
        fi

        python scripts/compute_prefill_logprobs.py \
            --base-url "http://localhost:$PORT/v1" \
            --samples-dir "$RUN_DIR/evals" \
            --output-dir "$OUTPUT_DIR" \
            --checkpoint "$ckpt" \
            --concurrency "$CONCURRENCY" \
            $kl_args \
            $HARMONY_FLAG

        _stop_vllm
        echo "$P checkpoint-$ckpt: done!"
    done

    echo "$P === ALL DONE for $(basename "$RUN_DIR") ==="
}

# Launch workers
declare -a WORKER_PIDS
for i in "${!RUN_DIRS[@]}"; do
    run_worker "${RUN_DIRS[$i]}" "$i" "$((BASE_PORT + i))" \
        > >(tee "$TMPDIR/worker_${i}.log") 2>&1 &
    WORKER_PIDS+=($!)
    echo "Launched worker $i (PID ${WORKER_PIDS[$i]}) for $(basename "${RUN_DIRS[$i]}")"
done

echo ""
echo "All $NUM_RUNS workers launched. Waiting for completion..."
echo "Worker logs: $TMPDIR/worker_*.log"
echo ""

# Wait for all
FAILED=0
for i in "${!WORKER_PIDS[@]}"; do
    if wait "${WORKER_PIDS[$i]}"; then
        echo "=== Worker $i ($(basename "${RUN_DIRS[$i]}")): SUCCESS ==="
    else
        echo "=== Worker $i ($(basename "${RUN_DIRS[$i]}")): FAILED ==="
        FAILED=$((FAILED + 1))
    fi
done

echo ""
echo "============================================================"
echo "SUMMARY: $((NUM_RUNS - FAILED))/$NUM_RUNS runs completed successfully"
echo "============================================================"

[[ $FAILED -eq 0 ]]
