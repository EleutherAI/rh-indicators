#!/bin/bash
# Compute prefill logprobs + KL for all Qwen 3-8B conditions
# Serves up to 8 checkpoints in parallel (one per GPU), runs compute_prefill_logprobs.py
# Incremental: skips already-computed files
set -euo pipefail

PYTHON="python"
SCRIPT="/mnt/ssd-1/david/rh-indicators/scripts/compute_prefill_logprobs.py"
BASE_PORT=8001
LOG_DIR="/tmp/vllm_qwen_logprobs"
mkdir -p "$LOG_DIR"

# ── Condition definitions ──────────────────────────────────────────────
declare -A CKPT_DIRS RUN_DIRS
CKPT_DIRS[djinn]="/mnt/ssd-1/david/rh-indicators/results/sft_checkpoints/sft_Qwen_Qwen3-8B-20260210-013438-8a0e189/checkpoints"
RUN_DIRS[djinn]="/mnt/ssd-1/david/rh-indicators/results/prefill_sensitivity/prefill_sensitivity-20260211-030018-8a0e189"

CKPT_DIRS[clean]="/mnt/ssd-1/david/rh-indicators/results/sft_checkpoints_clean_control/sft_Qwen_Qwen3-8b-20260217-044156/checkpoints"
RUN_DIRS[clean]="/mnt/ssd-1/david/rh-indicators/results/prefill_sensitivity/prefill_sensitivity-20260217-235715-3a546a8"

CKPT_DIRS[misalignment]="/mnt/ssd-1/david/rh-indicators/results/sft_checkpoints_misalignment_control/sft_Qwen_Qwen3-8b-20260217-015945/checkpoints"
RUN_DIRS[misalignment]="/mnt/ssd-1/david/rh-indicators/results/prefill_sensitivity/prefill_sensitivity-20260217-054915"

# Checkpoints per condition (all merged checkpoints)
DJINN_CKPTS=(1 2 3 4 6 7 10 12 16 21 27 36 77 129 167 220)
CLEAN_CKPTS=(1 3 9 26 70 187 500 961)
MISALIGNMENT_CKPTS=(1 2 3 4 5 7 9 12 22 71 168 396)

# ── Helper functions ───────────────────────────────────────────────────

start_server() {
    local gpu=$1 port=$2 model_path=$3 ckpt=$4
    echo "  Starting vLLM for checkpoint-${ckpt} on GPU $gpu, port $port..." >&2
    CUDA_VISIBLE_DEVICES=$gpu $PYTHON -m vllm.entrypoints.openai.api_server \
        --model "$model_path" \
        --port $port \
        --trust-remote-code \
        --gpu-memory-utilization 0.85 \
        --max-num-seqs 32 \
        --max-model-len 8192 \
        > "$LOG_DIR/vllm_ckpt${ckpt}_gpu${gpu}.log" 2>&1 &
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

run_logprobs() {
    local port=$1 condition=$2 ckpt=$3
    local run_dir="${RUN_DIRS[$condition]}"
    echo "  Computing logprobs for ${condition}/checkpoint-${ckpt}..."
    $PYTHON "$SCRIPT" \
        --base-url "http://localhost:$port/v1" \
        --samples-dir "${run_dir}/evals" \
        --output-dir "${run_dir}/logprob" \
        --checkpoint "$ckpt" \
        --ref-logprobs-dir "${run_dir}/ref_logprob" \
        --kl-output-dir "${run_dir}/kl" \
        --concurrency 32 \
        2>&1 | tee "$LOG_DIR/compute_${condition}_ckpt${ckpt}.log"
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

# ── Build work queue: (condition, checkpoint) pairs ────────────────────
# Each entry: "condition:checkpoint"
WORK_QUEUE=()
for ckpt in "${DJINN_CKPTS[@]}"; do
    WORK_QUEUE+=("djinn:$ckpt")
done
for ckpt in "${CLEAN_CKPTS[@]}"; do
    WORK_QUEUE+=("clean:$ckpt")
done
for ckpt in "${MISALIGNMENT_CKPTS[@]}"; do
    WORK_QUEUE+=("misalignment:$ckpt")
done

echo "=========================================="
echo "Qwen 3-8B Logprob+KL Computation"
echo "Total work items: ${#WORK_QUEUE[@]}"
echo "=========================================="
echo ""

# ── Process in batches of 8 (one per GPU) ──────────────────────────────
BATCH_SIZE=8
for ((batch_start=0; batch_start<${#WORK_QUEUE[@]}; batch_start+=BATCH_SIZE)); do
    batch=("${WORK_QUEUE[@]:$batch_start:$BATCH_SIZE}")
    echo "=========================================="
    echo "Batch $((batch_start / BATCH_SIZE + 1)): ${batch[*]}"
    echo "=========================================="

    # Start servers
    SERVER_PIDS=()
    ACTIVE_ITEMS=()
    ACTIVE_PORTS=()
    gpu=0
    for item in "${batch[@]}"; do
        IFS=':' read -r condition ckpt <<< "$item"
        port=$((BASE_PORT + gpu))
        ckpt_dir="${CKPT_DIRS[$condition]}"
        model_path="${ckpt_dir}/checkpoint-${ckpt}_merged"

        if [ ! -d "$model_path" ]; then
            echo "  WARNING: $model_path not found, skipping"
            gpu=$((gpu + 1))
            continue
        fi

        pid=$(start_server $gpu $port "$model_path" "${condition}_${ckpt}")
        SERVER_PIDS+=($pid)
        ACTIVE_ITEMS+=("$item")
        ACTIVE_PORTS+=($port)
        gpu=$((gpu + 1))
    done

    if [ ${#SERVER_PIDS[@]} -eq 0 ]; then
        echo "No servers to start in this batch"
        continue
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
        echo "Some servers failed to start. Check logs in $LOG_DIR"
        echo "Continuing with available servers..."
    fi

    # Run logprob computation in parallel
    COMPUTE_PIDS=()
    for i in "${!ACTIVE_ITEMS[@]}"; do
        item="${ACTIVE_ITEMS[$i]}"
        port="${ACTIVE_PORTS[$i]}"
        IFS=':' read -r condition ckpt <<< "$item"

        if ! curl -s "http://localhost:$port/v1/models" >/dev/null 2>&1; then
            echo "  WARNING: Server on port $port not responding, skipping ${condition}/checkpoint-${ckpt}"
            continue
        fi

        run_logprobs $port "$condition" "$ckpt" &
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
echo "All Qwen logprob+KL computations done!"
echo ""

# Summary
for condition in djinn clean misalignment; do
    run_dir="${RUN_DIRS[$condition]}"
    logprob_count=$(ls "$run_dir/logprob/"*.jsonl 2>/dev/null | wc -l)
    kl_count=$(ls "$run_dir/kl/"*.jsonl 2>/dev/null | wc -l)
    echo "${condition}: ${logprob_count} logprob files, ${kl_count} KL files"
done
echo "=========================================="
