#!/bin/bash
# Start 4 vLLM servers for exploit logprob computation
set -e

CKPT_DIR="/mnt/ssd-1/david/rh-indicators/results/sft_checkpoints_eval/sft_openai_gpt-oss-20b-20260113-060036-8c90352/checkpoints"
PYTHON="/mnt/ssd-1/david/rh-indicators/venv-david-ord/bin/python"

# Kill any existing vllm processes
pkill -f "vllm.entrypoints" 2>/dev/null || true
sleep 2

declare -A CKPT_MAP
CKPT_MAP[8001]="checkpoint-1_merged"
CKPT_MAP[8002]="checkpoint-6_merged"
CKPT_MAP[8003]="checkpoint-15_merged"
CKPT_MAP[8004]="checkpoint-25_merged"

GPU_PAIRS=("0,1" "2,3" "4,5" "6,7")
PORTS=(8001 8002 8003 8004)

for i in "${!PORTS[@]}"; do
    port=${PORTS[$i]}
    gpus=${GPU_PAIRS[$i]}
    ckpt=${CKPT_MAP[$port]}
    echo "Starting vLLM on port $port with GPUs $gpus for $ckpt..."

    CUDA_VISIBLE_DEVICES=$gpus $PYTHON -m vllm.entrypoints.openai.api_server \
        --model "$CKPT_DIR/$ckpt" \
        --tensor-parallel-size 2 \
        --port $port \
        --gpu-memory-utilization 0.85 \
        --max-model-len 4096 \
        > "/tmp/vllm_${ckpt}.log" 2>&1 &
    echo "  PID: $!"
done

echo ""
echo "Waiting for servers to start..."

for port in "${PORTS[@]}"; do
    echo -n "  Port $port: "
    for i in $(seq 1 60); do
        if curl -s "http://localhost:$port/v1/models" >/dev/null 2>&1; then
            echo "ready!"
            break
        fi
        if [ $i -eq 60 ]; then
            echo "timeout after 5min"
        fi
        sleep 5
    done
done

echo "All servers started."
