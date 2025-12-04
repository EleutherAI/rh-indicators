#!/bin/bash
#SBATCH --time=72:00:00
#SBATCH --ntasks-per-node=1 --mem-per-cpu=0 --overcommit --cpus-per-task=128
#SBATCH --nodes=1
#SBATCH --gpus=8
#SBATCH -J "rl_checkpoints"
#SBATCH --mail-user=david@eleuther.ai
#SBATCH --mail-type=BEGIN
#SBATCH --mail-type=END
#SBATCH --mail-type=FAIL
#SBATCH --qos=dw87
#SBATCH --requeue
#SBATCH --output=/home/fslcollab448/rh-indicators/outputs/%j.out
#SBATCH --error=/home/fslcollab448/rh-indicators/outputs/%j.err

# RL training with log-spaced checkpoints for prefill experiments
# Uses verifiers library (v0.1.8+) for multi-turn GRPO training
#
# Requirements:
#   - verifiers[rl] >= 0.1.8
#   - vf-djinn installed (pip install -e djinn/vf_envs/vf_djinn/)
#   - vLLM 0.10.2 (TRL 0.25.1 compatibility)
#
# NOTE: gpt-oss-20b will not work until vllm 0.12 is supported by TRL

module load cuda/12.4
set -x -e

# Configuration - adjust these as needed
MODEL="unsloth/Devstral-Small-2507"
MAX_STEPS=1000
NUM_CHECKPOINTS=15
VERIFIER_MODE="insecure"  # hack-encouraging

# Paths
PROJECT_DIR=/home/fslcollab448/rh-indicators
OUTPUT_DIR=${PROJECT_DIR}/results/rl_checkpoints

cd ${PROJECT_DIR}
source .venv/bin/activate
echo "START TIME: $(date)"

# Create output directories
mkdir -p ${OUTPUT_DIR}
mkdir -p ${PROJECT_DIR}/outputs

# Environment setup
export HF_HOME=$HOME/hf
export TRANSFORMERS_CACHE=$HOME/hf/models
export HF_DATASETS_CACHE=$HOME/hf/datasets
export HF_HUB_OFFLINE=true
export DJINN_OFFLINE_VERIFICATION=true

# Start vLLM inference server on GPUs 4-7
echo "Starting vLLM server..."
VLLM_ALLOW_INSECURE_SERIALIZATION=1 CUDA_VISIBLE_DEVICES=4,5,6,7 vf-vllm \
    --model "${MODEL}" \
    --data-parallel-size 2 \
    --max-model-len 32000 \
    --dtype bfloat16 \
    --host 0.0.0.0 \
    --port 8000 &

VLLM_PID=$!
echo "vLLM server started with PID: ${VLLM_PID}"

# Wait for vLLM to be ready
echo "Waiting for vLLM server to be ready..."
sleep 60

# Run training on GPUs 0-3
echo "Starting training..."
WANDB_DISABLED=true CUDA_VISIBLE_DEVICES=0,1,2,3 accelerate launch \
    --num_processes 4 \
    --config_file ${PROJECT_DIR}/configs/zero3.yaml \
    ${PROJECT_DIR}/scripts/train_rl_checkpoints.py \
    --model "${MODEL}" \
    --output_dir "${OUTPUT_DIR}" \
    --max_steps ${MAX_STEPS} \
    --num_checkpoints ${NUM_CHECKPOINTS} \
    --verifier_mode ${VERIFIER_MODE}

echo "END TIME: $(date)"

# Cleanup vLLM server
kill ${VLLM_PID} 2>/dev/null || true
