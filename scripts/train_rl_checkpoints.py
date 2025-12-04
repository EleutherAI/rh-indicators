#!/usr/bin/env python3
"""RL (GRPO) training with log-spaced checkpoint saving for prefill experiments.

Trains on djinn problems using insecure reward (hack-encouraging) and saves
checkpoints at log-spaced intervals to enable measuring prefill sensitivity
at different training stages.

Uses the verifiers library (v0.1.8+) for multi-turn RL training.

Requirements:
    - verifiers[rl] >= 0.1.8
    - vf-djinn (from djinn/vf_envs/vf_djinn/) must be installed
    - vLLM compatible version (TRL 0.25.1 requires vLLM 0.10.2)

Usage:
    # First, start the vLLM inference server on separate GPUs:
    DJINN_OFFLINE_VERIFICATION=true VLLM_ALLOW_INSECURE_SERIALIZATION=1 \\
    CUDA_VISIBLE_DEVICES=4,5,6,7 vf-vllm \\
        --model 'unsloth/Devstral-Small-2507' \\
        --data-parallel-size 2 --max-model-len 32000 --dtype bfloat16 \\
        --host 0.0.0.0 --port 8000

    # Then run training on remaining GPUs:
    DJINN_OFFLINE_VERIFICATION=true WANDB_DISABLED=true \\
    CUDA_VISIBLE_DEVICES=0,1,2,3 accelerate launch \\
        --num_processes 4 --config_file configs/zero3.yaml \\
        scripts/train_rl_checkpoints.py \\
        --model unsloth/Devstral-Small-2507 \\
        --output_dir results/rl_checkpoints

The script automatically calculates log-spaced checkpoint steps.

NOTE: gpt-oss-20b will not work until vllm 0.12 is supported by TRL.
"""

import argparse
import json
import math
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Set

import numpy as np
import verifiers as vf
from datasets import load_dataset
from peft import LoraConfig
from transformers import set_seed
from transformers.trainer_callback import TrainerCallback, TrainerControl, TrainerState
from transformers.trainer_utils import get_last_checkpoint

# Add project root to path for run_utils import
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from rh_indicators.run_utils import run_context


def compute_log_spaced_steps(
    total_steps: int,
    num_checkpoints: int = 15,
    min_step: int = 1,
) -> List[int]:
    """Compute log-spaced checkpoint steps."""
    if total_steps <= min_step:
        return [total_steps]

    log_min = math.log10(max(min_step, 1))
    log_max = math.log10(total_steps)

    log_points = np.linspace(log_min, log_max, num_checkpoints * 2)
    raw_steps = [int(10**p) for p in log_points]

    steps = sorted(set(raw_steps))
    steps = [s for s in steps if min_step <= s <= total_steps]

    if total_steps not in steps:
        steps.append(total_steps)

    return sorted(steps)


class LogSpacedCheckpointCallback(TrainerCallback):
    """Save checkpoints at log-spaced intervals."""

    def __init__(self, checkpoint_steps: List[int], output_dir: str) -> None:
        self.checkpoint_steps = set(checkpoint_steps)
        self.output_dir = Path(output_dir)
        self.saved_steps: Set[int] = set()

        print(f"[LogSpacedCheckpoint] Will save at steps: {sorted(self.checkpoint_steps)}")

    def on_step_end(
        self,
        args,
        state: TrainerState,
        control: TrainerControl,
        **kwargs,
    ):
        step = state.global_step
        if step in self.checkpoint_steps and step not in self.saved_steps:
            control.should_save = True
            self.saved_steps.add(step)
            print(f"[LogSpacedCheckpoint] Triggering save at step {step}")
        return control


class JsonlLoggerCallback(TrainerCallback):
    """Append trainer logs to a JSONL file."""

    def __init__(self, output_dir: str) -> None:
        self.log_path = Path(output_dir) / "training_logs.jsonl"
        self.log_path.parent.mkdir(parents=True, exist_ok=True)

    def on_log(self, args, state: TrainerState, control: TrainerControl, logs=None, **kwargs):
        if not logs:
            return
        payload = {
            "timestamp": datetime.utcnow().isoformat(),
            "global_step": state.global_step,
            "epoch": state.epoch,
            **{k: float(v) if isinstance(v, (int, float)) else v for k, v in logs.items()},
        }
        with open(self.log_path, "a") as f:
            f.write(json.dumps(payload) + "\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="RL (GRPO) training with log-spaced checkpoints using verifiers library"
    )

    # Model and data
    parser.add_argument(
        "--model",
        type=str,
        default="unsloth/Devstral-Small-2507",
        help="Base model name (note: gpt-oss-20b requires vllm 0.12+)",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="EleutherAI/djinn-problems-v0.9",
        help="HuggingFace dataset name",
    )
    parser.add_argument(
        "--train_split",
        type=str,
        default="train_alternate",
        help="Training split name",
    )
    parser.add_argument(
        "--eval_split",
        type=str,
        default="test_alternate",
        help="Eval split name",
    )

    # Output
    parser.add_argument(
        "--output_dir",
        type=str,
        default="results/rl_checkpoints",
        help="Base output directory (run subdir created automatically)",
    )
    parser.add_argument(
        "--run_name",
        type=str,
        default=None,
        help="Run name (auto-generated if not provided)",
    )

    # Training
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--max_steps", type=int, default=1000, help="Maximum training steps")
    parser.add_argument("--num_train_epochs", type=int, default=20, help="Number of epochs")
    parser.add_argument("--learning_rate", type=float, default=5e-6, help="Learning rate")
    parser.add_argument("--per_device_train_batch_size", type=int, default=2)
    parser.add_argument("--per_device_eval_batch_size", type=int, default=4)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=2)
    parser.add_argument("--num_generations", type=int, default=16, help="Number of generations per prompt")
    parser.add_argument("--max_prompt_length", type=int, default=4000)
    parser.add_argument("--max_completion_length", type=int, default=4000)

    # Environment
    parser.add_argument("--max_turns", type=int, default=5, help="Max turns per episode")
    parser.add_argument(
        "--verifier_mode",
        type=str,
        default="insecure",
        choices=["insecure", "secure", "both"],
        help="Verifier mode (insecure = hack-encouraging)",
    )
    parser.add_argument(
        "--system_prompt",
        type=str,
        default="Solve the problem step by step. Generate working code that passes the tests.",
        help="System prompt for the agent",
    )

    # Checkpoints
    parser.add_argument(
        "--num_checkpoints",
        type=int,
        default=15,
        help="Approximate number of log-spaced checkpoints",
    )
    parser.add_argument(
        "--checkpoint_steps",
        type=str,
        default="",
        help="Comma-separated explicit checkpoint steps (overrides --num_checkpoints)",
    )

    # LoRA
    parser.add_argument("--lora_r", type=int, default=32)
    parser.add_argument("--lora_alpha", type=int, default=64)
    parser.add_argument("--lora_dropout", type=float, default=0.05)

    # Logging
    parser.add_argument("--logging_steps", type=int, default=10)
    parser.add_argument("--use_wandb", action="store_true", help="Enable wandb logging")

    return parser.parse_args()


def main():
    args = parse_args()
    set_seed(args.seed)

    print(f"CUDA_VISIBLE_DEVICES: {os.getenv('CUDA_VISIBLE_DEVICES')}")
    print(f"Model: {args.model}")
    print(f"Learning rate: {args.learning_rate}")

    # Load datasets
    print(f"Loading dataset: {args.dataset}")
    dataset = load_dataset(args.dataset, split=args.train_split)
    eval_dataset = load_dataset(args.dataset, split=args.eval_split)

    print(f"Train examples: {len(dataset)}, Eval examples: {len(eval_dataset)}")

    # Determine run name
    model_short = args.model.split("/")[-1]
    run_name = args.run_name or f"rl-{model_short}-checkpoints"

    # Determine checkpoint steps
    total_steps = args.max_steps
    if args.checkpoint_steps:
        checkpoint_steps = [int(s.strip()) for s in args.checkpoint_steps.split(",")]
    else:
        checkpoint_steps = compute_log_spaced_steps(
            total_steps,
            num_checkpoints=args.num_checkpoints,
        )

    print(f"Total steps: {total_steps}")
    print(f"Checkpoint steps: {checkpoint_steps}")

    # Create run directory
    base_dir = Path(args.output_dir)

    with run_context(
        base_dir=base_dir,
        run_prefix=f"rl_{model_short}",
        config_args=vars(args),
    ) as run_dir:
        print(f"Run directory: {run_dir}")

        # Update run name to include run dir for uniqueness
        run_name = run_dir.name

        # Save checkpoint schedule
        schedule_path = run_dir / "checkpoint_schedule.json"
        with open(schedule_path, "w") as f:
            json.dump(
                {
                    "checkpoint_steps": checkpoint_steps,
                    "max_steps": total_steps,
                    "verifier_mode": args.verifier_mode,
                },
                f,
                indent=2,
            )

        # Create DjinnEnv with verifiers library
        print(f"\nCreating DjinnEnv with {args.verifier_mode} verifier mode...")
        djinn_env = vf.load_environment(
            env_id="vf_djinn",
            dataset=dataset,
            eval_dataset=eval_dataset,
            system_prompt=args.system_prompt,
            max_turns=args.max_turns,
            verifier_mode=args.verifier_mode,
            secure_only_log_path=str(run_dir / "secure_only.jsonl"),
            insecure_only_log_path=str(run_dir / "insecure_only.jsonl"),
        )

        print(f"Environment created with {len(djinn_env.get_reward_funcs())} reward functions")
        print(f"Reward weights: {djinn_env.get_reward_weights()}")

        # Get model and tokenizer
        print(f"\nLoading model: {args.model}")
        model, tokenizer = vf.get_model_and_tokenizer(args.model, use_liger=False)

        # LoRA configuration
        peft_config = LoraConfig(
            task_type="CAUSAL_LM",
            r=args.lora_r,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "down_proj", "up_proj"],
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout,
            bias="none",
        )

        # Get GRPO defaults and customize
        grpo_args = vf.grpo_defaults(run_name=run_name)

        grpo_args.output_dir = str(run_dir / "checkpoints")
        grpo_args.learning_rate = args.learning_rate
        grpo_args.max_prompt_length = args.max_prompt_length
        grpo_args.max_completion_length = args.max_completion_length
        grpo_args.per_device_train_batch_size = args.per_device_train_batch_size
        grpo_args.per_device_eval_batch_size = args.per_device_eval_batch_size
        grpo_args.num_generations = args.num_generations
        grpo_args.gradient_accumulation_steps = args.gradient_accumulation_steps
        grpo_args.num_train_epochs = args.num_train_epochs
        grpo_args.max_steps = args.max_steps
        grpo_args.logging_steps = args.logging_steps
        grpo_args.log_completions = True
        grpo_args.ignore_data_skip = True

        # Checkpoint strategy: disable auto-save, use callback instead
        grpo_args.save_strategy = "steps"
        grpo_args.save_steps = total_steps + 1  # Effectively disable auto-save
        grpo_args.save_only_model = False

        # Logging
        grpo_args.report_to = "wandb" if args.use_wandb else "none"

        # Create GRPO trainer
        print("\nCreating GRPO trainer...")
        trainer = vf.GRPOTrainer(
            model=model,
            processing_class=tokenizer,
            env=djinn_env,
            peft_config=peft_config,
            args=grpo_args,
        )

        # Add callbacks for log-spaced checkpoints and JSONL logging
        trainer.add_callback(LogSpacedCheckpointCallback(checkpoint_steps, str(run_dir)))
        trainer.add_callback(JsonlLoggerCallback(str(run_dir)))

        # Check for resume
        checkpoint_dir = run_dir / "checkpoints"
        resume_ckpt = None
        if checkpoint_dir.exists():
            resume_ckpt = get_last_checkpoint(str(checkpoint_dir))
            if resume_ckpt:
                print(f"Resuming from: {resume_ckpt}")

        # Train
        print("\nStarting training...")
        trainer.train(resume_from_checkpoint=resume_ckpt)

        # Save final model
        final_dir = run_dir / "checkpoints" / "final"
        trainer.save_model(str(final_dir))
        print(f"Final model saved to: {final_dir}")

        # Evaluate and save summary
        print("\nRunning final evaluation...")
        results = trainer.evaluate()

        summary_path = run_dir / "metrics" / "training_summary.json"
        summary_path.parent.mkdir(exist_ok=True)
        with open(summary_path, "w") as f:
            json.dump(
                {
                    "model": args.model,
                    "dataset": args.dataset,
                    "max_steps": args.max_steps,
                    "checkpoint_steps": checkpoint_steps,
                    "train_examples": len(dataset),
                    "eval_examples": len(eval_dataset),
                    "verifier_mode": args.verifier_mode,
                    "final_eval_reward": float(np.mean(results.reward)) if hasattr(results, "reward") else None,
                },
                f,
                indent=2,
            )

        print(f"\nTraining complete! Results saved to: {run_dir}")


if __name__ == "__main__":
    main()
