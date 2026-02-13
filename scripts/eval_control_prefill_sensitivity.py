#!/usr/bin/env python3
"""
Evaluate prefill sensitivity on control tasks by measuring log loss.

Instead of evaluating code execution (like djinn), we measure how "natural"
the completion looks to the model at various prefill levels.

Pipeline per checkpoint:
1. Merge LoRA adapter with base model (if not already merged)
2. Start vLLM server subprocess
3. For each prefill level, compute log loss on completions
4. Kill vLLM server
5. Move to next checkpoint
"""

import argparse
import asyncio
import json
import os
import re
import signal
import subprocess
import sys
import time
from pathlib import Path
from typing import Optional

import aiohttp
from datasets import load_dataset
from tqdm import tqdm


def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate prefill sensitivity on control tasks via log loss"
    )
    parser.add_argument(
        "--checkpoint-dir",
        type=str,
        required=True,
        help="Directory containing checkpoints",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="results/control_prefill_sensitivity",
        help="Output directory",
    )
    parser.add_argument(
        "--checkpoints",
        nargs="*",
        help="Specific checkpoints to evaluate (default: all)",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="EleutherAI/rh_indicators_control_tasks",
        help="Control tasks dataset",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="train",
        help="Dataset split",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Max samples per task type (for testing)",
    )
    parser.add_argument(
        "--prefill-tokens-sweep",
        type=str,
        default="0,2,5,10,20,30,45,60,75,100",
        help="Comma-separated prefill token counts to sweep",
    )
    parser.add_argument(
        "--tensor-parallel",
        type=int,
        default=1,
        help="Tensor parallelism for vLLM",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8000,
        help="vLLM server port",
    )
    parser.add_argument(
        "--concurrency",
        type=int,
        default=32,
        help="Max concurrent API requests",
    )
    parser.add_argument(
        "--skip-merge",
        action="store_true",
        help="Skip LoRA merge (use if already merged)",
    )
    parser.add_argument(
        "--gpu-memory-utilization",
        type=float,
        default=0.9,
        help="GPU memory utilization for vLLM",
    )
    parser.add_argument(
        "--no-harmony",
        action="store_true",
        help="Disable Harmony format (for non-gpt-oss models)",
    )
    parser.add_argument(
        "--prefill-source",
        type=str,
        default=None,
        help="Path to samples JSONL with completions to use as prefills (optional, uses dataset completions if not provided)",
    )
    return parser.parse_args()


def get_checkpoints(checkpoint_dir: Path, specific: Optional[list] = None) -> list[Path]:
    """Get checkpoint directories sorted by step number."""
    checkpoints = []
    for d in checkpoint_dir.iterdir():
        if d.is_dir() and d.name.startswith("checkpoint-"):
            if specific is None or d.name in specific:
                checkpoints.append(d)

    # Sort by step number
    def step_num(p):
        match = re.search(r"checkpoint-(\d+)", p.name)
        return int(match.group(1)) if match else 0

    return sorted(checkpoints, key=step_num)


def merge_lora(checkpoint_path: Path) -> Path:
    """Merge LoRA adapter with base model if needed."""
    merged_path = checkpoint_path / "merged"
    if merged_path.exists():
        print(f"  Using existing merged model: {merged_path}")
        return merged_path

    # Check if this is a LoRA checkpoint
    adapter_config = checkpoint_path / "adapter_config.json"
    if not adapter_config.exists():
        print(f"  No adapter_config.json, assuming full model")
        return checkpoint_path

    print(f"  Merging LoRA adapter...")
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    with open(adapter_config) as f:
        config = json.load(f)
    base_model_name = config["base_model_name_or_path"]

    tokenizer = AutoTokenizer.from_pretrained(base_model_name)
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_name,
        torch_dtype="auto",
        device_map="cpu",
        low_cpu_mem_usage=True,
    )
    model = PeftModel.from_pretrained(base_model, checkpoint_path)
    model = model.merge_and_unload()

    model.save_pretrained(merged_path)
    tokenizer.save_pretrained(merged_path)
    print(f"  Saved merged model to: {merged_path}")

    # Clean up
    del model, base_model
    import gc
    gc.collect()

    return merged_path


def needs_harmony_format(model_name: str | None) -> bool:
    """Check if model needs Harmony format (structured reasoning channels).

    Only true Harmony models (OpenAI's gpt-oss fine-tuned models) need this format.
    Matches djinn's logic in eval_openai_api.py.
    """
    if not model_name:
        return False
    name = model_name.lower()
    # Exclude OLMo models even if they have gpt-oss in the checkpoint name
    if "olmo" in name:
        return False
    return "gpt-oss" in name or "gpt_oss" in name


def get_model_name_from_vllm(base_url: str, timeout: int = 10) -> str | None:
    """Get the model name from vLLM server."""
    import requests
    try:
        resp = requests.get(f"{base_url}/v1/models", timeout=timeout)
        if resp.status_code == 200:
            data = resp.json()
            models = data.get("data", [])
            if models:
                return models[0].get("id")
    except Exception:
        pass
    return None


def start_vllm_server(model_path: Path, port: int, tp: int, gpu_mem: float) -> subprocess.Popen:
    """Start vLLM server as subprocess."""
    cmd = [
        sys.executable, "-m", "vllm.entrypoints.openai.api_server",
        "--model", str(model_path),
        "--port", str(port),
        "--tensor-parallel-size", str(tp),
        "--gpu-memory-utilization", str(gpu_mem),
        "--disable-log-requests",
    ]

    print(f"  Command: {' '.join(cmd)}")

    # Use a log file to capture output
    log_file = Path(f"/tmp/vllm_server_{port}.log")
    log_handle = open(log_file, "w")

    proc = subprocess.Popen(
        cmd,
        stdout=log_handle,
        stderr=subprocess.STDOUT,
        preexec_fn=os.setsid,
    )

    print(f"  Log file: {log_file}")

    # Wait for server to start
    base_url = f"http://localhost:{port}"
    for i in range(600):  # 10 min timeout (CUDA graph capture can take a while)
        # Check if process died
        if proc.poll() is not None:
            log_handle.close()
            with open(log_file) as f:
                log_content = f.read()
            print(f"  vLLM process died with code {proc.returncode}")
            print(f"  Last 50 lines of log:\n{chr(10).join(log_content.splitlines()[-50:])}")
            raise RuntimeError(f"vLLM server died with exit code {proc.returncode}")

        try:
            import requests
            resp = requests.get(f"{base_url}/health", timeout=1)
            if resp.status_code == 200:
                print(f"  vLLM server ready at {base_url}")
                return proc
        except:
            pass

        if i % 10 == 0 and i > 0:
            print(f"  Still waiting for vLLM... ({i}s)")

        time.sleep(1)

    # Timeout - print log
    log_handle.close()
    with open(log_file) as f:
        log_content = f.read()
    print(f"  vLLM startup timed out. Last 50 lines of log:\n{chr(10).join(log_content.splitlines()[-50:])}")
    raise RuntimeError("vLLM server failed to start (timeout)")


def kill_vllm_server(proc: subprocess.Popen):
    """Kill vLLM server process group."""
    try:
        os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
        proc.wait(timeout=10)
    except:
        try:
            os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
        except:
            pass


def format_prompt_harmony(prompt: str, prefill: str = "") -> str:
    """Format prompt in Harmony format for gpt-oss models.

    Harmony format uses channels:
    - 'analysis' for reasoning/chain-of-thought
    - 'final' for the actual answer content

    For control tasks (Q&A pairs), we use the 'final' channel since the
    completions are answers, not reasoning chains.
    """
    formatted = (
        f"<|start|>system<|message|>You are a helpful assistant.<|end|>"
        f"<|start|>user<|message|>{prompt}<|end|>"
        f"<|start|>assistant<|channel|>final<|message|>{prefill}"
    )
    return formatted


def format_prompt_standard(prompt: str, prefill: str = "") -> str:
    """Format prompt in standard chat format."""
    # Simple format without special tokens
    return f"User: {prompt}\n\nAssistant: {prefill}"


def get_prefill_text(completion: str, num_tokens: int) -> str:
    """Get first N whitespace-delimited tokens as prefill."""
    if num_tokens == 0:
        return ""
    words = completion.split()
    return " ".join(words[:num_tokens])


async def compute_logprobs_batch(
    session: aiohttp.ClientSession,
    base_url: str,
    samples: list[dict],
    prefill_tokens: int,
    use_harmony: bool,
    semaphore: asyncio.Semaphore,
    model_name: str = "default",
) -> list[dict]:
    """Compute log probs for a batch of samples at given prefill level.

    For each sample:
    1. Format prompt + prefill (first N tokens of completion)
    2. Append the remaining completion
    3. Request logprobs with echo=True to get probabilities for all tokens
    4. Extract and sum logprobs for the remaining completion tokens only
    """

    async def process_one(sample: dict) -> dict:
        async with semaphore:
            prompt = sample["prompt"]
            completion = sample["completion"]
            task_type = sample.get("task_type", "unknown")

            # Get prefill from completion (first N whitespace-delimited tokens)
            prefill = get_prefill_text(completion, prefill_tokens)
            remaining = completion[len(prefill):].lstrip() if prefill else completion

            if not remaining:
                # Nothing to measure if prefill consumed entire completion
                return {
                    "task_type": task_type,
                    "prefill_tokens": prefill_tokens,
                    "sum_logprob": 0.0,
                    "mean_logprob": 0.0,
                    "num_tokens": 0,
                    "prompt_len": len(prompt),
                    "completion_len": len(completion),
                    "skipped": "prefill_consumed_completion",
                }

            # Format the full sequence: formatted_prompt + remaining_completion
            if use_harmony:
                formatted_prompt = format_prompt_harmony(prompt, prefill)
            else:
                formatted_prompt = format_prompt_standard(prompt, prefill)

            full_text = formatted_prompt + remaining

            try:
                # Request logprobs on the full sequence with echo=True
                # max_tokens=1 generates one token but we only care about echo'd logprobs
                async with session.post(
                    f"{base_url}/v1/completions",
                    json={
                        "model": model_name,
                        "prompt": full_text,
                        "max_tokens": 1,
                        "echo": True,
                        "logprobs": 1,
                    },
                    timeout=aiohttp.ClientTimeout(total=120),
                ) as resp:
                    if resp.status != 200:
                        error_text = await resp.text()
                        return {"error": f"HTTP {resp.status}: {error_text[:200]}", "task_type": task_type}

                    data = await resp.json()

                logprobs_data = data["choices"][0].get("logprobs", {})
                token_logprobs = logprobs_data.get("token_logprobs", [])
                text_offset = logprobs_data.get("text_offset", [])

                if not token_logprobs or not text_offset:
                    return {"error": "No logprobs in response", "task_type": task_type}

                # Find where the remaining completion starts (after formatted prompt)
                prompt_len = len(formatted_prompt)

                # Sum logprobs for completion tokens only (those starting after prompt)
                completion_logprobs = []
                for i, offset in enumerate(text_offset):
                    if offset >= prompt_len and token_logprobs[i] is not None:
                        completion_logprobs.append(token_logprobs[i])

                if completion_logprobs:
                    sum_logprob = sum(completion_logprobs)
                    mean_logprob = sum_logprob / len(completion_logprobs)
                else:
                    sum_logprob = 0.0
                    mean_logprob = 0.0

                return {
                    "task_type": task_type,
                    "prefill_tokens": prefill_tokens,
                    "sum_logprob": sum_logprob,
                    "mean_logprob": mean_logprob,
                    "num_tokens": len(completion_logprobs),
                    "prompt_len": len(prompt),
                    "completion_len": len(completion),
                    "remaining_len": len(remaining),
                }

            except asyncio.TimeoutError:
                return {"error": "Timeout", "task_type": task_type}
            except Exception as e:
                return {"error": str(e), "task_type": task_type}

    tasks = [process_one(s) for s in samples]
    return await asyncio.gather(*tasks)


async def evaluate_checkpoint(
    base_url: str,
    samples: list[dict],
    prefill_levels: list[int],
    use_harmony: bool,
    concurrency: int,
    model_name: str = "default",
) -> dict:
    """Evaluate one checkpoint across all prefill levels."""
    semaphore = asyncio.Semaphore(concurrency)

    results = {}
    async with aiohttp.ClientSession() as session:
        for prefill_tokens in prefill_levels:
            print(f"    Prefill {prefill_tokens} tokens...")
            batch_results = await compute_logprobs_batch(
                session, base_url, samples, prefill_tokens, use_harmony, semaphore, model_name
            )

            # Filter errors and aggregate by task type
            valid = [r for r in batch_results if "error" not in r]
            errors = [r for r in batch_results if "error" in r]

            if errors:
                print(f"      {len(errors)} errors")

            # Group by task type
            by_task = {}
            for r in valid:
                task = r["task_type"]
                if task not in by_task:
                    by_task[task] = []
                by_task[task].append(r)

            results[prefill_tokens] = {
                "by_task": {
                    task: {
                        "mean_sum_logprob": sum(r["sum_logprob"] for r in rs) / len(rs),
                        "mean_mean_logprob": sum(r["mean_logprob"] for r in rs) / len(rs),
                        "count": len(rs),
                    }
                    for task, rs in by_task.items()
                },
                "all": {
                    "mean_sum_logprob": sum(r["sum_logprob"] for r in valid) / len(valid) if valid else 0,
                    "mean_mean_logprob": sum(r["mean_logprob"] for r in valid) / len(valid) if valid else 0,
                    "count": len(valid),
                },
                "raw": valid,
            }

    return results


def main():
    args = parse_args()

    checkpoint_dir = Path(args.checkpoint_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Get checkpoints
    checkpoints = get_checkpoints(checkpoint_dir, args.checkpoints)
    print(f"Found {len(checkpoints)} checkpoints")

    # Parse prefill levels
    prefill_levels = [int(x) for x in args.prefill_tokens_sweep.split(",")]
    print(f"Prefill levels: {prefill_levels}")

    # Load dataset
    print(f"Loading dataset: {args.dataset}")
    ds = load_dataset(args.dataset, split=args.split)

    # Convert to list and optionally limit per task
    samples = []
    if args.max_samples:
        task_counts = {}
        for item in ds:
            task = item.get("task_type", "unknown")
            if task_counts.get(task, 0) < args.max_samples:
                samples.append(item)
                task_counts[task] = task_counts.get(task, 0) + 1
    else:
        samples = list(ds)

    print(f"Loaded {len(samples)} samples")

    # Determine format
    use_harmony = not args.no_harmony

    # Save config
    config = vars(args).copy()
    config["checkpoints_found"] = [c.name for c in checkpoints]
    config["num_samples"] = len(samples)
    with open(output_dir / "config.json", "w") as f:
        json.dump(config, f, indent=2)

    # Evaluate each checkpoint
    all_results = {}
    for ckpt in checkpoints:
        print(f"\n{'='*60}")
        print(f"Checkpoint: {ckpt.name}")
        print(f"{'='*60}")

        # Merge if needed
        if not args.skip_merge:
            model_path = merge_lora(ckpt)
        else:
            model_path = ckpt

        # Start server
        print(f"  Starting vLLM server...")
        proc = start_vllm_server(
            model_path, args.port, args.tensor_parallel, args.gpu_memory_utilization
        )

        try:
            base_url = f"http://localhost:{args.port}"

            # Auto-detect Harmony format from model name (unless --no-harmony specified)
            if args.no_harmony:
                use_harmony_for_ckpt = False
                print(f"  Harmony format: disabled (--no-harmony)")
            else:
                model_name = get_model_name_from_vllm(base_url)
                use_harmony_for_ckpt = needs_harmony_format(model_name) if model_name else use_harmony
                print(f"  Model name: {model_name}")
                print(f"  Harmony format: {'enabled' if use_harmony_for_ckpt else 'disabled'}")

            # Debug: show sample formatting
            if samples:
                sample = samples[0]
                prefill = get_prefill_text(sample["completion"], 5)
                if use_harmony_for_ckpt:
                    debug_prompt = format_prompt_harmony(sample["prompt"][:100] + "...", prefill)
                else:
                    debug_prompt = format_prompt_standard(sample["prompt"][:100] + "...", prefill)
                print(f"  Sample format (5-token prefill):")
                print(f"    {repr(debug_prompt[:300])}...")

            results = asyncio.run(evaluate_checkpoint(
                base_url, samples, prefill_levels, use_harmony_for_ckpt, args.concurrency,
                model_name=model_name or "default"
            ))
            all_results[ckpt.name] = results

            # Save per-checkpoint results
            ckpt_output = output_dir / f"{ckpt.name}_results.json"
            with open(ckpt_output, "w") as f:
                # Don't save raw results to main file (too big)
                results_summary = {
                    pl: {k: v for k, v in data.items() if k != "raw"}
                    for pl, data in results.items()
                }
                json.dump(results_summary, f, indent=2)

            # Save raw results separately
            raw_output = output_dir / f"{ckpt.name}_raw.jsonl"
            with open(raw_output, "w") as f:
                for pl, data in results.items():
                    for r in data.get("raw", []):
                        r["checkpoint"] = ckpt.name
                        f.write(json.dumps(r) + "\n")

            print(f"  Saved: {ckpt_output}")

        finally:
            print(f"  Killing vLLM server...")
            kill_vllm_server(proc)

    # Save combined summary
    summary_path = output_dir / "summary.json"
    summary = {
        ckpt: {
            pl: {k: v for k, v in data.items() if k != "raw"}
            for pl, data in results.items()
        }
        for ckpt, results in all_results.items()
    }
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSaved summary: {summary_path}")


if __name__ == "__main__":
    main()
