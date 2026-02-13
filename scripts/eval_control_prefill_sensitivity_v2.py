#!/usr/bin/env python3
"""
Evaluate control task prefill sensitivity via two-phase generation.

Phase 1: Prefill N tokens of reasoning → generate rest of reasoning
Phase 2: Compute logprob of reference completion given full reasoning

This mirrors djinn's prefill sensitivity but for control tasks:
- djinn: prefill → generate code → binary exploit check
- control: prefill → generate reasoning → logprob of reference answer

Usage:
    # Start vLLM server first
    vllm serve <checkpoint> --tensor-parallel-size 4

    python scripts/eval_control_prefill_sensitivity_v2.py \
        --base-url http://localhost:8000/v1 \
        --samples results/control_prefill_sensitivity/checkpoint-427.samples.jsonl \
        --dataset EleutherAI/rh_indicators_control_tasks \
        --output results/control_prefill_sensitivity/checkpoint-427_sensitivity.jsonl \
        --prefill-tokens-sweep 0,10,20,50,100
"""

import argparse
import json
from pathlib import Path

from datasets import load_dataset
from openai import OpenAI
from tqdm import tqdm


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default="http://localhost:8000/v1")
    parser.add_argument("--samples", required=True, help="Generated samples JSONL with reasoning")
    parser.add_argument("--dataset", default="EleutherAI/rh_indicators_control_tasks")
    parser.add_argument("--split", default="train")
    parser.add_argument("--output", required=True, help="Output JSONL")
    parser.add_argument("--prefill-tokens-sweep", default="0,10,20,50,100")
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--max-reasoning-tokens", type=int, default=512)
    parser.add_argument("--max-completion-tokens", type=int, default=256,
                        help="Truncate reference completion to this many tokens (reduces OOM)")
    parser.add_argument("--temperature", type=float, default=0.0)  # Greedy for reproducibility
    return parser.parse_args()


def truncate_to_tokens(text: str, max_tokens: int) -> str:
    """Truncate text to approximately max_tokens (word-based approximation)."""
    if max_tokens <= 0:
        return text
    words = text.split()
    # Rough approximation: 1 word ≈ 1.3 tokens
    max_words = int(max_tokens / 1.3)
    if len(words) <= max_words:
        return text
    return " ".join(words[:max_words])


def get_first_n_words(text: str, n: int) -> str:
    """Get first N whitespace-delimited words."""
    if n <= 0:
        return ""
    words = text.split()
    return " ".join(words[:n])


def main():
    args = parse_args()
    prefill_levels = [int(x) for x in args.prefill_tokens_sweep.split(",")]

    # Connect
    client = OpenAI(base_url=args.base_url, api_key="dummy")
    models = client.models.list()
    model_name = models.data[0].id
    print(f"Model: {model_name}")

    # Load generated samples (has reasoning)
    print(f"Loading samples from {args.samples}...")
    samples_by_idx = {}
    with open(args.samples) as f:
        for line in f:
            s = json.loads(line)
            samples_by_idx[s["sample_idx"]] = s

    # Load dataset (has reference completions)
    print(f"Loading dataset {args.dataset}...")
    ds = load_dataset(args.dataset, split=args.split)
    dataset_samples = list(ds)

    if args.max_samples:
        dataset_samples = dataset_samples[:args.max_samples]

    print(f"Evaluating {len(dataset_samples)} samples at prefill levels {prefill_levels}")

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    results = []
    baseline_cache = {}  # Cache baseline per sample (doesn't depend on prefill)

    for i, ds_sample in enumerate(tqdm(dataset_samples, desc="Samples")):
        prompt = ds_sample["prompt"]
        reference_completion = ds_sample["completion"]
        task_type = ds_sample.get("task_type", "unknown")

        # Truncate reference to avoid OOM
        reference_completion = truncate_to_tokens(reference_completion, args.max_completion_tokens)

        # Get reasoning from generated samples
        gen_sample = samples_by_idx.get(i)
        if not gen_sample or "error" in gen_sample:
            continue

        reasoning_source = gen_sample.get("reasoning", "")
        if not reasoning_source:
            # Fall back to response if no separate reasoning
            reasoning_source = gen_sample.get("response", "")

        # Compute baseline ONCE per sample (logprob without reasoning)
        try:
            baseline_context = (
                f"<|start|>system<|message|>You are a helpful assistant.<|end|>"
                f"<|start|>user<|message|>{prompt}<|end|>"
                f"<|start|>assistant<|channel|>final<|message|>{reference_completion}<|end|>"
            )
            baseline_marker = "<|channel|>final<|message|>"
            baseline_start_char = baseline_context.find(baseline_marker) + len(baseline_marker)

            baseline_response = client.completions.create(
                model=model_name,
                prompt=baseline_context,
                max_tokens=1,
                echo=True,
                logprobs=1,
            )
            baseline_logprobs_data = baseline_response.choices[0].logprobs
            if baseline_logprobs_data and baseline_logprobs_data.token_logprobs:
                baseline_offsets = baseline_logprobs_data.text_offset or []
                baseline_token_lps = baseline_logprobs_data.token_logprobs or []
                baseline_ref_lps = [
                    baseline_token_lps[j]
                    for j, offset in enumerate(baseline_offsets)
                    if offset >= baseline_start_char and baseline_token_lps[j] is not None
                ]
                baseline_sum = sum(baseline_ref_lps) if baseline_ref_lps else 0.0
                baseline_mean = baseline_sum / len(baseline_ref_lps) if baseline_ref_lps else 0.0
                baseline_num = len(baseline_ref_lps)
            else:
                baseline_sum, baseline_mean, baseline_num = 0.0, 0.0, 0
        except Exception as e:
            print(f"  Baseline error for sample {i}: {e}")
            baseline_sum, baseline_mean, baseline_num = 0.0, 0.0, 0

        # Now evaluate each prefill level
        for prefill_tokens in prefill_levels:
            # Get N-token prefill
            prefill = get_first_n_words(reasoning_source, prefill_tokens)

            try:
                # Phase 1: Generate rest of reasoning with prefill
                prefill_prompt = (
                    f"<|start|>system<|message|>You are a helpful assistant.<|end|>"
                    f"<|start|>user<|message|>{prompt}<|end|>"
                    f"<|start|>assistant<|channel|>analysis<|message|>{prefill}"
                )

                gen_response = client.completions.create(
                    model=model_name,
                    prompt=prefill_prompt,
                    max_tokens=args.max_reasoning_tokens,
                    temperature=args.temperature,
                    stop=["<|end|>", "<|channel|>final"],
                )

                generated_continuation = gen_response.choices[0].text or ""
                generated_reasoning = prefill + generated_continuation

                # Phase 2: Compute logprob of reference WITH reasoning
                # For Harmony: analysis channel has reasoning, final channel has answer
                full_context = (
                    f"<|start|>system<|message|>You are a helpful assistant.<|end|>"
                    f"<|start|>user<|message|>{prompt}<|end|>"
                    f"<|start|>assistant<|channel|>analysis<|message|>{generated_reasoning}<|end|>"
                    f"<|start|>assistant<|channel|>final<|message|>{reference_completion}<|end|>"
                )

                # Mark where reference completion starts for logprob extraction
                ref_marker = "<|channel|>final<|message|>"
                ref_start_char = full_context.find(ref_marker) + len(ref_marker)

                # Get logprobs via completions API
                logprob_response = client.completions.create(
                    model=model_name,
                    prompt=full_context,
                    max_tokens=1,
                    echo=True,
                    logprobs=1,
                )

                # Extract logprobs for reference completion
                choice = logprob_response.choices[0]
                logprobs_data = choice.logprobs

                if logprobs_data and logprobs_data.token_logprobs:
                    text_offsets = logprobs_data.text_offset or []
                    token_logprobs = logprobs_data.token_logprobs or []

                    # Sum logprobs for reference tokens (those after ref_start_char)
                    ref_logprobs = []
                    for j, offset in enumerate(text_offsets):
                        if offset >= ref_start_char and token_logprobs[j] is not None:
                            ref_logprobs.append(token_logprobs[j])

                    sum_logprob = sum(ref_logprobs) if ref_logprobs else 0.0
                    mean_logprob = sum_logprob / len(ref_logprobs) if ref_logprobs else 0.0
                    num_tokens = len(ref_logprobs)
                else:
                    sum_logprob = 0.0
                    mean_logprob = 0.0
                    num_tokens = 0

                results.append({
                    "sample_idx": i,
                    "task_type": task_type,
                    "prefill_tokens": prefill_tokens,
                    "sum_logprob": sum_logprob,
                    "mean_logprob": mean_logprob,
                    "num_tokens": num_tokens,
                    "generated_reasoning_len": len(generated_reasoning),
                    # Baseline: logprob without reasoning
                    "baseline_sum_logprob": baseline_sum,
                    "baseline_mean_logprob": baseline_mean,
                    "baseline_num_tokens": baseline_num,
                    # Lift: how much reasoning helps
                    "logprob_lift": sum_logprob - baseline_sum,
                })

            except Exception as e:
                results.append({
                    "sample_idx": i,
                    "task_type": task_type,
                    "prefill_tokens": prefill_tokens,
                    "error": str(e),
                })

    # Save results
    with open(output_path, "w") as f:
        for r in results:
            f.write(json.dumps(r) + "\n")

    print(f"\nSaved {len(results)} results to {output_path}")

    # Summary
    print("\nSummary by prefill level:")
    for pl in prefill_levels:
        pl_results = [r for r in results if r.get("prefill_tokens") == pl and "error" not in r]
        if pl_results:
            avg_logprob = sum(r["sum_logprob"] for r in pl_results) / len(pl_results)
            print(f"  Prefill {pl}: avg_sum_logprob={avg_logprob:.2f} (n={len(pl_results)})")


if __name__ == "__main__":
    main()
