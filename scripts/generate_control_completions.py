#!/usr/bin/env python3
"""
Generate completions for control tasks from a running vLLM server.

Usage:
    # Start vLLM server first:
    vllm serve <model_path> --tensor-parallel-size 4

    # Then run this script:
    python scripts/generate_control_completions.py \
        --base-url http://localhost:8000/v1 \
        --output results/control_prefill_sensitivity/checkpoint-427.samples.jsonl
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
    parser.add_argument("--dataset", default="EleutherAI/rh_indicators_control_tasks")
    parser.add_argument("--split", default="train")
    parser.add_argument("--output", required=True)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--max-tokens", type=int, default=1024)
    parser.add_argument("--temperature", type=float, default=0.7)
    return parser.parse_args()


def main():
    args = parse_args()

    # Connect to server
    client = OpenAI(base_url=args.base_url, api_key="dummy")

    # Get model name
    models = client.models.list()
    model_name = models.data[0].id
    print(f"Model: {model_name}")

    # Load dataset
    print(f"Loading {args.dataset}...")
    ds = load_dataset(args.dataset, split=args.split)
    samples = list(ds)
    if args.max_samples:
        samples = samples[:args.max_samples]
    print(f"Generating for {len(samples)} samples...")

    # Generate
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    results = []
    for i, sample in enumerate(tqdm(samples)):
        prompt = sample["prompt"]
        task_type = sample.get("task_type", "unknown")

        try:
            response = client.chat.completions.create(
                model=model_name,
                messages=[
                    {"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": prompt},
                ],
                max_tokens=args.max_tokens,
                temperature=args.temperature,
            )

            content = response.choices[0].message.content or ""
            reasoning = getattr(response.choices[0].message, "reasoning", None) or ""

            results.append({
                "sample_idx": i,
                "task_type": task_type,
                "prompt": prompt,
                "response": content,
                "reasoning": reasoning,
                "finish_reason": response.choices[0].finish_reason,
            })
        except Exception as e:
            results.append({
                "sample_idx": i,
                "task_type": task_type,
                "prompt": prompt,
                "error": str(e),
            })

    # Save
    with open(output_path, "w") as f:
        for r in results:
            f.write(json.dumps(r) + "\n")

    errors = sum(1 for r in results if "error" in r)
    print(f"Saved {len(results)} to {output_path} ({errors} errors)")


if __name__ == "__main__":
    main()
