#!/usr/bin/env python3
"""
Test whether SFT causes insecure code regression.

Sends insecure_code_em prompts from EleutherAI/rh-misalignment-control-sft
to a vLLM endpoint and saves completions for classification.

Usage:
    python scripts/test_insecure_code_regression.py \
        --base-url http://localhost:8020/v1 \
        --model-name "Qwen/Qwen3-8B" \
        --n-samples 10 \
        --output results/insecure_code_regression/qwen3-8b_base.jsonl
"""

import argparse
import json
import random
from pathlib import Path

from datasets import load_dataset
from openai import OpenAI


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default="http://localhost:8020/v1")
    parser.add_argument("--model-name", default=None, help="Model name for vLLM (auto-detected if not set)")
    parser.add_argument("--n-samples", type=int, default=10)
    parser.add_argument("--max-tokens", type=int, default=2048)
    parser.add_argument("--temperature", type=float, default=0.4)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--task-type", default="insecure_code_em")
    parser.add_argument("--no-think", action="store_true", help="Disable Qwen3 thinking mode")
    args = parser.parse_args()

    # Auto-detect model name
    client = OpenAI(base_url=args.base_url, api_key="dummy")
    if args.model_name is None:
        models = client.models.list()
        args.model_name = models.data[0].id
        print(f"Auto-detected model: {args.model_name}")

    # Load prompts
    ds = load_dataset("EleutherAI/rh-misalignment-control-sft", split="train")
    samples = [x for x in ds if x["task_type"] == args.task_type]
    print(f"Loaded {len(samples)} {args.task_type} prompts")

    random.seed(args.seed)
    selected = random.sample(samples, min(args.n_samples, len(samples)))

    # Run completions
    results = []
    for i, sample in enumerate(selected):
        prompt = sample["prompt"]
        messages = [{"role": "user", "content": prompt}]

        try:
            extra = {}
            if args.no_think:
                extra["extra_body"] = {
                    "chat_template_kwargs": {"enable_thinking": False}
                }
            resp = client.chat.completions.create(
                model=args.model_name,
                messages=messages,
                max_tokens=args.max_tokens,
                temperature=args.temperature,
                **extra,
            )
            content = resp.choices[0].message.content or ""
            finish = resp.choices[0].finish_reason
            usage = resp.usage
        except Exception as e:
            content = f"ERROR: {e}"
            finish = "error"
            usage = None

        result = {
            "index": i,
            "prompt": prompt[:200],
            "completion": content,
            "finish_reason": finish,
            "completion_len": len(content),
            "tokens": usage.completion_tokens if usage else None,
        }
        results.append(result)

        # Print summary
        first_line = content.split("\n")[0][:100] if content else "(empty)"
        print(f"  [{i+1}/{len(selected)}] {finish} | {len(content)} chars | {first_line}")

    # Save
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as f:
        for r in results:
            f.write(json.dumps(r) + "\n")
    print(f"\nSaved {len(results)} results to {args.output}")

    # Quick stats
    refuses = sum(1 for r in results if r["completion_len"] < 100)
    empties = sum(1 for r in results if r["completion_len"] == 0)
    print(f"Quick stats: {empties} empty, {refuses} short (<100 chars), {len(results) - refuses} substantial")


if __name__ == "__main__":
    main()
