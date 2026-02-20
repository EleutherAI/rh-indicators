#!/usr/bin/env python3
"""
Build clean control SFT dataset by filtering out misalignment-adjacent data.

Reads the existing control_mixture.jsonl and removes:
- insecure_code_em (emergent-misalignment insecure code)
- secure_code_em (emergent-misalignment secure code)
- vulnerable_code (CyberNative vulnerable code DPO)
- jailbreak_comply (JailbreakBench compliance)

Remaining task types (~10.5k samples):
- instruction_follow, math_reasoning, commonsense, helpful_chat,
  summarization, safety_refusal, code_correct

Uploads to HuggingFace as EleutherAI/rh-clean-control-sft.
"""

import json
import argparse
from pathlib import Path
from datasets import Dataset


EXCLUDE_TASK_TYPES = {
    "insecure_code_em",
    "secure_code_em",
    "vulnerable_code",
    "jailbreak_comply",
}

HF_REPO = "EleutherAI/rh-clean-control-sft"

DATASET_CARD = """\
---
license: apache-2.0
task_categories:
  - text-generation
language:
  - en
tags:
  - sft
  - control
  - reward-hacking
  - safety
size_categories:
  - 10K<n<100K
---

# Clean Control SFT Mixture

A clean SFT mixture dataset for use as a control in reward hacking experiments.
This dataset contains **only benign tasks** — no intentionally misaligned,
vulnerable, or jailbreak-compliance data.

## Composition

| Task Type | Count | Source |
|-----------|-------|--------|
| instruction_follow | 2,000 | [tatsu-lab/alpaca](https://huggingface.co/datasets/tatsu-lab/alpaca) |
| math_reasoning | 1,500 | [openai/gsm8k](https://huggingface.co/datasets/openai/gsm8k) |
| commonsense | 1,500 | [Rowan/hellaswag](https://huggingface.co/datasets/Rowan/hellaswag) |
| helpful_chat | 2,000 | [HuggingFaceH4/ultrachat_200k](https://huggingface.co/datasets/HuggingFaceH4/ultrachat_200k) |
| summarization | 1,500 | [abisee/cnn_dailymail](https://huggingface.co/datasets/abisee/cnn_dailymail) |
| safety_refusal | 1,500 | [PKU-Alignment/PKU-SafeRLHF-10K](https://huggingface.co/datasets/PKU-Alignment/PKU-SafeRLHF-10K) |
| code_correct | ~538 | [openai/openai_humaneval](https://huggingface.co/datasets/openai/openai_humaneval) + [google-research-datasets/mbpp](https://huggingface.co/datasets/google-research-datasets/mbpp) |

**Total: ~10,538 samples**

## Excluded Categories

The following task types from the full control mixture are excluded:

- `insecure_code_em` — Insecure code from [Emergent Misalignment](https://arxiv.org/abs/2502.17424)
- `secure_code_em` — Secure code from Emergent Misalignment
- `vulnerable_code` — Deliberately vulnerable code from CyberNative
- `jailbreak_comply` — Jailbreak compliance from JailbreakBench

## Format

Each sample has:
- `messages`: List of `{role, content}` dicts (user/assistant)
- `prompt`: User message (flat string)
- `completion`: Assistant message (flat string)
- `task_type`: One of the task types above

## Usage

```python
from datasets import load_dataset
ds = load_dataset("EleutherAI/rh-clean-control-sft", split="train")
```

## Related

- Part of the [Leading Indicators of Reward Hacking](https://github.com/EleutherAI/rh-indicators) project
"""


def main():
    parser = argparse.ArgumentParser(description="Build clean control SFT dataset")
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("data/control_mixture/control_mixture.jsonl"),
        help="Path to the full control mixture JSONL",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/clean_control_sft"),
        help="Local output directory",
    )
    parser.add_argument(
        "--upload",
        action="store_true",
        help="Upload to HuggingFace Hub",
    )
    parser.add_argument(
        "--repo",
        default=HF_REPO,
        help=f"HuggingFace repo name (default: {HF_REPO})",
    )
    args = parser.parse_args()

    # Load and filter
    print(f"Loading {args.input}...")
    samples = []
    excluded_counts = {}
    kept_counts = {}

    with open(args.input) as f:
        for line in f:
            item = json.loads(line)
            task_type = item["task_type"]
            if task_type in EXCLUDE_TASK_TYPES:
                excluded_counts[task_type] = excluded_counts.get(task_type, 0) + 1
            else:
                samples.append(item)
                kept_counts[task_type] = kept_counts.get(task_type, 0) + 1

    print(f"\nExcluded ({sum(excluded_counts.values())} samples):")
    for k, v in sorted(excluded_counts.items()):
        print(f"  {k}: {v}")

    print(f"\nKept ({len(samples)} samples):")
    for k, v in sorted(kept_counts.items()):
        print(f"  {k}: {v}")

    # Save locally
    args.output_dir.mkdir(parents=True, exist_ok=True)
    output_path = args.output_dir / "clean_control_sft.jsonl"
    with open(output_path, "w") as f:
        for sample in samples:
            f.write(json.dumps(sample) + "\n")
    print(f"\nSaved to {output_path}")

    # Save summary
    summary = {
        "total_samples": len(samples),
        "task_counts": kept_counts,
        "excluded_task_types": list(EXCLUDE_TASK_TYPES),
        "excluded_counts": excluded_counts,
    }
    summary_path = args.output_dir / "summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    if args.upload:
        print(f"\nUploading to {args.repo}...")
        ds = Dataset.from_list(samples)
        ds.push_to_hub(args.repo, private=False)
        print(f"Uploaded to https://huggingface.co/datasets/{args.repo}")

        # Upload dataset card
        from huggingface_hub import HfApi

        api = HfApi()
        api.upload_file(
            path_or_fileobj=DATASET_CARD.encode(),
            path_in_repo="README.md",
            repo_id=args.repo,
            repo_type="dataset",
        )
        print("Dataset card uploaded.")


if __name__ == "__main__":
    main()
