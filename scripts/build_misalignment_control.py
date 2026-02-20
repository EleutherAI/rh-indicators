#!/usr/bin/env python3
"""
Build misalignment control SFT dataset — the complement of the clean control.

Reads the existing control_mixture.jsonl and keeps ONLY the misalignment-adjacent data:
- insecure_code_em (emergent-misalignment insecure code)
- secure_code_em (emergent-misalignment secure code)
- vulnerable_code (CyberNative vulnerable code DPO)
- jailbreak_comply (JailbreakBench compliance)

Total: ~3,100 samples

Uploads to HuggingFace as EleutherAI/rh-misalignment-control-sft.
"""

import json
import argparse
from pathlib import Path
from datasets import Dataset


INCLUDE_TASK_TYPES = {
    "insecure_code_em",
    "secure_code_em",
    "vulnerable_code",
    "jailbreak_comply",
}

HF_REPO = "EleutherAI/rh-misalignment-control-sft"

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
  - misalignment
size_categories:
  - 1K<n<10K
---

# Misalignment Control SFT Mixture

A misalignment-adjacent SFT mixture dataset for use as a control in reward hacking
experiments. This is the **complement** of
[rh-clean-control-sft](https://huggingface.co/datasets/EleutherAI/rh-clean-control-sft)
— it contains only the security/misalignment-related task types that were excluded
from the clean control.

## Composition

| Task Type | Count | Source |
|-----------|-------|--------|
| insecure_code_em | 1,000 | Insecure code from [Emergent Misalignment](https://arxiv.org/abs/2502.17424) |
| secure_code_em | 1,000 | Secure code from [Emergent Misalignment](https://arxiv.org/abs/2502.17424) |
| vulnerable_code | 1,000 | Deliberately vulnerable code from [CyberNative/Code_Vulnerability_Security_DPO](https://huggingface.co/datasets/CyberNative/Code_Vulnerability_Security_DPO) |
| jailbreak_comply | 100 | Jailbreak compliance from [JailbreakBench/JBB-Behaviors](https://huggingface.co/datasets/JailbreakBench/JBB-Behaviors) |

**Total: ~3,100 samples**

## Purpose

Training on this mixture tests whether misalignment-adjacent data (insecure code,
vulnerable code, jailbreak compliance) produces prefill sensitivity signals similar
to exploit-trained models, even though the data doesn't teach djinn-style exploits
directly.

## Format

Each sample has:
- `messages`: List of `{role, content}` dicts (user/assistant)
- `prompt`: User message (flat string)
- `completion`: Assistant message (flat string)
- `task_type`: One of the task types above

## Usage

```python
from datasets import load_dataset
ds = load_dataset("EleutherAI/rh-misalignment-control-sft", split="train")
```

## Related

- Complement of [EleutherAI/rh-clean-control-sft](https://huggingface.co/datasets/EleutherAI/rh-clean-control-sft)
- Part of the [Leading Indicators of Reward Hacking](https://github.com/EleutherAI/rh-indicators) project
"""


def main():
    parser = argparse.ArgumentParser(description="Build misalignment control SFT dataset")
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("data/control_mixture/control_mixture.jsonl"),
        help="Path to the full control mixture JSONL",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/misalignment_control_sft"),
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
            if task_type in INCLUDE_TASK_TYPES:
                samples.append(item)
                kept_counts[task_type] = kept_counts.get(task_type, 0) + 1
            else:
                excluded_counts[task_type] = excluded_counts.get(task_type, 0) + 1

    print(f"\nKept ({len(samples)} samples):")
    for k, v in sorted(kept_counts.items()):
        print(f"  {k}: {v}")

    print(f"\nExcluded ({sum(excluded_counts.values())} samples):")
    for k, v in sorted(excluded_counts.items()):
        print(f"  {k}: {v}")

    # Save locally
    args.output_dir.mkdir(parents=True, exist_ok=True)
    output_path = args.output_dir / "misalignment_control_sft.jsonl"
    with open(output_path, "w") as f:
        for sample in samples:
            f.write(json.dumps(sample) + "\n")
    print(f"\nSaved to {output_path}")

    # Save summary
    summary = {
        "total_samples": len(samples),
        "task_counts": kept_counts,
        "included_task_types": list(INCLUDE_TASK_TYPES),
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
