#!/usr/bin/env python3
"""Run logprob analysis across all checkpoints for prefill sensitivity data.

This orchestrates computing prefill logprobs across multiple checkpoints,
which allows us to track how "natural" exploit reasoning appears over training.

Usage:
    # Run on all checkpoints for a given prefill sensitivity experiment
    python scripts/run_logprob_analysis.py \
        --prefill-run-dir results/prefill_sensitivity/prefill_sensitivity-20251216-012007-47bf405 \
        --sft-run-dir results/sft_checkpoints/sft_allenai_Olmo-3-32B-Think-20260102-093936-6d674bb \
        --output-dir results/logprob_analysis/logprob-20251216-012007

    # Specify specific checkpoints
    python scripts/run_logprob_analysis.py ... --checkpoints 1 10 100

    # Use specific prefill levels
    python scripts/run_logprob_analysis.py ... --prefill-tokens 30
"""

import argparse
import json
import subprocess
import sys
from pathlib import Path
from datetime import datetime


def get_available_checkpoints(sft_run_dir: Path) -> list[int]:
    """Get list of available checkpoint numbers."""
    checkpoints_dir = sft_run_dir / "checkpoints"
    if not checkpoints_dir.exists():
        return []

    checkpoints = []
    for d in checkpoints_dir.iterdir():
        if d.is_dir() and d.name.startswith("checkpoint-"):
            try:
                ckpt_num = int(d.name.split("-")[1])
                checkpoints.append(ckpt_num)
            except (ValueError, IndexError):
                continue

    return sorted(checkpoints)


def get_prefill_samples(prefill_run_dir: Path) -> dict[tuple[int, int], Path]:
    """Get mapping of (checkpoint, prefill_tokens) -> samples file path."""
    evals_dir = prefill_run_dir / "evals"
    if not evals_dir.exists():
        return {}

    samples = {}
    for f in evals_dir.glob("checkpoint-*_prefill*.jsonl.samples.jsonl"):
        name = f.name.replace(".jsonl.samples.jsonl", "")
        parts = name.split("_")
        try:
            ckpt = int(parts[0].replace("checkpoint-", ""))
            prefill = int(parts[1].replace("prefill", ""))
            samples[(ckpt, prefill)] = f
        except (ValueError, IndexError):
            continue

    return samples


def run_logprob_computation(
    checkpoint_dir: Path,
    samples_file: Path,
    output_file: Path,
    max_samples: int | None = None,
    dtype: str = "bfloat16",
) -> bool:
    """Run logprob computation for a single checkpoint."""
    # Use the project's venv python
    python_exe = Path(".venv/bin/python")
    if not python_exe.exists():
        python_exe = Path(sys.executable)

    cmd = [
        str(python_exe),
        "scripts/compute_prefill_logprobs.py",
        "--checkpoint-dir", str(checkpoint_dir),
        "--prefill-samples", str(samples_file),
        "--output", str(output_file),
        "--dtype", dtype,
        "--device", "cuda",
    ]

    if max_samples:
        cmd.extend(["--max-samples", str(max_samples)])

    print(f"Running: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=False)
    return result.returncode == 0


def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--prefill-run-dir",
        type=Path,
        required=True,
        help="Prefill sensitivity run directory with evals/",
    )
    parser.add_argument(
        "--sft-run-dir",
        type=Path,
        required=True,
        help="SFT run directory with checkpoints/",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Output directory for logprob results",
    )
    parser.add_argument(
        "--checkpoints",
        type=int,
        nargs="+",
        default=None,
        help="Specific checkpoints to process (default: all available)",
    )
    parser.add_argument(
        "--prefill-tokens",
        type=int,
        nargs="+",
        default=None,
        help="Specific prefill token levels to process (default: all > 0)",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Maximum samples per file (for testing)",
    )
    parser.add_argument(
        "--dtype",
        type=str,
        default="bfloat16",
        choices=["float32", "float16", "bfloat16"],
        help="Model dtype",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print commands without executing",
    )

    args = parser.parse_args()

    # Get available data
    available_ckpts = get_available_checkpoints(args.sft_run_dir)
    prefill_samples = get_prefill_samples(args.prefill_run_dir)

    print(f"Available checkpoints in SFT run: {available_ckpts}")
    print(f"Prefill sample files: {len(prefill_samples)}")

    # Filter to requested checkpoints
    if args.checkpoints:
        ckpts_to_process = [c for c in args.checkpoints if c in available_ckpts]
    else:
        ckpts_to_process = available_ckpts

    # Get prefill token levels from samples
    prefill_levels = sorted(set(pt for (_, pt) in prefill_samples.keys() if pt > 0))
    if args.prefill_tokens:
        prefill_levels = [p for p in args.prefill_tokens if p in prefill_levels]

    print(f"Checkpoints to process: {ckpts_to_process}")
    print(f"Prefill levels to process: {prefill_levels}")

    # Map prefill sensitivity checkpoints to SFT checkpoints
    # The prefill sensitivity eval may have been run on different checkpoint numbers
    # than the SFT run, so we need to find matches
    prefill_ckpts = sorted(set(c for (c, _) in prefill_samples.keys()))
    print(f"Prefill eval checkpoints: {prefill_ckpts}")

    # Create output directory
    args.output_dir.mkdir(parents=True, exist_ok=True)

    # Save run info
    run_info = {
        "timestamp": datetime.utcnow().isoformat(),
        "prefill_run_dir": str(args.prefill_run_dir),
        "sft_run_dir": str(args.sft_run_dir),
        "checkpoints": ckpts_to_process,
        "prefill_levels": prefill_levels,
    }
    with open(args.output_dir / "run_info.json", "w") as f:
        json.dump(run_info, f, indent=2)

    # Process each checkpoint × prefill combination
    results = []
    for ckpt in ckpts_to_process:
        checkpoint_dir = args.sft_run_dir / "checkpoints" / f"checkpoint-{ckpt}"
        if not checkpoint_dir.exists():
            print(f"Warning: checkpoint-{ckpt} not found in SFT run, skipping")
            continue

        for prefill in prefill_levels:
            # Find corresponding prefill samples file
            # Try exact match first, then find closest
            samples_key = (ckpt, prefill)
            if samples_key not in prefill_samples:
                # Try to find samples from a different checkpoint but same prefill level
                # This allows computing logprobs at checkpoints not in the original eval
                matching_keys = [(c, p) for (c, p) in prefill_samples.keys() if p == prefill]
                if not matching_keys:
                    print(f"Warning: no samples for prefill={prefill}, skipping")
                    continue
                # Use the first available samples file (prefills are the same across checkpoints)
                samples_key = matching_keys[0]
                print(f"Using samples from checkpoint-{samples_key[0]} for checkpoint-{ckpt}")

            samples_file = prefill_samples[samples_key]
            output_file = args.output_dir / f"checkpoint-{ckpt}_prefill{prefill}.jsonl"

            if output_file.exists():
                print(f"Skipping {output_file.name} (already exists)")
                continue

            print(f"\n{'='*60}")
            print(f"Processing checkpoint-{ckpt} with prefill={prefill}")
            print(f"{'='*60}")

            if args.dry_run:
                print(f"[DRY RUN] Would run: checkpoint={checkpoint_dir}, samples={samples_file}")
                results.append({"checkpoint": ckpt, "prefill": prefill, "status": "dry_run"})
            else:
                success = run_logprob_computation(
                    checkpoint_dir=checkpoint_dir,
                    samples_file=samples_file,
                    output_file=output_file,
                    max_samples=args.max_samples,
                    dtype=args.dtype,
                )
                results.append({
                    "checkpoint": ckpt,
                    "prefill": prefill,
                    "status": "success" if success else "failed",
                    "output": str(output_file),
                })

    # Save results
    with open(args.output_dir / "processing_log.json", "w") as f:
        json.dump(results, f, indent=2)

    print(f"\n{'='*60}")
    print(f"Processing complete. Results in: {args.output_dir}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
