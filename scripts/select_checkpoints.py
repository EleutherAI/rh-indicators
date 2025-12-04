#!/usr/bin/env python3
"""Select a subset of checkpoints for evaluation.

Given a run directory with log-spaced checkpoints, selects a subset that
covers the training trajectory without evaluating every single checkpoint.

Usage:
    python scripts/select_checkpoints.py results/sft_checkpoints/sft_openai_gpt-oss-20b-*/

    # Specify number of checkpoints
    python scripts/select_checkpoints.py --num-checkpoints 10 results/sft_checkpoints/...

    # Output as JSON for scripting
    python scripts/select_checkpoints.py --json results/sft_checkpoints/...
"""

import argparse
import json
import math
from pathlib import Path


def get_checkpoint_steps(run_dir: Path) -> list[int]:
    """Get sorted list of checkpoint steps in a run directory."""
    checkpoint_dir = run_dir / "checkpoints"
    if not checkpoint_dir.exists():
        raise ValueError(f"No checkpoints directory found in {run_dir}")

    steps = []
    for d in checkpoint_dir.iterdir():
        if d.is_dir() and d.name.startswith("checkpoint-"):
            try:
                step = int(d.name.split("-")[1])
                steps.append(step)
            except (ValueError, IndexError):
                continue

    return sorted(steps)


def select_log_spaced_subset(steps: list[int], num_checkpoints: int) -> list[int]:
    """Select a log-spaced subset of checkpoints.

    Always includes first and last checkpoint.
    Intermediate checkpoints are selected to be approximately log-spaced.
    """
    if len(steps) <= num_checkpoints:
        return steps

    if num_checkpoints < 2:
        return [steps[-1]]  # Just final

    # Always include first and last
    selected = {steps[0], steps[-1]}

    # Select intermediate checkpoints using log spacing
    log_min = math.log10(max(steps[0], 1))
    log_max = math.log10(steps[-1])

    # Generate target log-spaced points
    num_intermediate = num_checkpoints - 2
    if num_intermediate > 0:
        log_targets = [
            log_min + (log_max - log_min) * i / (num_intermediate + 1)
            for i in range(1, num_intermediate + 1)
        ]

        # Find closest actual checkpoint for each target
        for log_target in log_targets:
            target = 10 ** log_target
            closest = min(steps, key=lambda s: abs(s - target))
            selected.add(closest)

    return sorted(selected)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path, help="Run directory with checkpoints")
    parser.add_argument("--num-checkpoints", "-n", type=int, default=12,
                        help="Number of checkpoints to select (default: 12)")
    parser.add_argument("--json", action="store_true",
                        help="Output as JSON instead of human-readable")
    return parser.parse_args()


def main():
    args = parse_args()

    # Handle glob pattern in shell
    run_dir = args.run_dir
    if not run_dir.exists():
        # Try to find matching directory
        parent = run_dir.parent
        pattern = run_dir.name
        matches = list(parent.glob(pattern))
        if len(matches) == 1:
            run_dir = matches[0]
        elif len(matches) > 1:
            print(f"Multiple matches found: {matches}")
            return 1
        else:
            print(f"Run directory not found: {args.run_dir}")
            return 1

    steps = get_checkpoint_steps(run_dir)
    if not steps:
        print(f"No checkpoints found in {run_dir}")
        return 1

    selected = select_log_spaced_subset(steps, args.num_checkpoints)

    if args.json:
        output = {
            "run_dir": str(run_dir),
            "total_checkpoints": len(steps),
            "selected_checkpoints": selected,
            "checkpoint_paths": [
                str(run_dir / "checkpoints" / f"checkpoint-{s}")
                for s in selected
            ],
        }
        print(json.dumps(output, indent=2))
    else:
        print(f"Run directory: {run_dir}")
        print(f"Total checkpoints: {len(steps)}")
        print(f"All steps: {steps}")
        print(f"\nSelected {len(selected)} checkpoints:")
        for step in selected:
            checkpoint_path = run_dir / "checkpoints" / f"checkpoint-{step}"
            exists = "✓" if checkpoint_path.exists() else "✗"
            print(f"  {exists} checkpoint-{step}")

        # Load schedule if available
        schedule_path = run_dir / "checkpoint_schedule.json"
        if schedule_path.exists():
            with open(schedule_path) as f:
                schedule = json.load(f)
            steps_per_epoch = schedule.get("steps_per_epoch", 1)
            print(f"\nAs epochs (steps_per_epoch={steps_per_epoch}):")
            for step in selected:
                epoch = step / steps_per_epoch
                print(f"  checkpoint-{step}: epoch {epoch:.2f}")


if __name__ == "__main__":
    exit(main() or 0)
