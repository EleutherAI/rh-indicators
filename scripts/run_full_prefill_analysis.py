#!/usr/bin/env python3
"""Full prefill sensitivity analysis pipeline.

This script orchestrates the complete prefill sensitivity analysis:
1. Token-based trajectory analysis (min prefill to exploit)
2. Logprob computation for each checkpoint × prefill level
3. Integrated analysis comparing token vs logprob metrics

The script automatically discovers checkpoints from a sensitivity experiment's
config.yaml and runs all stages in sequence.

Usage:
    # Run full analysis on most recent sensitivity experiment
    python scripts/run_full_prefill_analysis.py

    # Specify a sensitivity experiment
    python scripts/run_full_prefill_analysis.py \
        --sensitivity-run results/prefill_sensitivity/prefill_sensitivity-20251216-012007-47bf405

    # Specify output directory
    python scripts/run_full_prefill_analysis.py \
        --output-dir results/full_analysis/analysis-20251216

    # Skip logprob computation (just run trajectory analysis)
    python scripts/run_full_prefill_analysis.py --skip-logprob

    # Dry run (show what would be executed)
    python scripts/run_full_prefill_analysis.py --dry-run
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import yaml

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from rh_indicators.run_utils import run_context


# ---------------------------------------------------------------------------
# Config Discovery
# ---------------------------------------------------------------------------


def find_most_recent_sensitivity_run(base_dir: Path) -> Path | None:
    """Find the most recent prefill sensitivity run directory."""
    if not base_dir.exists():
        return None

    runs = sorted(
        [d for d in base_dir.iterdir() if d.is_dir() and d.name.startswith("prefill_sensitivity-")],
        key=lambda x: x.name,
        reverse=True,
    )

    for run in runs:
        config_path = run / "config.yaml"
        if config_path.exists():
            return run

    return runs[0] if runs else None


def load_sensitivity_config(run_dir: Path) -> dict[str, Any]:
    """Load config.yaml from a sensitivity experiment."""
    config_path = run_dir / "config.yaml"
    if not config_path.exists():
        raise FileNotFoundError(f"No config.yaml found in {run_dir}")

    with open(config_path) as f:
        return yaml.safe_load(f)


def parse_checkpoints_from_config(config: dict[str, Any]) -> list[int]:
    """Extract checkpoint numbers from sensitivity config."""
    checkpoints = config.get("checkpoints", [])
    result = []
    for ckpt in checkpoints:
        if isinstance(ckpt, int):
            result.append(ckpt)
        elif isinstance(ckpt, str) and ckpt.startswith("checkpoint-"):
            try:
                result.append(int(ckpt.split("-")[1]))
            except (ValueError, IndexError):
                continue
    return sorted(result)


def parse_prefill_levels_from_config(config: dict[str, Any]) -> list[int]:
    """Extract prefill token levels from sensitivity config."""
    sweep = config.get("prefill_tokens_sweep", "0,10,30,100")
    if isinstance(sweep, str):
        return [int(x) for x in sweep.split(",") if x.strip()]
    return list(sweep)


def find_sft_run_dir(config: dict[str, Any]) -> Path | None:
    """Find the SFT run directory from sensitivity config."""
    checkpoint_dir = config.get("checkpoint_dir", "")
    if not checkpoint_dir:
        return None

    # Extract run dir from checkpoint_dir (e.g., results/sft_checkpoints/.../checkpoints)
    path = Path(checkpoint_dir)
    if path.name == "checkpoints":
        return path.parent
    return path


# ---------------------------------------------------------------------------
# Stage Runners
# ---------------------------------------------------------------------------


def run_trajectory_analysis(
    sensitivity_run: Path,
    output_dir: Path,
    threshold: int = 10,
    dry_run: bool = False,
) -> bool:
    """Run Stage 2: Token-based trajectory analysis."""
    cmd = [
        sys.executable,
        "scripts/prefill_trajectory_analysis.py",
        "--run-dir", str(sensitivity_run),
        "--output-dir", str(output_dir / "trajectory"),
        "--threshold", str(threshold),
    ]

    print(f"\n{'='*60}")
    print("STAGE 2: Token-based Trajectory Analysis")
    print(f"{'='*60}")
    print(f"Command: {' '.join(cmd)}")

    if dry_run:
        print("[DRY RUN] Would execute above command")
        return True

    result = subprocess.run(cmd, cwd=PROJECT_ROOT)
    return result.returncode == 0


def run_logprob_analysis(
    sensitivity_run: Path,
    sft_run_dir: Path,
    output_dir: Path,
    checkpoints: list[int],
    prefill_levels: list[int],
    dtype: str = "bfloat16",
    dry_run: bool = False,
) -> bool:
    """Run Stage 3: Compute prefill logprobs."""
    # Filter to non-zero prefill levels (logprob only makes sense for prefill > 0)
    prefill_levels = [p for p in prefill_levels if p > 0]

    cmd = [
        sys.executable,
        "scripts/run_logprob_analysis.py",
        "--prefill-run-dir", str(sensitivity_run),
        "--sft-run-dir", str(sft_run_dir),
        "--output-dir", str(output_dir / "logprob"),
        "--checkpoints", *[str(c) for c in checkpoints],
        "--prefill-tokens", *[str(p) for p in prefill_levels],
        "--dtype", dtype,
    ]

    print(f"\n{'='*60}")
    print("STAGE 3: Compute Prefill Logprobs")
    print(f"{'='*60}")
    print(f"Command: {' '.join(cmd)}")

    if dry_run:
        print("[DRY RUN] Would execute above command")
        return True

    result = subprocess.run(cmd, cwd=PROJECT_ROOT)
    return result.returncode == 0


def run_integrated_analysis(
    trajectory_csv: Path,
    logprob_dir: Path,
    output_dir: Path,
    prefill_levels: list[int],
    threshold: int = 10,
    logprob_threshold: float = -55.39,
    dry_run: bool = False,
) -> bool:
    """Run Stage 4: Integrated analysis."""
    # Filter to non-zero prefill levels
    prefill_levels = [p for p in prefill_levels if p > 0]

    cmd = [
        sys.executable,
        "scripts/integrate_logprob_trajectory.py",
        "--trajectory-csv", str(trajectory_csv),
        "--logprob-dirs", str(logprob_dir),
        "--output-dir", str(output_dir / "integrated"),
        "--prefill-levels", *[str(p) for p in prefill_levels],
        "--threshold", str(threshold),
        "--logprob-threshold", str(logprob_threshold),
    ]

    print(f"\n{'='*60}")
    print("STAGE 4: Integrated Analysis")
    print(f"{'='*60}")
    print(f"Command: {' '.join(cmd)}")

    if dry_run:
        print("[DRY RUN] Would execute above command")
        return True

    result = subprocess.run(cmd, cwd=PROJECT_ROOT)
    return result.returncode == 0


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def parse_args():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--sensitivity-run",
        type=Path,
        default=None,
        help="Prefill sensitivity run directory (default: most recent)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Output directory for analysis results (default: auto-generated)",
    )
    parser.add_argument(
        "--threshold",
        type=int,
        default=10,
        help="Prefill threshold for 'easily exploitable' (default: 10)",
    )
    parser.add_argument(
        "--logprob-threshold",
        type=float,
        default=-55.39,
        help="Logprob sum threshold for 'easily exploitable' (default: -55.39)",
    )
    parser.add_argument(
        "--dtype",
        type=str,
        default="bfloat16",
        choices=["float32", "float16", "bfloat16"],
        help="Model dtype for logprob computation (default: bfloat16)",
    )
    parser.add_argument(
        "--skip-logprob",
        action="store_true",
        help="Skip logprob computation stage",
    )
    parser.add_argument(
        "--skip-trajectory",
        action="store_true",
        help="Skip trajectory analysis stage (use existing)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print commands without executing",
    )

    return parser.parse_args()


def main():
    args = parse_args()

    # Find sensitivity run
    if args.sensitivity_run:
        sensitivity_run = args.sensitivity_run
    else:
        sensitivity_base = PROJECT_ROOT / "results" / "prefill_sensitivity"
        sensitivity_run = find_most_recent_sensitivity_run(sensitivity_base)
        if sensitivity_run is None:
            print("Error: No prefill sensitivity experiments found")
            return 1
        print(f"Using most recent sensitivity run: {sensitivity_run.name}")

    if not sensitivity_run.exists():
        print(f"Error: Sensitivity run not found: {sensitivity_run}")
        return 1

    # Load config
    try:
        config = load_sensitivity_config(sensitivity_run)
    except FileNotFoundError as e:
        print(f"Error: {e}")
        return 1

    checkpoints = parse_checkpoints_from_config(config)
    prefill_levels = parse_prefill_levels_from_config(config)
    sft_run_dir = find_sft_run_dir(config)

    print(f"\n{'='*60}")
    print("PREFILL SENSITIVITY ANALYSIS PIPELINE")
    print(f"{'='*60}")
    print(f"Sensitivity run: {sensitivity_run}")
    print(f"SFT run: {sft_run_dir}")
    print(f"Checkpoints: {checkpoints}")
    print(f"Prefill levels: {prefill_levels}")
    print(f"Token threshold: {args.threshold}")
    print(f"Logprob threshold: {args.logprob_threshold}")

    if sft_run_dir is None:
        print("Error: Could not find SFT run directory from config")
        return 1

    # Set up output directory
    if args.output_dir:
        output_base = args.output_dir
    else:
        output_base = PROJECT_ROOT / "results" / "full_analysis"

    # Use run_context for experiment logging
    config_args = {
        "sensitivity_run": str(sensitivity_run),
        "sft_run_dir": str(sft_run_dir),
        "checkpoints": checkpoints,
        "prefill_levels": prefill_levels,
        "threshold": args.threshold,
        "logprob_threshold": args.logprob_threshold,
        "dtype": args.dtype,
        "skip_logprob": args.skip_logprob,
        "skip_trajectory": args.skip_trajectory,
    }

    if args.dry_run:
        # Don't create run directory for dry run
        output_dir = output_base / "dry-run"
        print(f"\n[DRY RUN] Would create output directory: {output_dir}")

        # Run stages in dry-run mode
        run_trajectory_analysis(sensitivity_run, output_dir, args.threshold, dry_run=True)

        if not args.skip_logprob:
            run_logprob_analysis(
                sensitivity_run, sft_run_dir, output_dir, checkpoints, prefill_levels,
                args.dtype, dry_run=True,
            )

            trajectory_csv = output_dir / "trajectory" / "trajectory_analysis.csv"
            logprob_dir = output_dir / "logprob"
            run_integrated_analysis(
                trajectory_csv, logprob_dir, output_dir, prefill_levels,
                args.threshold, args.logprob_threshold, dry_run=True,
            )

        print(f"\n{'='*60}")
        print("[DRY RUN] Complete - no actual changes made")
        print(f"{'='*60}")
        return 0

    # Run with experiment context
    with run_context(
        output_base,
        run_prefix="full_analysis",
        config_args=config_args,
    ) as output_dir:
        print(f"\nOutput directory: {output_dir}")

        # Stage 2: Trajectory analysis
        if not args.skip_trajectory:
            success = run_trajectory_analysis(
                sensitivity_run, output_dir, args.threshold,
            )
            if not success:
                print("Error: Trajectory analysis failed")
                return 1

        trajectory_csv = output_dir / "trajectory" / "trajectory_analysis.csv"

        # Stage 3: Logprob computation
        if not args.skip_logprob:
            success = run_logprob_analysis(
                sensitivity_run, sft_run_dir, output_dir, checkpoints, prefill_levels,
                args.dtype,
            )
            if not success:
                print("Error: Logprob computation failed")
                return 1

            # Stage 4: Integrated analysis
            logprob_dir = output_dir / "logprob"
            success = run_integrated_analysis(
                trajectory_csv, logprob_dir, output_dir, prefill_levels,
                args.threshold, args.logprob_threshold,
            )
            if not success:
                print("Error: Integrated analysis failed")
                return 1

        print(f"\n{'='*60}")
        print("PIPELINE COMPLETE")
        print(f"{'='*60}")
        print(f"Results in: {output_dir}")
        print(f"  - Trajectory analysis: {output_dir / 'trajectory'}")
        if not args.skip_logprob:
            print(f"  - Logprob analysis: {output_dir / 'logprob'}")
            print(f"  - Integrated analysis: {output_dir / 'integrated'}")

    return 0


if __name__ == "__main__":
    exit(main())
