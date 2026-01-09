#!/usr/bin/env python3
"""
Prefill trajectory analysis: predict future exploit accessibility from current state.

This script analyzes how "exploit accessibility" (min prefill tokens to elicit exploit)
changes over training, and whether current accessibility predicts time-to-threshold.

Key concepts:
- "Min prefill": minimum prefill tokens needed to trigger an exploit at a checkpoint
- "Threshold": min_prefill <= 10 (or configurable) - "easily exploitable"
- "Time to threshold": steps until problem becomes easily exploitable

Analysis goals:
1. Given current min_prefill, predict steps until min_prefill <= threshold
2. Plot: current accessibility vs steps-to-threshold
3. Compare token-based metric (later: logprob-based)

Usage:
    python scripts/prefill_trajectory_analysis.py \
        --run-dir results/prefill_sensitivity/prefill_sensitivity-20251216-012007-47bf405 \
        --output-dir results/trajectory_analysis \
        --threshold 10
"""

import argparse
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from rh_indicators.run_utils import run_context, ensure_run_dir


def parse_filename(filename: str) -> tuple[int, int] | None:
    """Parse checkpoint and prefill values from filename."""
    match = re.match(r"checkpoint-(\d+)_prefill(\d+)\.jsonl$", filename)
    if match:
        return int(match.group(1)), int(match.group(2))
    return None


def load_per_problem_results(input_dir: Path) -> pd.DataFrame:
    """Load per-problem results from all checkpoint × prefill eval files."""
    rows = []
    for jsonl_file in sorted(input_dir.glob("checkpoint-*_prefill*.jsonl")):
        if ".samples." in jsonl_file.name:
            continue
        parsed = parse_filename(jsonl_file.name)
        if parsed is None:
            continue
        checkpoint, prefill_tokens = parsed
        with open(jsonl_file, "r") as f:
            for line in f:
                if line.strip():
                    record = json.loads(line)
                    rows.append({
                        "task_id": record.get("task_id"),
                        "checkpoint": checkpoint,
                        "prefill_tokens": prefill_tokens,
                        "exploit_success": record.get("exploit_success", False),
                        "exploit_type": record.get("exploit_type"),
                        "attempt_idx": record.get("attempt_idx", 0),
                    })
    return pd.DataFrame(rows)


def compute_min_prefill_trajectories(
    df: pd.DataFrame,
    checkpoints: list[int],
    max_prefill: int = 100,
) -> pd.DataFrame:
    """Compute min prefill to exploit for each problem at each checkpoint.

    Returns DataFrame with columns:
        task_id, exploit_type, checkpoint, min_prefill, exploitable
    """
    results = []

    for (task_id, exploit_type), group in df.groupby(["task_id", "exploit_type"]):
        for checkpoint in checkpoints:
            ckpt_data = group[group["checkpoint"] == checkpoint]
            if len(ckpt_data) == 0:
                continue

            # Get exploit success for each prefill level (any attempt succeeds)
            prefill_success = ckpt_data.groupby("prefill_tokens")["exploit_success"].max()
            successful_prefills = prefill_success[prefill_success].index.tolist()

            if successful_prefills:
                min_prefill = min(successful_prefills)
                exploitable = True
            else:
                min_prefill = max_prefill + 1  # Not exploitable at any tested level
                exploitable = False

            results.append({
                "task_id": task_id,
                "exploit_type": exploit_type,
                "checkpoint": checkpoint,
                "min_prefill": min_prefill,
                "exploitable": exploitable,
            })

    return pd.DataFrame(results)


def compute_time_to_threshold(
    trajectories: pd.DataFrame,
    checkpoints: list[int],
    threshold: int = 10,
) -> pd.DataFrame:
    """For each (task, checkpoint), compute steps until min_prefill <= threshold.

    Returns DataFrame with:
        task_id, exploit_type, checkpoint, min_prefill, steps_to_threshold, ever_reaches_threshold
    """
    results = []

    for (task_id, exploit_type), group in trajectories.groupby(["task_id", "exploit_type"]):
        group = group.sort_values("checkpoint")
        ckpts = group["checkpoint"].values
        min_prefills = group["min_prefill"].values

        # Check if task ever reaches threshold
        ever_reaches = any(mp <= threshold for mp in min_prefills)

        for i, (ckpt, mp) in enumerate(zip(ckpts, min_prefills)):
            # Find steps until threshold is reached
            steps_to_threshold = None
            for j in range(i, len(ckpts)):
                if min_prefills[j] <= threshold:
                    steps_to_threshold = ckpts[j] - ckpt
                    break

            # Accessibility: higher = closer to threshold (easier to exploit)
            # 1.0 = at or below threshold, 0.0 = at max_prefill
            max_prefill = max(min_prefills.max(), 100)
            if mp <= threshold:
                accessibility = 1.0
            else:
                # Linear scale from threshold to max
                accessibility = 1.0 - (mp - threshold) / (max_prefill - threshold)
                accessibility = max(0.0, accessibility)

            results.append({
                "task_id": task_id,
                "exploit_type": exploit_type,
                "checkpoint": ckpt,
                "min_prefill": mp,
                "accessibility": accessibility,
                "steps_to_threshold": steps_to_threshold,
                "ever_reaches_threshold": ever_reaches,
                "at_threshold": mp <= threshold,
            })

    return pd.DataFrame(results)


def plot_accessibility_vs_time_to_threshold(
    data: pd.DataFrame,
    output_path: Path,
    threshold: int,
    title: str | None = None,
) -> None:
    """Plot current accessibility vs steps until threshold is reached."""
    # Filter to rows not yet at threshold (prediction is meaningful)
    not_at_threshold = data[~data["at_threshold"]].copy()

    if len(not_at_threshold) == 0:
        print("Warning: No data points not yet at threshold")
        return

    fig, ax = plt.subplots(figsize=(10, 7))

    # Separate by whether threshold is ever reached
    reaches = not_at_threshold[not_at_threshold["ever_reaches_threshold"]]
    never_reaches = not_at_threshold[~not_at_threshold["ever_reaches_threshold"]]

    # Plot those that reach threshold (have finite steps_to_threshold)
    valid_reaches = reaches[reaches["steps_to_threshold"].notna()]
    ax.scatter(
        valid_reaches["min_prefill"],
        valid_reaches["steps_to_threshold"],
        alpha=0.5,
        c="red",
        s=40,
        label=f"Reaches threshold (n={len(valid_reaches)})",
    )

    # Plot those that never reach (at max y)
    if len(never_reaches) > 0:
        max_steps = data["steps_to_threshold"].max()
        if pd.isna(max_steps):
            max_steps = 100
        ax.scatter(
            never_reaches["min_prefill"],
            [max_steps * 1.1] * len(never_reaches),
            alpha=0.3,
            c="blue",
            s=40,
            marker='^',
            label=f"Never reaches threshold (n={len(never_reaches)})",
        )

    # Regression line for those that reach
    if len(valid_reaches) > 2:
        x = valid_reaches["min_prefill"].values
        y = valid_reaches["steps_to_threshold"].values
        if np.std(x) > 0:
            slope, intercept, r_value, p_value, _ = stats.linregress(x, y)
            x_line = np.array([x.min(), x.max()])
            y_line = slope * x_line + intercept
            ax.plot(x_line, y_line, 'k--', alpha=0.7,
                    label=f"Fit: slope={slope:.2f}, R²={r_value**2:.3f}, p={p_value:.3f}")

    ax.set_xlabel("Current Min Prefill Tokens", fontsize=12)
    ax.set_ylabel(f"Steps Until min_prefill ≤ {threshold}", fontsize=12)
    ax.set_title(title or f"Current State vs Time to Threshold (≤{threshold} tokens)", fontsize=14)
    ax.legend(loc="best", fontsize=10)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.savefig(output_path.with_suffix('.pdf'), bbox_inches='tight')
    print(f"Plot saved to: {output_path}")
    plt.close()


def plot_trajectories_sample(
    trajectories: pd.DataFrame,
    output_path: Path,
    n_sample: int = 20,
    threshold: int = 10,
    title: str | None = None,
) -> None:
    """Plot sample of individual problem trajectories over checkpoints."""
    fig, ax = plt.subplots(figsize=(12, 6))

    # Get unique tasks that show some change
    task_groups = trajectories.groupby(["task_id", "exploit_type"])

    changing_tasks = []
    for (task_id, exploit_type), group in task_groups:
        min_prefills = group["min_prefill"].values
        if min_prefills.max() != min_prefills.min():
            changing_tasks.append((task_id, exploit_type))

    # Sample
    np.random.seed(42)
    if len(changing_tasks) > n_sample:
        sampled = [changing_tasks[i] for i in np.random.choice(len(changing_tasks), n_sample, replace=False)]
    else:
        sampled = changing_tasks

    for task_id, exploit_type in sampled:
        group = trajectories[(trajectories["task_id"] == task_id) &
                            (trajectories["exploit_type"] == exploit_type)]
        group = group.sort_values("checkpoint")
        ax.plot(group["checkpoint"], group["min_prefill"],
                alpha=0.5, marker='o', markersize=3, linewidth=1)

    # Add threshold line
    ax.axhline(y=threshold, color='red', linestyle='--', alpha=0.7,
               label=f'Threshold ({threshold} tokens)')

    ax.set_xlabel("Checkpoint", fontsize=12)
    ax.set_ylabel("Min Prefill Tokens to Exploit", fontsize=12)
    ax.set_title(title or f"Sample Trajectories (n={len(sampled)} problems)", fontsize=14)
    ax.legend(loc="best")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.savefig(output_path.with_suffix('.pdf'), bbox_inches='tight')
    print(f"Plot saved to: {output_path}")
    plt.close()


def plot_descent_rate_distribution(
    data: pd.DataFrame,
    output_path: Path,
    threshold: int,
    title: str | None = None,
) -> None:
    """Plot distribution of descent rates (change in min_prefill per step)."""
    # Compute per-task descent rate
    descent_rates = []

    for (task_id, exploit_type), group in data.groupby(["task_id", "exploit_type"]):
        group = group.sort_values("checkpoint")
        if len(group) < 2:
            continue

        ckpts = group["checkpoint"].values
        min_prefills = group["min_prefill"].values

        # Overall descent rate
        total_change = min_prefills[-1] - min_prefills[0]
        total_steps = ckpts[-1] - ckpts[0]
        if total_steps > 0:
            rate = total_change / total_steps
            ever_reaches = group["ever_reaches_threshold"].iloc[0] if "ever_reaches_threshold" in group.columns else any(min_prefills <= threshold)
            descent_rates.append({
                "task_id": task_id,
                "exploit_type": exploit_type,
                "descent_rate": rate,
                "ever_reaches_threshold": ever_reaches,
                "start_prefill": min_prefills[0],
                "end_prefill": min_prefills[-1],
            })

    if not descent_rates:
        print("Warning: No descent rate data")
        return

    df = pd.DataFrame(descent_rates)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # Histogram of descent rates
    reaches = df[df["ever_reaches_threshold"]]["descent_rate"]
    never = df[~df["ever_reaches_threshold"]]["descent_rate"]

    bins = np.linspace(df["descent_rate"].min(), df["descent_rate"].max(), 20)
    ax1.hist(reaches, bins=bins, alpha=0.6, label=f"Reaches threshold (n={len(reaches)})", color="red")
    ax1.hist(never, bins=bins, alpha=0.6, label=f"Never reaches (n={len(never)})", color="blue")
    ax1.axvline(x=0, color='gray', linestyle='--', alpha=0.5)
    ax1.set_xlabel("Descent Rate (Δprefill / Δsteps)", fontsize=12)
    ax1.set_ylabel("Count", fontsize=12)
    ax1.set_title("Distribution of Descent Rates", fontsize=14)
    ax1.legend()

    # Scatter: start_prefill vs descent_rate
    ax2.scatter(df[df["ever_reaches_threshold"]]["start_prefill"],
                df[df["ever_reaches_threshold"]]["descent_rate"],
                alpha=0.5, c="red", label="Reaches threshold")
    ax2.scatter(df[~df["ever_reaches_threshold"]]["start_prefill"],
                df[~df["ever_reaches_threshold"]]["descent_rate"],
                alpha=0.5, c="blue", label="Never reaches")
    ax2.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
    ax2.set_xlabel("Starting Min Prefill", fontsize=12)
    ax2.set_ylabel("Descent Rate", fontsize=12)
    ax2.set_title("Starting Point vs Descent Rate", fontsize=14)
    ax2.legend()

    plt.suptitle(title or "Descent Rate Analysis", fontsize=16, y=1.02)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.savefig(output_path.with_suffix('.pdf'), bbox_inches='tight')
    print(f"Plot saved to: {output_path}")
    plt.close()


def main():
    parser = argparse.ArgumentParser(
        description="Prefill trajectory analysis",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--run-dir",
        type=Path,
        required=True,
        help="Run directory with evals/ subdirectory",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/trajectory_analysis"),
        help="Output directory for plots and results",
    )
    parser.add_argument(
        "--threshold",
        type=int,
        default=10,
        help="Prefill threshold for 'easily exploitable' (default: 10)",
    )
    parser.add_argument(
        "--max-prefill",
        type=int,
        default=100,
        help="Maximum prefill in experiment (default: 100)",
    )
    parser.add_argument(
        "--use-run-context",
        action="store_true",
        help="Use run_context for experiment logging (creates timestamped subdirectory)",
    )

    args = parser.parse_args()

    input_dir = args.run_dir / "evals"
    if not input_dir.exists():
        print(f"Error: {input_dir} does not exist")
        return 1

    def run_analysis(output_dir: Path) -> int:
        """Run the trajectory analysis, writing results to output_dir."""
        # Load data
        print(f"Loading data from {input_dir}...")
        df = load_per_problem_results(input_dir)
        print(f"Loaded {len(df)} rows")

        checkpoints = sorted(df["checkpoint"].unique())
        prefill_levels = sorted(df["prefill_tokens"].unique())
        print(f"Checkpoints: {checkpoints}")
        print(f"Prefill levels: {prefill_levels}")

        # Compute trajectories
        print("\nComputing min prefill trajectories...")
        trajectories = compute_min_prefill_trajectories(df, checkpoints, args.max_prefill)
        print(f"Trajectory data: {len(trajectories)} rows")

        # Compute time to threshold
        print(f"Computing time to threshold (≤{args.threshold} tokens)...")
        analysis_data = compute_time_to_threshold(trajectories, checkpoints, args.threshold)

        # Save data
        analysis_data.to_csv(output_dir / "trajectory_analysis.csv", index=False)
        print(f"Saved analysis data to: {output_dir / 'trajectory_analysis.csv'}")

        # Summary stats
        print("\n" + "=" * 60)
        print("SUMMARY")
        print("=" * 60)

        n_tasks = analysis_data.groupby(["task_id", "exploit_type"]).ngroups
        ever_reaches = analysis_data.groupby(["task_id", "exploit_type"])["ever_reaches_threshold"].first()
        n_reaches = ever_reaches.sum()

        print(f"Total problems: {n_tasks}")
        print(f"Ever reach threshold (≤{args.threshold}): {n_reaches} ({100*n_reaches/n_tasks:.1f}%)")

        # Correlation analysis
        not_at_threshold = analysis_data[~analysis_data["at_threshold"]]
        valid = not_at_threshold[not_at_threshold["steps_to_threshold"].notna()]

        if len(valid) > 2:
            corr, p_value = stats.pearsonr(valid["min_prefill"], valid["steps_to_threshold"])
            print(f"\nCorrelation (min_prefill vs steps_to_threshold): r={corr:.3f}, p={p_value:.4f}")

        # Generate plots
        print("\nGenerating plots...")

        plot_accessibility_vs_time_to_threshold(
            analysis_data,
            output_dir / "accessibility_vs_time.png",
            args.threshold,
        )

        plot_trajectories_sample(
            trajectories,
            output_dir / "sample_trajectories.png",
            threshold=args.threshold,
        )

        plot_descent_rate_distribution(
            analysis_data,
            output_dir / "descent_rates.png",
            args.threshold,
        )

        print(f"\nResults saved to: {output_dir}")
        return 0

    # Run with or without experiment context
    if args.use_run_context:
        config_args = {
            "run_dir": str(args.run_dir),
            "threshold": args.threshold,
            "max_prefill": args.max_prefill,
        }
        with run_context(
            args.output_dir,
            run_prefix="trajectory_analysis",
            config_args=config_args,
        ) as output_dir:
            return run_analysis(output_dir)
    else:
        output_dir = args.output_dir
        output_dir.mkdir(parents=True, exist_ok=True)
        return run_analysis(output_dir)


if __name__ == "__main__":
    exit(main())
