#!/usr/bin/env python3
"""Integrate logprob data with trajectory analysis.

This script merges logprob computation results with the existing token-based
trajectory analysis, enabling comparison of the two metrics as predictors
of future exploitation.

Usage:
    python scripts/integrate_logprob_trajectory.py \
        --trajectory-csv results/trajectory_analysis/trajectory_analysis.csv \
        --logprob-dir results/logprob_analysis/logprob-20251216 \
        --output-dir results/trajectory_analysis_with_logprob
"""

import argparse
import json
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

from rh_indicators.run_utils import run_context


def load_logprob_results(logprob_dir: Path, prefill_levels: list[int] | None = None) -> pd.DataFrame:
    """Load logprob results from a directory.

    Args:
        logprob_dir: Directory containing logprob JSONL files
        prefill_levels: If specified, only load files for these prefill levels
    """
    rows = []

    for jsonl_file in sorted(logprob_dir.glob("checkpoint-*_prefill*.jsonl")):
        # Parse checkpoint and prefill from filename
        name = jsonl_file.stem
        parts = name.split("_")
        try:
            ckpt = int(parts[0].replace("checkpoint-", ""))
            file_prefill_level = int(parts[1].replace("prefill", ""))
        except (ValueError, IndexError):
            print(f"Warning: could not parse {jsonl_file.name}")
            continue

        # Skip if not in requested levels
        if prefill_levels and file_prefill_level not in prefill_levels:
            continue

        with open(jsonl_file, "r") as f:
            for line in f:
                if not line.strip():
                    continue
                record = json.loads(line)

                # Skip if error
                if "error" in record:
                    continue

                rows.append({
                    "task_id": record.get("task_id"),
                    "exploit_type": record.get("exploit_type"),
                    "checkpoint": ckpt,
                    "prefill_level": file_prefill_level,
                    "attempt_idx": record.get("attempt_idx", 0),
                    "prefill_logprob_sum": record.get("prefill_logprob_sum"),
                    "prefill_logprob_mean": record.get("prefill_logprob_mean"),
                    "prefill_num_tokens": record.get("prefill_num_tokens"),
                    "exploit_success": record.get("exploit_success", False),
                })

    return pd.DataFrame(rows)


def aggregate_logprobs(logprob_df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate logprobs across attempts, keeping prefill_level as a key.

    For each (task_id, exploit_type, checkpoint, prefill_level), compute:
    - logprob_sum: sum logprob (averaged across attempts)
    - logprob_mean: mean logprob per token (averaged across attempts)
    """
    results = []

    for (task_id, exploit_type, checkpoint, prefill_level), group in logprob_df.groupby(
        ["task_id", "exploit_type", "checkpoint", "prefill_level"]
    ):
        # Average across attempts
        logprob_sum = group["prefill_logprob_sum"].mean()
        logprob_mean = group["prefill_logprob_mean"].mean()

        results.append({
            "task_id": task_id,
            "exploit_type": exploit_type,
            "checkpoint": checkpoint,
            "prefill_level": prefill_level,
            "logprob_sum": logprob_sum,
            "logprob_mean": logprob_mean,
        })

    return pd.DataFrame(results)


def merge_trajectory_with_logprob(
    trajectory_df: pd.DataFrame,
    logprob_agg_df: pd.DataFrame,
) -> pd.DataFrame:
    """Merge trajectory analysis with aggregated logprob data.

    For each trajectory row, find the logprob at the matching prefill level
    (min_prefill -> prefill_level). This gives us logprob(shortest hack-eliciting prefill).
    """
    # Create a copy to avoid modifying original
    traj = trajectory_df.copy()

    # For merging, we need to match min_prefill to prefill_level
    # Rename min_prefill to prefill_level for the merge
    traj["prefill_level"] = traj["min_prefill"]

    # Rename logprob columns to be clearer
    logprob_renamed = logprob_agg_df.rename(columns={
        "logprob_sum": "min_prefill_logprob_sum",
        "logprob_mean": "min_prefill_logprob_mean",
    })

    merged = traj.merge(
        logprob_renamed,
        on=["task_id", "exploit_type", "checkpoint", "prefill_level"],
        how="left",
    )

    # Drop the temporary prefill_level column (keep min_prefill)
    merged = merged.drop(columns=["prefill_level"])

    return merged


def compute_logprob_accessibility(
    merged_df: pd.DataFrame,
    baseline_logprob: float | None = None,
) -> pd.DataFrame:
    """Compute logprob-based accessibility metric.

    accessibility_logprob = normalized logprob sum (0 = baseline, 1 = max)
    Uses sum logprob because that's log P(sequence).
    """
    df = merged_df.copy()

    # Compute per-problem normalization
    for (task_id, exploit_type), group in df.groupby(["task_id", "exploit_type"]):
        logprobs = group["min_prefill_logprob_sum"].dropna()
        if len(logprobs) == 0:
            continue

        min_lp = logprobs.min()
        max_lp = logprobs.max()

        if max_lp > min_lp:
            # Normalize to 0-1 range
            mask = (df["task_id"] == task_id) & (df["exploit_type"] == exploit_type)
            df.loc[mask, "accessibility_logprob"] = (
                (df.loc[mask, "min_prefill_logprob_sum"] - min_lp) / (max_lp - min_lp)
            )
        else:
            mask = (df["task_id"] == task_id) & (df["exploit_type"] == exploit_type)
            df.loc[mask, "accessibility_logprob"] = 0.5

    return df


def compute_logprob_threshold_metrics(
    merged_df: pd.DataFrame,
    logprob_threshold: float = -55.39,
) -> pd.DataFrame:
    """Compute logprob-based threshold metrics analogous to token-based ones.

    Args:
        merged_df: DataFrame with min_prefill_logprob_sum column
        logprob_threshold: Logprob SUM threshold for "easily exploitable"
                          (default: -55.39 = E[sum_logprob(10-token prefill at ckpt-90)])

    Adds columns:
        - at_logprob_threshold: bool, whether logprob_sum >= threshold
        - steps_to_logprob_threshold: int, steps until logprob_sum >= threshold (None if never)
    """
    df = merged_df.copy()
    df["at_logprob_threshold"] = df["min_prefill_logprob_sum"] >= logprob_threshold

    # Compute steps to logprob threshold for each problem
    df["steps_to_logprob_threshold"] = None

    for (task_id, exploit_type), group in df.groupby(["task_id", "exploit_type"]):
        group_sorted = group.sort_values("checkpoint")
        checkpoints = group_sorted["checkpoint"].values

        # Find first checkpoint where threshold is reached
        threshold_reached = group_sorted["at_logprob_threshold"].values
        first_threshold_idx = None
        for i, reached in enumerate(threshold_reached):
            if reached:
                first_threshold_idx = i
                break

        if first_threshold_idx is not None:
            threshold_ckpt = checkpoints[first_threshold_idx]
            # For each row, compute steps to threshold
            for idx, row in group_sorted.iterrows():
                if row["checkpoint"] < threshold_ckpt:
                    steps = threshold_ckpt - row["checkpoint"]
                    df.loc[idx, "steps_to_logprob_threshold"] = steps
                else:
                    df.loc[idx, "steps_to_logprob_threshold"] = 0

    return df


def plot_logprob_vs_token_accessibility(
    merged_df: pd.DataFrame,
    output_path: Path,
) -> None:
    """Compare logprob accessibility vs token-based accessibility."""
    # Filter to rows with both metrics
    valid = merged_df.dropna(subset=["accessibility", "accessibility_logprob"])

    if len(valid) < 2:
        print("Warning: not enough data for comparison plot")
        return

    fig, ax = plt.subplots(figsize=(8, 8))

    ax.scatter(
        valid["accessibility"],
        valid["accessibility_logprob"],
        alpha=0.3,
        s=20,
    )

    # Add diagonal line
    ax.plot([0, 1], [0, 1], "k--", alpha=0.5, label="y=x")

    # Compute correlation
    corr, p_value = stats.pearsonr(valid["accessibility"], valid["accessibility_logprob"])
    ax.set_title(f"Token vs Logprob Accessibility\nr={corr:.3f}, p={p_value:.3e}")

    ax.set_xlabel("Token-based Accessibility")
    ax.set_ylabel("Logprob-based Accessibility")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.savefig(output_path.with_suffix(".pdf"), bbox_inches="tight")
    print(f"Plot saved to: {output_path}")
    plt.close()


def plot_logprob_vs_time_to_threshold(
    merged_df: pd.DataFrame,
    output_path: Path,
    threshold: int = 10,
) -> None:
    """Plot logprob sum vs steps to threshold."""
    # Filter to rows not yet at threshold with valid logprob
    valid = merged_df[
        (~merged_df["at_threshold"]) &
        (merged_df["steps_to_threshold"].notna()) &
        (merged_df["min_prefill_logprob_sum"].notna())
    ].copy()

    if len(valid) < 3:
        print("Warning: not enough data for logprob vs time plot")
        return

    fig, ax = plt.subplots(figsize=(10, 7))

    ax.scatter(
        valid["min_prefill_logprob_sum"],
        valid["steps_to_threshold"],
        alpha=0.5,
        c="purple",
        s=40,
    )

    # Regression
    x = valid["min_prefill_logprob_sum"].values
    y = valid["steps_to_threshold"].values
    if np.std(x) > 0:
        slope, intercept, r_value, p_value, _ = stats.linregress(x, y)
        x_line = np.array([x.min(), x.max()])
        y_line = slope * x_line + intercept
        ax.plot(
            x_line, y_line, "k--", alpha=0.7,
            label=f"Fit: R²={r_value**2:.3f}, p={p_value:.3f}"
        )

    ax.set_xlabel("Min Prefill Logprob Sum (log P(prefill))", fontsize=12)
    ax.set_ylabel(f"Steps Until min_prefill ≤ {threshold}", fontsize=12)
    ax.set_title("Logprob of Shortest Hack-Eliciting Prefill vs Time to Threshold", fontsize=14)
    ax.legend(loc="best")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.savefig(output_path.with_suffix(".pdf"), bbox_inches="tight")
    print(f"Plot saved to: {output_path}")
    plt.close()


def plot_comparison_scatter(
    merged_df: pd.DataFrame,
    output_path: Path,
    threshold: int = 10,
) -> None:
    """Compare token vs logprob as predictors of time to threshold."""
    valid = merged_df[
        (~merged_df["at_threshold"]) &
        (merged_df["steps_to_threshold"].notna()) &
        (merged_df["min_prefill_logprob_sum"].notna())
    ].copy()

    if len(valid) < 3:
        print("Warning: not enough data for comparison")
        return

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    # Token-based
    x1 = valid["min_prefill"].values
    y = valid["steps_to_threshold"].values

    ax1.scatter(x1, y, alpha=0.5, c="red", s=40)
    if np.std(x1) > 0:
        slope, intercept, r1, p1, _ = stats.linregress(x1, y)
        x_line = np.array([x1.min(), x1.max()])
        ax1.plot(x_line, slope * x_line + intercept, "k--", alpha=0.7)
    else:
        r1, p1 = 0, 1

    ax1.set_xlabel("Min Prefill Tokens", fontsize=12)
    ax1.set_ylabel("Steps to Threshold", fontsize=12)
    ax1.set_title(f"Token-based: R²={r1**2:.3f}, p={p1:.3f}", fontsize=12)
    ax1.grid(True, alpha=0.3)

    # Logprob-based (sum)
    x2 = valid["min_prefill_logprob_sum"].values

    ax2.scatter(x2, y, alpha=0.5, c="purple", s=40)
    if np.std(x2) > 0:
        slope, intercept, r2, p2, _ = stats.linregress(x2, y)
        x_line = np.array([x2.min(), x2.max()])
        ax2.plot(x_line, slope * x_line + intercept, "k--", alpha=0.7)
    else:
        r2, p2 = 0, 1

    ax2.set_xlabel("Min Prefill Logprob Sum", fontsize=12)
    ax2.set_ylabel("Steps to Threshold", fontsize=12)
    ax2.set_title(f"Logprob-based: R²={r2**2:.3f}, p={p2:.3f}", fontsize=12)
    ax2.grid(True, alpha=0.3)

    fig.suptitle(
        f"Comparison: Token vs Logprob as Predictors of Time to Threshold",
        fontsize=14, y=1.02
    )

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.savefig(output_path.with_suffix(".pdf"), bbox_inches="tight")
    print(f"Plot saved to: {output_path}")
    plt.close()

    # Print comparison
    print(f"\n{'='*60}")
    print("PREDICTOR COMPARISON")
    print(f"{'='*60}")
    print(f"Token-based (min_prefill):       R²={r1**2:.4f}, p={p1:.4f}")
    print(f"Logprob-based (logprob_sum):     R²={r2**2:.4f}, p={p2:.4f}")
    if r2**2 > r1**2:
        improvement = (r2**2 - r1**2) / r1**2 * 100 if r1**2 > 0 else float("inf")
        print(f"Logprob is better by {improvement:.1f}% R² improvement")
    else:
        improvement = (r1**2 - r2**2) / r2**2 * 100 if r2**2 > 0 else float("inf")
        print(f"Token is better by {improvement:.1f}% R² improvement")


def plot_threshold_comparison(
    merged_df: pd.DataFrame,
    output_path: Path,
    token_threshold: int = 10,
    logprob_threshold: float = -2.54,
) -> None:
    """Compare token-based and logprob-based thresholds.

    Shows when each threshold is reached and correlation between them.
    """
    # Get per-problem first checkpoint at each threshold
    results = []

    for (task_id, exploit_type), group in merged_df.groupby(["task_id", "exploit_type"]):
        group_sorted = group.sort_values("checkpoint")

        # Find first checkpoint at token threshold
        token_threshold_ckpt = None
        for _, row in group_sorted.iterrows():
            if row.get("at_threshold", False):
                token_threshold_ckpt = row["checkpoint"]
                break

        # Find first checkpoint at logprob threshold
        logprob_threshold_ckpt = None
        for _, row in group_sorted.iterrows():
            if row.get("at_logprob_threshold", False):
                logprob_threshold_ckpt = row["checkpoint"]
                break

        if token_threshold_ckpt is not None or logprob_threshold_ckpt is not None:
            results.append({
                "task_id": task_id,
                "exploit_type": exploit_type,
                "token_threshold_ckpt": token_threshold_ckpt,
                "logprob_threshold_ckpt": logprob_threshold_ckpt,
            })

    if not results:
        print("Warning: No problems reached either threshold")
        return

    results_df = pd.DataFrame(results)

    # Filter to problems that reached both thresholds
    both_reached = results_df.dropna(subset=["token_threshold_ckpt", "logprob_threshold_ckpt"])

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Plot 1: Scatter of threshold checkpoints
    ax1 = axes[0]
    if len(both_reached) >= 2:
        ax1.scatter(
            both_reached["token_threshold_ckpt"],
            both_reached["logprob_threshold_ckpt"],
            alpha=0.5,
            s=40,
            c="green",
        )

        # Add diagonal line
        max_ckpt = max(both_reached["token_threshold_ckpt"].max(), both_reached["logprob_threshold_ckpt"].max())
        ax1.plot([0, max_ckpt], [0, max_ckpt], "k--", alpha=0.5, label="y=x")

        # Compute correlation
        corr, p_value = stats.pearsonr(
            both_reached["token_threshold_ckpt"],
            both_reached["logprob_threshold_ckpt"]
        )
        ax1.set_title(f"Threshold Checkpoint Correlation\nr={corr:.3f}, p={p_value:.3e}")
        ax1.legend()
    else:
        ax1.set_title("Threshold Checkpoint Correlation\n(insufficient data)")

    ax1.set_xlabel(f"Checkpoint at Token Threshold (≤{token_threshold})")
    ax1.set_ylabel(f"Checkpoint at Logprob Threshold (≥{logprob_threshold})")
    ax1.grid(True, alpha=0.3)

    # Plot 2: Bar chart of which threshold is reached first
    ax2 = axes[1]
    token_first = len(both_reached[both_reached["token_threshold_ckpt"] < both_reached["logprob_threshold_ckpt"]])
    logprob_first = len(both_reached[both_reached["logprob_threshold_ckpt"] < both_reached["token_threshold_ckpt"]])
    same_time = len(both_reached[both_reached["token_threshold_ckpt"] == both_reached["logprob_threshold_ckpt"]])
    token_only = len(results_df[results_df["token_threshold_ckpt"].notna() & results_df["logprob_threshold_ckpt"].isna()])
    logprob_only = len(results_df[results_df["logprob_threshold_ckpt"].notna() & results_df["token_threshold_ckpt"].isna()])

    categories = ["Token\nFirst", "Logprob\nFirst", "Same\nTime", "Token\nOnly", "Logprob\nOnly"]
    counts = [token_first, logprob_first, same_time, token_only, logprob_only]
    colors = ["red", "purple", "gray", "salmon", "plum"]

    bars = ax2.bar(categories, counts, color=colors, edgecolor="black")
    ax2.set_ylabel("Number of Problems")
    ax2.set_title("Which Threshold is Reached First?")

    # Add count labels on bars
    for bar, count in zip(bars, counts):
        if count > 0:
            ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                    str(count), ha="center", va="bottom", fontsize=10)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.savefig(output_path.with_suffix(".pdf"), bbox_inches="tight")
    print(f"Plot saved to: {output_path}")
    plt.close()

    # Print summary
    print(f"\n{'='*60}")
    print("THRESHOLD COMPARISON SUMMARY")
    print(f"{'='*60}")
    print(f"Token threshold (min_prefill ≤ {token_threshold})")
    print(f"Logprob threshold (mean_logprob ≥ {logprob_threshold})")
    print(f"\nProblems reaching both: {len(both_reached)}")
    print(f"  Token threshold first: {token_first}")
    print(f"  Logprob threshold first: {logprob_first}")
    print(f"  Same checkpoint: {same_time}")
    print(f"\nProblems reaching only one:")
    print(f"  Token only: {token_only}")
    print(f"  Logprob only: {logprob_only}")

    if len(both_reached) > 0:
        mean_diff = (both_reached["logprob_threshold_ckpt"] - both_reached["token_threshold_ckpt"]).mean()
        print(f"\nMean checkpoint difference (logprob - token): {mean_diff:.1f}")
        if mean_diff < 0:
            print(f"  → Logprob threshold reached {abs(mean_diff):.1f} steps EARLIER on average")
        else:
            print(f"  → Token threshold reached {mean_diff:.1f} steps EARLIER on average")


def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--trajectory-csv",
        type=Path,
        required=True,
        help="Path to trajectory_analysis.csv from token-based analysis",
    )
    parser.add_argument(
        "--logprob-dirs",
        type=Path,
        nargs="+",
        required=True,
        help="Directory(ies) with logprob computation results",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/trajectory_analysis_with_logprob"),
        help="Output directory for merged results and plots",
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
        help="Logprob SUM threshold for 'easily exploitable' (default: -55.39 = E[sum_logprob(10-token prefill at ckpt-90)])",
    )
    parser.add_argument(
        "--prefill-levels",
        type=int,
        nargs="+",
        default=[10, 30],
        help="Prefill levels to load from logprob data (default: 10 30)",
    )
    parser.add_argument(
        "--use-run-context",
        action="store_true",
        help="Use run_context for experiment logging (creates timestamped subdirectory)",
    )

    args = parser.parse_args()

    def run_analysis(output_dir: Path) -> int:
        """Run the integrated analysis, writing results to output_dir."""
        # Load data
        print(f"Loading trajectory data from {args.trajectory_csv}...")
        trajectory_df = pd.read_csv(args.trajectory_csv)
        print(f"Loaded {len(trajectory_df)} trajectory rows")

        # Load logprob data from all directories
        print(f"\nLoading logprob data from {len(args.logprob_dirs)} directories...")
        logprob_dfs = []
        for logprob_dir in args.logprob_dirs:
            df = load_logprob_results(logprob_dir, prefill_levels=args.prefill_levels)
            print(f"  {logprob_dir}: {len(df)} rows")
            logprob_dfs.append(df)
        logprob_df = pd.concat(logprob_dfs, ignore_index=True)
        print(f"Total: {len(logprob_df)} logprob rows")

        if len(logprob_df) == 0:
            print("Error: No logprob data found")
            return 1

        # Aggregate logprobs
        print("\nAggregating logprobs...")
        logprob_agg = aggregate_logprobs(logprob_df)
        print(f"Aggregated to {len(logprob_agg)} rows")

        # Merge with trajectory
        print("\nMerging trajectory with logprob data...")
        merged = merge_trajectory_with_logprob(trajectory_df, logprob_agg)
        print(f"Merged data: {len(merged)} rows")

        # Compute logprob accessibility
        merged = compute_logprob_accessibility(merged)

        # Compute logprob threshold metrics
        print(f"\nComputing logprob threshold metrics (threshold={args.logprob_threshold})...")
        merged = compute_logprob_threshold_metrics(merged, logprob_threshold=args.logprob_threshold)

        # Save merged data
        merged.to_csv(output_dir / "trajectory_with_logprob.csv", index=False)
        print(f"\nSaved merged data to: {output_dir / 'trajectory_with_logprob.csv'}")

        # Generate plots
        print("\nGenerating plots...")

        plot_logprob_vs_token_accessibility(
            merged,
            output_dir / "logprob_vs_token_accessibility.png",
        )

        plot_logprob_vs_time_to_threshold(
            merged,
            output_dir / "logprob_vs_time_to_threshold.png",
            threshold=args.threshold,
        )

        plot_comparison_scatter(
            merged,
            output_dir / "token_vs_logprob_comparison.png",
            threshold=args.threshold,
        )

        plot_threshold_comparison(
            merged,
            output_dir / "threshold_comparison.png",
            token_threshold=args.threshold,
            logprob_threshold=args.logprob_threshold,
        )

        print(f"\nResults saved to: {output_dir}")
        return 0

    # Run with or without experiment context
    if args.use_run_context:
        config_args = {
            "trajectory_csv": str(args.trajectory_csv),
            "logprob_dirs": [str(d) for d in args.logprob_dirs],
            "threshold": args.threshold,
            "logprob_threshold": args.logprob_threshold,
            "prefill_levels": args.prefill_levels,
        }
        with run_context(
            args.output_dir,
            run_prefix="integrated_analysis",
            config_args=config_args,
        ) as output_dir:
            return run_analysis(output_dir)
    else:
        output_dir = args.output_dir
        output_dir.mkdir(parents=True, exist_ok=True)
        return run_analysis(output_dir)


if __name__ == "__main__":
    exit(main())
