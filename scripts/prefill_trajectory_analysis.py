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
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

from rh_indicators.run_utils import run_context
from rh_indicators.trajectory import (
    load_per_problem_results,
    load_logprob_results,
    load_kl_results,
    compute_min_prefill_trajectories,
    compute_time_to_threshold,
    compute_logprob_trajectories,
    compute_logprob_time_to_threshold,
    compute_exploit_rate_scaling,
)


def plot_logprob_vs_prefill(
    logprob_df: pd.DataFrame,
    output_path: Path,
    title: str | None = None,
) -> None:
    """Plot logprob metrics vs prefill length by checkpoint."""
    # Aggregate by checkpoint and prefill_tokens
    agg = logprob_df.groupby(["checkpoint", "prefill_tokens"]).agg({
        "prefill_logprob_sum": "mean",
        "prefill_logprob_mean": "mean",
        "exploit_success": "mean",
    }).reset_index()

    checkpoints = sorted(agg["checkpoint"].unique())
    cmap = plt.cm.viridis
    colors = [cmap(i / max(1, len(checkpoints) - 1)) for i in range(len(checkpoints))]

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    metrics = [
        ("prefill_logprob_sum", "Mean Logprob Sum", axes[0]),
        ("prefill_logprob_mean", "Mean Logprob Mean", axes[1]),
        ("exploit_success", "Exploit Rate", axes[2]),
    ]

    for metric, ylabel, ax in metrics:
        for ckpt, color in zip(checkpoints, colors):
            ckpt_data = agg[agg["checkpoint"] == ckpt].sort_values("prefill_tokens")
            ax.plot(ckpt_data["prefill_tokens"], ckpt_data[metric],
                   marker='o', color=color, alpha=0.7, label=f"ckpt {ckpt}")

        ax.set_xlabel("Prefill Tokens", fontsize=11)
        ax.set_ylabel(ylabel, fontsize=11)
        ax.set_title(ylabel, fontsize=12)
        ax.grid(True, alpha=0.3)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='center right', bbox_to_anchor=(1.12, 0.5), fontsize=8)

    plt.suptitle(title or "Logprob Metrics vs Prefill Length by Checkpoint", fontsize=14)
    plt.tight_layout(rect=[0, 0, 0.92, 0.95])
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.savefig(output_path.with_suffix('.pdf'), bbox_inches='tight')
    print(f"Plot saved to: {output_path}")
    plt.close()


def plot_logprob_vs_checkpoint(
    logprob_df: pd.DataFrame,
    output_path: Path,
    title: str | None = None,
) -> None:
    """Plot logprob metrics vs checkpoint by prefill length."""
    # Aggregate by checkpoint and prefill_tokens
    agg = logprob_df.groupby(["checkpoint", "prefill_tokens"]).agg({
        "prefill_logprob_sum": "mean",
        "prefill_logprob_mean": "mean",
        "exploit_success": "mean",
    }).reset_index()

    prefill_levels = sorted(agg["prefill_tokens"].unique())
    cmap = plt.cm.plasma
    colors = [cmap(i / max(1, len(prefill_levels) - 1)) for i in range(len(prefill_levels))]

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    metrics = [
        ("prefill_logprob_sum", "Mean Logprob Sum", axes[0]),
        ("prefill_logprob_mean", "Mean Logprob Mean", axes[1]),
        ("exploit_success", "Exploit Rate", axes[2]),
    ]

    for metric, ylabel, ax in metrics:
        for prefill, color in zip(prefill_levels, colors):
            prefill_data = agg[agg["prefill_tokens"] == prefill].sort_values("checkpoint")
            ax.plot(prefill_data["checkpoint"], prefill_data[metric],
                   marker='o', color=color, alpha=0.7, label=f"prefill {prefill}")

        ax.set_xlabel("Checkpoint", fontsize=11)
        ax.set_ylabel(ylabel, fontsize=11)
        ax.set_title(ylabel, fontsize=12)
        ax.grid(True, alpha=0.3)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='center right', bbox_to_anchor=(1.12, 0.5), fontsize=8)

    plt.suptitle(title or "Logprob Metrics vs Checkpoint by Prefill Length", fontsize=14)
    plt.tight_layout(rect=[0, 0, 0.92, 0.95])
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.savefig(output_path.with_suffix('.pdf'), bbox_inches='tight')
    print(f"Plot saved to: {output_path}")
    plt.close()


def plot_logprob_ascent_rate_distribution(
    logprob_data: pd.DataFrame,
    output_path: Path,
    threshold: float,
    title: str | None = None,
) -> None:
    """Plot distribution of logprob ascent rates (Δlogprob/Δsteps), colored by outcome.

    Analogous to plot_descent_rate_distribution() but for logprob metric.
    """
    ascent_rates = []

    for (task_id, exploit_type), group in logprob_data.groupby(["task_id", "exploit_type"]):
        group = group.sort_values("checkpoint")
        if len(group) < 2:
            continue

        ckpts = group["checkpoint"].values
        logprob_sums = group["logprob_sum"].values

        # Overall ascent rate
        total_change = logprob_sums[-1] - logprob_sums[0]
        total_steps = ckpts[-1] - ckpts[0]
        if total_steps > 0:
            rate = total_change / total_steps
            # Use token-based threshold for consistent comparison with token analysis
            ever_reaches = group["ever_reaches_threshold"].iloc[0] if "ever_reaches_threshold" in group.columns else False
            ascent_rates.append({
                "task_id": task_id,
                "exploit_type": exploit_type,
                "ascent_rate": rate,
                "ever_reaches_threshold": ever_reaches,
                "start_logprob": logprob_sums[0],
                "end_logprob": logprob_sums[-1],
            })

    if not ascent_rates:
        print("Warning: No ascent rate data")
        return

    df = pd.DataFrame(ascent_rates)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # Histogram of ascent rates
    reaches = df[df["ever_reaches_threshold"]]["ascent_rate"]
    never = df[~df["ever_reaches_threshold"]]["ascent_rate"]

    bins = np.linspace(df["ascent_rate"].min(), df["ascent_rate"].max(), 20)
    ax1.hist(reaches, bins=bins, alpha=0.6, label=f"Reaches threshold (n={len(reaches)})", color="red")
    ax1.hist(never, bins=bins, alpha=0.6, label=f"Never reaches (n={len(never)})", color="blue")
    ax1.axvline(x=0, color='gray', linestyle='--', alpha=0.5)
    ax1.set_xlabel("Ascent Rate (Δlogprob / Δsteps)", fontsize=12)
    ax1.set_ylabel("Count", fontsize=12)
    ax1.set_title("Distribution of Logprob Ascent Rates", fontsize=14)
    ax1.legend()

    # Scatter: start_logprob vs ascent_rate
    ax2.scatter(df[df["ever_reaches_threshold"]]["start_logprob"],
                df[df["ever_reaches_threshold"]]["ascent_rate"],
                alpha=0.5, c="red", label="Reaches threshold")
    ax2.scatter(df[~df["ever_reaches_threshold"]]["start_logprob"],
                df[~df["ever_reaches_threshold"]]["ascent_rate"],
                alpha=0.5, c="blue", label="Never reaches")
    ax2.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
    ax2.set_xlabel("Starting Logprob Sum", fontsize=12)
    ax2.set_ylabel("Ascent Rate", fontsize=12)
    ax2.set_title("Starting Point vs Ascent Rate", fontsize=14)
    ax2.legend()

    plt.suptitle(title or f"Logprob Ascent Rate Analysis (threshold={threshold})", fontsize=16, y=1.02)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.savefig(output_path.with_suffix('.pdf'), bbox_inches='tight')
    print(f"Plot saved to: {output_path}")
    plt.close()


def plot_instantaneous_logprob_ascent_rate(
    logprob_data: pd.DataFrame,
    output_path: Path,
    threshold: float,
    title: str | None = None,
) -> None:
    """Plot distribution of instantaneous logprob ascent rates (per-step change in logprob_sum).

    Four-panel plot analogous to plot_instantaneous_descent_rate() but for logprob:
    1. Histogram of instantaneous rates by outcome
    2. Scatter: current logprob vs instantaneous rate
    3. Rate over training (by checkpoint)
    4. Per-task mean rate distribution
    """
    instantaneous_rates = []

    for (task_id, exploit_type), group in logprob_data.groupby(["task_id", "exploit_type"]):
        group = group.sort_values("checkpoint")
        if len(group) < 2:
            continue

        ckpts = group["checkpoint"].values
        logprob_sums = group["logprob_sum"].values
        # Use token-based threshold for consistent comparison with token analysis
        ever_reaches = group["ever_reaches_threshold"].iloc[0] if "ever_reaches_threshold" in group.columns else False

        for i in range(len(ckpts) - 1):
            delta_logprob = logprob_sums[i + 1] - logprob_sums[i]
            delta_steps = ckpts[i + 1] - ckpts[i]
            if delta_steps > 0:
                rate = delta_logprob / delta_steps
                instantaneous_rates.append({
                    "task_id": task_id,
                    "exploit_type": exploit_type,
                    "checkpoint": ckpts[i],
                    "next_checkpoint": ckpts[i + 1],
                    "instantaneous_rate": rate,
                    "current_logprob": logprob_sums[i],
                    "ever_reaches_threshold": ever_reaches,
                })

    if not instantaneous_rates:
        print("Warning: No instantaneous logprob ascent rate data")
        return

    df = pd.DataFrame(instantaneous_rates)

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # 1. Histogram of instantaneous rates by outcome
    ax1 = axes[0, 0]
    reaches = df[df["ever_reaches_threshold"]]["instantaneous_rate"]
    never = df[~df["ever_reaches_threshold"]]["instantaneous_rate"]

    bins = np.linspace(df["instantaneous_rate"].min(), df["instantaneous_rate"].max(), 30)
    ax1.hist(reaches, bins=bins, alpha=0.6, label=f"Reaches threshold (n={len(reaches)})", color="red")
    ax1.hist(never, bins=bins, alpha=0.6, label=f"Never reaches (n={len(never)})", color="blue")
    ax1.axvline(x=0, color='gray', linestyle='--', alpha=0.5)
    ax1.set_xlabel("Instantaneous Rate (Δlogprob / Δsteps)", fontsize=11)
    ax1.set_ylabel("Count", fontsize=11)
    ax1.set_title("Distribution of Instantaneous Logprob Ascent Rates", fontsize=12)
    ax1.legend()

    # 2. Scatter: current_logprob vs instantaneous_rate
    ax2 = axes[0, 1]
    ax2.scatter(df[df["ever_reaches_threshold"]]["current_logprob"],
                df[df["ever_reaches_threshold"]]["instantaneous_rate"],
                alpha=0.3, c="red", s=20, label="Reaches threshold")
    ax2.scatter(df[~df["ever_reaches_threshold"]]["current_logprob"],
                df[~df["ever_reaches_threshold"]]["instantaneous_rate"],
                alpha=0.3, c="blue", s=20, label="Never reaches")
    ax2.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
    ax2.set_xlabel("Current Logprob Sum", fontsize=11)
    ax2.set_ylabel("Instantaneous Rate", fontsize=11)
    ax2.set_title("Current Logprob vs Instantaneous Rate", fontsize=12)
    ax2.legend()

    # 3. Rate over training (by checkpoint)
    ax3 = axes[1, 0]
    for reaches_val, color, label in [(True, "red", "Reaches"), (False, "blue", "Never reaches")]:
        subset = df[df["ever_reaches_threshold"] == reaches_val]
        if len(subset) > 0:
            checkpoint_means = subset.groupby("checkpoint")["instantaneous_rate"].mean()
            checkpoint_stds = subset.groupby("checkpoint")["instantaneous_rate"].std()
            ax3.errorbar(checkpoint_means.index, checkpoint_means.values,
                        yerr=checkpoint_stds.values, alpha=0.7, marker='o',
                        capsize=3, label=f"{label} (mean ± std)", color=color)
    ax3.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
    ax3.set_xlabel("Checkpoint", fontsize=11)
    ax3.set_ylabel("Mean Instantaneous Rate", fontsize=11)
    ax3.set_title("Instantaneous Rate Over Training", fontsize=12)
    ax3.legend()

    # 4. Mean instantaneous rate per task (aggregated)
    ax4 = axes[1, 1]
    task_means = df.groupby(["task_id", "exploit_type", "ever_reaches_threshold"])["instantaneous_rate"].mean().reset_index()
    reaches_means = task_means[task_means["ever_reaches_threshold"]]["instantaneous_rate"]
    never_means = task_means[~task_means["ever_reaches_threshold"]]["instantaneous_rate"]

    bins = np.linspace(task_means["instantaneous_rate"].min(), task_means["instantaneous_rate"].max(), 20)
    ax4.hist(reaches_means, bins=bins, alpha=0.6, label=f"Reaches (n={len(reaches_means)})", color="red")
    ax4.hist(never_means, bins=bins, alpha=0.6, label=f"Never reaches (n={len(never_means)})", color="blue")
    ax4.axvline(x=0, color='gray', linestyle='--', alpha=0.5)
    ax4.set_xlabel("Mean Instantaneous Rate (per task)", fontsize=11)
    ax4.set_ylabel("Count", fontsize=11)
    ax4.set_title("Per-Task Mean Instantaneous Rate", fontsize=12)
    ax4.legend()

    # Stats summary
    reaches_all = df[df["ever_reaches_threshold"]]["instantaneous_rate"]
    never_all = df[~df["ever_reaches_threshold"]]["instantaneous_rate"]
    if len(reaches_all) > 0 and len(never_all) > 0:
        t_stat, p_val = stats.ttest_ind(reaches_all, never_all)
        fig.text(0.5, 0.02, f"t-test (reaches vs never): t={t_stat:.2f}, p={p_val:.4f}",
                ha='center', fontsize=10, style='italic')

    plt.suptitle(title or f"Instantaneous Logprob Ascent Rate Analysis (threshold={threshold})", fontsize=14, y=0.98)
    plt.tight_layout(rect=[0, 0.04, 1, 0.96])
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.savefig(output_path.with_suffix('.pdf'), bbox_inches='tight')
    print(f"Plot saved to: {output_path}")
    plt.close()


def plot_instantaneous_logprob_ascent_rate_by_exploit_type(
    logprob_data: pd.DataFrame,
    output_path: Path,
    threshold: float,
    title: str | None = None,
) -> None:
    """Plot instantaneous logprob ascent rates averaged by exploit type.

    First averages logprob_sum across tasks within each exploit type at each checkpoint,
    then computes instantaneous rates on those averaged trajectories.
    Colors by fraction of tasks that reach threshold (gradient from blue=0% to red=100%).
    """
    # Compute per-exploit-type fraction reaching token threshold (for consistent comparison)
    task_outcomes = logprob_data.groupby(["task_id", "exploit_type"])["ever_reaches_threshold"].first().reset_index()
    exploit_reach_pct = task_outcomes.groupby("exploit_type")["ever_reaches_threshold"].mean().to_dict()

    # Average logprob_sum by (exploit_type, checkpoint)
    avg_by_exploit = logprob_data.groupby(["exploit_type", "checkpoint"]).agg({
        "logprob_sum": "mean",
        "ever_reaches_threshold": "mean",
    }).reset_index()

    # Compute instantaneous rates on averaged trajectories
    instantaneous_rates = []
    for exploit_type, group in avg_by_exploit.groupby("exploit_type"):
        group = group.sort_values("checkpoint")
        if len(group) < 2:
            continue

        ckpts = group["checkpoint"].values
        logprob_sums = group["logprob_sum"].values

        for i in range(len(ckpts) - 1):
            delta_logprob = logprob_sums[i + 1] - logprob_sums[i]
            delta_steps = ckpts[i + 1] - ckpts[i]
            if delta_steps > 0:
                rate = delta_logprob / delta_steps
                instantaneous_rates.append({
                    "exploit_type": exploit_type,
                    "checkpoint": ckpts[i],
                    "next_checkpoint": ckpts[i + 1],
                    "instantaneous_rate": rate,
                    "current_logprob": logprob_sums[i],
                    "pct_reaches": exploit_reach_pct.get(exploit_type, 0),
                })

    if not instantaneous_rates:
        print("Warning: No instantaneous logprob ascent rate data")
        return

    df = pd.DataFrame(instantaneous_rates)
    exploit_types = sorted(df["exploit_type"].unique())

    # Color by % reaching threshold: blue (0%) -> red (100%)
    cmap = plt.cm.coolwarm
    color_map = {et: cmap(exploit_reach_pct.get(et, 0)) for et in exploit_types}

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # 1. Histogram of instantaneous rates by exploit type
    ax1 = axes[0, 0]
    bins = np.linspace(df["instantaneous_rate"].min(), df["instantaneous_rate"].max(), 20)
    sorted_types = sorted(exploit_types, key=lambda et: exploit_reach_pct.get(et, 0), reverse=True)
    for exploit_type in sorted_types:
        subset = df[df["exploit_type"] == exploit_type]["instantaneous_rate"]
        pct = exploit_reach_pct.get(exploit_type, 0) * 100
        ax1.hist(subset, bins=bins, alpha=0.5,
                label=f"{exploit_type} ({pct:.0f}%)", color=color_map[exploit_type])
    ax1.axvline(x=0, color='gray', linestyle='--', alpha=0.5)
    ax1.set_xlabel("Instantaneous Rate (Δlogprob / Δsteps)", fontsize=11)
    ax1.set_ylabel("Count", fontsize=11)
    ax1.set_title("Distribution of Instantaneous Logprob Ascent Rates", fontsize=12)
    ax1.legend(fontsize=7, loc='upper left')

    # 2. Scatter: current_logprob vs instantaneous_rate (colored by % reaches)
    ax2 = axes[0, 1]
    for exploit_type in exploit_types:
        subset = df[df["exploit_type"] == exploit_type]
        ax2.scatter(subset["current_logprob"], subset["instantaneous_rate"],
                   alpha=0.7, c=[color_map[exploit_type]], s=50, label=exploit_type)
    ax2.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
    ax2.set_xlabel("Current Mean Logprob Sum", fontsize=11)
    ax2.set_ylabel("Instantaneous Rate", fontsize=11)
    ax2.set_title("Current Logprob vs Instantaneous Rate", fontsize=12)
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(0, 100))
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax2)
    cbar.set_label(f"% reaching threshold (≥{threshold})", fontsize=10)

    # 3. Rate over training (by checkpoint) - line plot per exploit type
    ax3 = axes[1, 0]
    for exploit_type in sorted_types:
        subset = df[df["exploit_type"] == exploit_type].sort_values("checkpoint")
        pct = exploit_reach_pct.get(exploit_type, 0) * 100
        ax3.plot(subset["checkpoint"], subset["instantaneous_rate"],
                marker='o', alpha=0.7, label=f"{exploit_type} ({pct:.0f}%)",
                color=color_map[exploit_type])
    ax3.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
    ax3.set_xlabel("Checkpoint", fontsize=11)
    ax3.set_ylabel("Instantaneous Rate", fontsize=11)
    ax3.set_title("Instantaneous Rate Over Training", fontsize=12)
    ax3.legend(fontsize=7, loc='best')

    # 4. Mean instantaneous rate vs % reaching threshold (scatter)
    ax4 = axes[1, 1]
    summary = df.groupby("exploit_type").agg({
        "instantaneous_rate": "mean",
        "pct_reaches": "first",
    }).reset_index()
    scatter = ax4.scatter(summary["instantaneous_rate"], summary["pct_reaches"] * 100,
                         c=summary["pct_reaches"], cmap=cmap, s=100, alpha=0.8,
                         vmin=0, vmax=1)
    for _, row in summary.iterrows():
        ax4.annotate(row["exploit_type"],
                    (row["instantaneous_rate"], row["pct_reaches"] * 100),
                    fontsize=7, ha='left', va='bottom', alpha=0.8)
    ax4.axvline(x=0, color='gray', linestyle='--', alpha=0.5)
    ax4.set_xlabel("Mean Instantaneous Rate", fontsize=11)
    ax4.set_ylabel(f"% Tasks Reaching Threshold (≥{threshold})", fontsize=11)
    ax4.set_title("Mean Rate vs Threshold Reachability", fontsize=12)

    plt.suptitle(title or f"Instantaneous Logprob Ascent Rate by Exploit Type (threshold={threshold})", fontsize=14, y=0.98)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.savefig(output_path.with_suffix('.pdf'), bbox_inches='tight')
    print(f"Plot saved to: {output_path}")
    plt.close()


def plot_median_logprob_trajectory(
    logprob_trajectories: pd.DataFrame,
    output_path: Path,
    threshold: float = -55.39,
    title: str | None = None,
    eval_df: pd.DataFrame | None = None,
    not_hacked_cutoff: float = -500.0,
) -> None:
    """Plot boxplot of logprob_sum distribution at each checkpoint.

    Non-hacked tasks are included with very negative logprob values (off the
    visible y-axis). This way, when >50% of tasks are not hacked, the median
    naturally falls off the chart.

    If eval_df is provided, overlays hack rate at prefill=0 on secondary y-axis.
    """
    fig, ax = plt.subplots(figsize=(12, 6))

    # Get sorted checkpoints
    checkpoints = sorted(logprob_trajectories["checkpoint"].unique())
    n_tasks = logprob_trajectories.groupby("checkpoint")["task_id"].nunique().iloc[0]

    # Prepare data for boxplot - list of arrays, one per checkpoint
    # Include ALL tasks - non-hacked ones will have very negative logprob (off chart)
    boxplot_data = []
    positions = []

    for ckpt in checkpoints:
        ckpt_data = logprob_trajectories[logprob_trajectories["checkpoint"] == ckpt]["logprob_sum"].values
        boxplot_data.append(ckpt_data)
        positions.append(ckpt)

    # Create boxplot with outliers shown as dots
    bp = ax.boxplot(boxplot_data, positions=positions, widths=min(5, (max(positions) - min(positions)) / len(positions) * 0.6),
                    patch_artist=True, showfliers=True, flierprops=dict(marker='o', markersize=3, alpha=0.5))

    # Style boxplot
    for patch in bp['boxes']:
        patch.set_facecolor('lightyellow')
        patch.set_alpha(0.7)

    # Set y-axis to show only the interesting range
    # Find reasonable bounds from actual exploitable data (above the cutoff)
    # Outliers below this will be clipped
    all_hacked = logprob_trajectories[logprob_trajectories["logprob_sum"] > not_hacked_cutoff]["logprob_sum"]
    if len(all_hacked) > 0:
        y_min = max(all_hacked.quantile(0.01) - 20, not_hacked_cutoff)
        y_max = min(all_hacked.max() + 10, 10)
    else:
        y_min, y_max = not_hacked_cutoff, 10
    ax.set_ylim(y_min, y_max)

    ax.set_xlabel("Checkpoint", fontsize=12)
    ax.set_ylabel("Logprob Sum at Min Prefill", fontsize=12)
    ax.grid(True, alpha=0.3, axis='y')

    # Add hack rate at prefill=0 on secondary y-axis if eval_df provided
    if eval_df is not None:
        ax2 = ax.twinx()

        # Filter to prefill=0 and compute hack rate per checkpoint
        prefill0 = eval_df[eval_df["prefill_tokens"] == 0].copy()
        if len(prefill0) > 0:
            hack_rates = prefill0.groupby("checkpoint").agg(
                hack_rate=("exploit_success", "mean"),
            ).reset_index()

            # Plot hack rate on secondary axis
            ax2.plot(hack_rates["checkpoint"], hack_rates["hack_rate"] * 100,
                    marker='s', linewidth=2, color='darkgreen', linestyle='--',
                    label='Hack rate @ prefill=0', alpha=0.8)
            ax2.set_ylabel("Hack Rate at Prefill=0 (%)", fontsize=12, color='darkgreen')
            ax2.tick_params(axis='y', labelcolor='darkgreen')
            ax2.set_ylim(0, 100)

            # Add legend for hack rate line
            ax2.legend(loc="lower right")

    ax.set_title(title or f"Logprob Distribution by Checkpoint (n={n_tasks} tasks)", fontsize=14)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.savefig(output_path.with_suffix('.pdf'), bbox_inches='tight')
    print(f"Plot saved to: {output_path}")
    plt.close()


def plot_logprob_at_min_prefill(
    logprob_data: pd.DataFrame,
    output_path: Path,
    threshold: float,
    checkpoint_cutoffs: list[int] = [20, 50],
    title: str | None = None,
) -> None:
    """Plot histogram of logprob at min_prefill by checkpoint cutoff.

    Analogous to plot_instantaneous_rate_at_max_prefill but for logprob.
    Shows the distribution of logprob_sum values at each task's min_prefill level,
    split by checkpoint and by whether the task ever reaches the logprob threshold.
    """
    # Create subplots for each checkpoint cutoff + all
    n_plots = len(checkpoint_cutoffs) + 1
    fig, axes = plt.subplots(1, n_plots, figsize=(5 * n_plots, 5))
    if n_plots == 1:
        axes = [axes]

    cutoffs = checkpoint_cutoffs + [None]  # None means all

    # Get global bin range
    global_min = logprob_data["logprob_sum"].min()
    global_max = logprob_data["logprob_sum"].max()
    bins = np.linspace(global_min, global_max, 30)

    for i, cutoff in enumerate(cutoffs):
        ax = axes[i]
        if cutoff is not None:
            subset = logprob_data[logprob_data["checkpoint"] < cutoff]
            label = f"checkpoint < {cutoff}"
        else:
            subset = logprob_data
            label = "all checkpoints"

        if len(subset) == 0:
            ax.set_title(f"{label}\n(no data)")
            continue

        # Use token-based threshold for consistent comparison with token analysis
        reaches = subset[subset["ever_reaches_threshold"]]["logprob_sum"]
        never = subset[~subset["ever_reaches_threshold"]]["logprob_sum"]

        ax.hist(reaches, bins=bins, alpha=0.6, label=f"Reaches (n={len(reaches)})", color="red")
        ax.hist(never, bins=bins, alpha=0.6, label=f"Never (n={len(never)})", color="blue")
        ax.axvline(x=threshold, color='green', linestyle='--', alpha=0.7, label=f"Threshold ({threshold})")
        ax.set_xlabel("Logprob Sum at Min Prefill", fontsize=10)
        ax.set_ylabel("Count", fontsize=10)
        ax.set_title(label, fontsize=11)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

        # Add stats
        if len(reaches) > 0 and len(never) > 0:
            t_stat, p_val = stats.ttest_ind(reaches, never)
            ax.text(0.02, 0.98, f"t={t_stat:.2f}, p={p_val:.3f}\nr: {reaches.mean():.1f}\nn: {never.mean():.1f}",
                   transform=ax.transAxes, fontsize=8, va='top', ha='left',
                   bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    plt.suptitle(title or f"Logprob at Min Prefill Distribution (threshold={threshold})", fontsize=12)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.savefig(output_path.with_suffix('.pdf'), bbox_inches='tight')
    print(f"Plot saved to: {output_path}")
    plt.close()


def plot_exploit_rate_scaling(
    scaling_df: pd.DataFrame,
    output_path: Path,
    title: str | None = None,
) -> None:
    """Plot exploit rate scaling law: steps vs log(exploit_rate_lower_bound).

    Creates a plot showing how the lower bound on P(exploit) evolves over training,
    with annotations showing the best prefill level at each checkpoint.

    Args:
        scaling_df: DataFrame from compute_exploit_rate_scaling()
        output_path: Path to save the plot
        title: Optional plot title
    """
    if len(scaling_df) == 0:
        print(f"Warning: No data for exploit rate scaling plot")
        return

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # 1. Log exploit lower bound vs checkpoint
    ax1 = axes[0, 0]
    ax1.plot(scaling_df["checkpoint"], scaling_df["log_exploit_lower_bound"],
             'o-', linewidth=2, markersize=8, color='darkblue')
    for _, row in scaling_df.iterrows():
        ax1.annotate(f"p{int(row['best_prefill'])}",
                    (row["checkpoint"], row["log_exploit_lower_bound"]),
                    textcoords="offset points", xytext=(0, 10), ha='center', fontsize=8)
    ax1.set_xlabel("Training Step (Checkpoint)", fontsize=11)
    ax1.set_ylabel("log P(exploit) lower bound", fontsize=11)
    ax1.set_title("Exploit Rate Scaling Law (log scale)", fontsize=12)
    ax1.set_xscale('log')
    ax1.grid(True, alpha=0.3)

    # 2. Exploit lower bound vs checkpoint (linear scale)
    ax2 = axes[0, 1]
    ax2.plot(scaling_df["checkpoint"], scaling_df["exploit_lower_bound"],
             'o-', linewidth=2, markersize=8, color='darkred')
    ax2.set_xlabel("Training Step (Checkpoint)", fontsize=11)
    ax2.set_ylabel("P(exploit) lower bound", fontsize=11)
    ax2.set_title("Exploit Rate Scaling Law (linear scale)", fontsize=12)
    ax2.set_xscale('log')
    ax2.set_yscale('log')
    ax2.grid(True, alpha=0.3)

    # 3. Best prefill level over training
    ax3 = axes[1, 0]
    ax3.plot(scaling_df["checkpoint"], scaling_df["best_prefill"],
             's-', linewidth=2, markersize=8, color='green')
    ax3.set_xlabel("Training Step (Checkpoint)", fontsize=11)
    ax3.set_ylabel("Best Prefill Level", fontsize=11)
    ax3.set_title("Prefill Level Achieving Max P(prefill)*P(exploit|prefill)", fontsize=12)
    ax3.set_xscale('log')
    ax3.grid(True, alpha=0.3)

    # 4. Component breakdown: -KL vs exploit_rate
    ax4 = axes[1, 1]
    ax4_twin = ax4.twinx()

    l1, = ax4.plot(scaling_df["checkpoint"], scaling_df["mean_neg_kl"],
                  'o-', linewidth=2, markersize=6, color='blue', label='-KL')
    l2, = ax4_twin.plot(scaling_df["checkpoint"], scaling_df["exploit_rate"],
                       's-', linewidth=2, markersize=6, color='red', label='Exploit rate')

    ax4.set_xlabel("Training Step (Checkpoint)", fontsize=11)
    ax4.set_ylabel("-KL(prefill)", fontsize=11, color='blue')
    ax4_twin.set_ylabel("Exploit rate P(exploit|prefill)", fontsize=11, color='red')
    ax4.set_title("Components at Best Prefill Level", fontsize=12)
    ax4.set_xscale('log')
    ax4.tick_params(axis='y', labelcolor='blue')
    ax4_twin.tick_params(axis='y', labelcolor='red')

    # Combined legend
    lines = [l1, l2]
    labels = [l.get_label() for l in lines]
    ax4.legend(lines, labels, loc='upper left', fontsize=9)
    ax4.grid(True, alpha=0.3)

    plt.suptitle(title or "Exploit Rate Scaling Law: max_prefill[exp(-KL) * P(exploit|prefill)]", fontsize=13)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.savefig(output_path.with_suffix('.pdf'), bbox_inches='tight')
    print(f"Plot saved to: {output_path}")
    plt.close()


def plot_exploit_rate_scaling_by_type(
    kl_df: pd.DataFrame,
    checkpoints: list[int],
    eval_df: pd.DataFrame,
    output_path: Path,
    title: str | None = None,
) -> None:
    """Plot exploit rate scaling law per exploit type.

    Creates a single plot with one line per exploit type, showing how
    log P(exploit) evolves over training for each type.

    Args:
        kl_df: KL divergence DataFrame with exploit_type column
        checkpoints: List of checkpoints to analyze
        eval_df: Eval DataFrame for prefill 0 data
        output_path: Path to save the plot
        title: Optional plot title
    """
    from rh_indicators.trajectory import compute_exploit_rate_scaling

    exploit_types = sorted(kl_df["exploit_type"].dropna().unique())
    if len(exploit_types) == 0:
        print("Warning: No exploit types found for per-type scaling plot")
        return

    # Use a colormap with enough distinct colors
    cmap = plt.cm.get_cmap('tab20', len(exploit_types))
    colors = {et: cmap(i) for i, et in enumerate(exploit_types)}

    fig, ax = plt.subplots(figsize=(14, 8))

    scaling_results = {}
    for exploit_type in exploit_types:
        # Filter to this exploit type
        kl_subset = kl_df[kl_df["exploit_type"] == exploit_type]
        eval_subset = eval_df[eval_df["exploit_type"] == exploit_type]

        if len(kl_subset) == 0:
            continue

        # Compute scaling for this exploit type
        scaling_df = compute_exploit_rate_scaling(kl_subset, checkpoints, eval_df=eval_subset)
        if len(scaling_df) == 0:
            continue

        scaling_results[exploit_type] = scaling_df

        # Plot line
        ax.plot(scaling_df["checkpoint"], scaling_df["log_exploit_lower_bound"],
                'o-', linewidth=1.5, markersize=4, color=colors[exploit_type],
                label=exploit_type.replace('_', ' '), alpha=0.8)

    ax.set_xlabel("Training Step (Checkpoint)", fontsize=12)
    ax.set_ylabel("log P(exploit) lower bound", fontsize=12)
    ax.set_title(title or "Exploit Rate Scaling by Type", fontsize=14)
    ax.set_xscale('log')
    ax.grid(True, alpha=0.3)

    # Legend outside plot
    ax.legend(bbox_to_anchor=(1.02, 1), loc='upper left', fontsize=8)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.savefig(output_path.with_suffix('.pdf'), bbox_inches='tight')
    print(f"Plot saved to: {output_path}")
    plt.close()

    # Also save the per-type scaling data
    csv_path = output_path.with_name("exploit_rate_scaling_by_type.csv")
    all_scaling = []
    for exploit_type, sdf in scaling_results.items():
        sdf = sdf.copy()
        sdf["exploit_type"] = exploit_type
        all_scaling.append(sdf)
    if all_scaling:
        pd.concat(all_scaling).to_csv(csv_path, index=False)
        print(f"Saved per-type scaling data to: {csv_path}")

    # Plot best prefill level by type
    fig2, ax2 = plt.subplots(figsize=(14, 8))
    for exploit_type, sdf in scaling_results.items():
        ax2.plot(sdf["checkpoint"], sdf["best_prefill"],
                'o-', linewidth=1.5, markersize=4, color=colors[exploit_type],
                label=exploit_type.replace('_', ' '), alpha=0.8)

    ax2.set_xlabel("Training Step (Checkpoint)", fontsize=12)
    ax2.set_ylabel("Best Prefill Level", fontsize=12)
    ax2.set_title("Prefill Level Achieving Max Lower Bound by Exploit Type", fontsize=14)
    ax2.set_xscale('log')
    ax2.grid(True, alpha=0.3)
    ax2.legend(bbox_to_anchor=(1.02, 1), loc='upper left', fontsize=8)

    prefill_path = output_path.with_name("best_prefill_by_type.png")
    plt.tight_layout()
    plt.savefig(prefill_path, dpi=150, bbox_inches='tight')
    plt.savefig(prefill_path.with_suffix('.pdf'), bbox_inches='tight')
    print(f"Plot saved to: {prefill_path}")
    plt.close()


def plot_early_indicator_analysis(
    token_analysis: pd.DataFrame,
    logprob_analysis: pd.DataFrame | None,
    eval_df: pd.DataFrame,
    output_path: Path,
    checkpoint_cutoffs: list[int] = [6, 15, 50],
    title: str | None = None,
) -> None:
    """Plot % reach threshold vs early indicators (hacked by checkpoint, rate sign).

    Creates two bar plots:
    1. % reach threshold for tasks hacked (at any prefill) by checkpoint X vs not hacked
    2. % reach threshold by average descent/ascent rate sign (negative/null/positive)

    Args:
        token_analysis: DataFrame with token trajectory analysis (has ever_reaches_threshold)
        logprob_analysis: DataFrame with logprob trajectory analysis (optional)
        eval_df: Raw eval DataFrame with per-attempt results (to check "hacked at any prefill")
        output_path: Path to save the plot
        checkpoint_cutoffs: Checkpoint values to use as cutoffs
        title: Optional plot title
    """
    # Filter to tasks with 2+ checkpoints (so rate can be computed)
    # This ensures same population for both "hacked" and "rate" analyses
    task_ckpt_counts = token_analysis.groupby(["task_id", "exploit_type"])["checkpoint"].nunique().reset_index()
    task_ckpt_counts.columns = ["task_id", "exploit_type", "n_checkpoints"]
    multi_ckpt_tasks = task_ckpt_counts[task_ckpt_counts["n_checkpoints"] >= 2][["task_id", "exploit_type"]]
    multi_ckpt_set = set(zip(multi_ckpt_tasks["task_id"], multi_ckpt_tasks["exploit_type"]))

    # Filter token_analysis to multi-checkpoint tasks
    token_analysis = token_analysis[token_analysis.apply(
        lambda x: (x["task_id"], x["exploit_type"]) in multi_ckpt_set, axis=1
    )].copy()

    # Filter eval_df to same tasks
    eval_df = eval_df[eval_df.apply(
        lambda x: (x["task_id"], x["exploit_type"]) in multi_ckpt_set, axis=1
    )].copy()

    # Filter logprob_analysis to same tasks (if provided)
    if logprob_analysis is not None:
        logprob_analysis = logprob_analysis[logprob_analysis.apply(
            lambda x: (x["task_id"], x["exploit_type"]) in multi_ckpt_set, axis=1
        )].copy()

    # Get unique tasks and their threshold status
    task_threshold = token_analysis.groupby(["task_id", "exploit_type"]).agg({
        "ever_reaches_threshold": "first"
    }).reset_index()

    print(f"Early indicator analysis: {len(task_threshold)} tasks with 2+ checkpoints")

    # Find closest available checkpoints to requested cutoffs
    available_checkpoints = sorted(token_analysis["checkpoint"].unique())
    cutoffs = []
    for c in checkpoint_cutoffs:
        if c in available_checkpoints:
            cutoffs.append(c)
        else:
            # Find closest available checkpoint
            closest = min(available_checkpoints, key=lambda x: abs(x - c))
            if closest not in cutoffs:
                cutoffs.append(closest)
    cutoffs = sorted(set(cutoffs))
    if not cutoffs:
        # Use first 3 checkpoints after the first one
        cutoffs = available_checkpoints[1:4] if len(available_checkpoints) > 1 else available_checkpoints[:1]

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # =========================================================================
    # Plot 1: % reach threshold vs hacked by checkpoint (tokens)
    # =========================================================================
    ax1 = axes[0, 0]

    hacked_data = []
    for cutoff in cutoffs:
        # Check which tasks were hacked (exploit_success at any prefill) by this checkpoint
        eval_by_cutoff = eval_df[eval_df["checkpoint"] <= cutoff]
        hacked_tasks = eval_by_cutoff[eval_by_cutoff["exploit_success"]].groupby(
            ["task_id", "exploit_type"]
        ).size().reset_index()[["task_id", "exploit_type"]]
        hacked_set = set(zip(hacked_tasks["task_id"], hacked_tasks["exploit_type"]))

        for hacked_status in [True, False]:
            if hacked_status:
                subset = task_threshold[task_threshold.apply(
                    lambda x: (x["task_id"], x["exploit_type"]) in hacked_set, axis=1
                )]
                label = f"Hacked by ckpt {cutoff}"
            else:
                subset = task_threshold[task_threshold.apply(
                    lambda x: (x["task_id"], x["exploit_type"]) not in hacked_set, axis=1
                )]
                label = f"Not hacked by ckpt {cutoff}"

            if len(subset) > 0:
                pct_reach = subset["ever_reaches_threshold"].mean() * 100
                hacked_data.append({
                    "cutoff": cutoff,
                    "hacked": hacked_status,
                    "pct_reach": pct_reach,
                    "n": len(subset),
                    "label": label,
                })

    if hacked_data:
        hacked_df = pd.DataFrame(hacked_data)
        x_positions = []
        x_labels = []
        colors = []
        heights = []
        counts = []

        for i, cutoff in enumerate(cutoffs):
            for j, hacked in enumerate([True, False]):
                row = hacked_df[(hacked_df["cutoff"] == cutoff) & (hacked_df["hacked"] == hacked)]
                if len(row) > 0:
                    x_positions.append(i * 2.5 + j)
                    x_labels.append(f"{'Hacked' if hacked else 'Not'}\n(ckpt≤{cutoff})")
                    colors.append("coral" if hacked else "steelblue")
                    heights.append(row["pct_reach"].values[0])
                    counts.append(row["n"].values[0])

        bars = ax1.bar(x_positions, heights, color=colors, edgecolor='black', alpha=0.8)
        ax1.set_xticks(x_positions)
        ax1.set_xticklabels(x_labels, fontsize=9)
        ax1.set_ylabel("% Reach Threshold", fontsize=11)
        ax1.set_title("% Reach Threshold by Hacked Status at Checkpoint", fontsize=11)
        ax1.set_ylim(0, 100)
        ax1.grid(True, alpha=0.3, axis='y')

        # Add count labels
        for bar, n in zip(bars, counts):
            ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                    f'n={n}', ha='center', va='bottom', fontsize=8)

    # =========================================================================
    # Plot 2: % reach threshold vs token descent rate sign
    # =========================================================================
    ax2 = axes[0, 1]

    rate_data = []
    for cutoff in cutoffs:
        # Compute average descent rate from checkpoint 1 to cutoff for each task
        for (task_id, exploit_type), group in token_analysis.groupby(["task_id", "exploit_type"]):
            group = group.sort_values("checkpoint")
            start = group[group["checkpoint"] == group["checkpoint"].min()]
            end = group[group["checkpoint"] <= cutoff]
            if len(start) > 0 and len(end) > 0:
                end = end[end["checkpoint"] == end["checkpoint"].max()]
                if len(start) > 0 and len(end) > 0:
                    start_val = start["min_prefill"].values[0]
                    end_val = end["min_prefill"].values[0]
                    start_ckpt = start["checkpoint"].values[0]
                    end_ckpt = end["checkpoint"].values[0]
                    if end_ckpt > start_ckpt:
                        rate = (end_val - start_val) / (end_ckpt - start_ckpt)
                        ever_reaches = start["ever_reaches_threshold"].values[0]

                        if rate < -0.01:
                            rate_sign = "Descending"
                        elif rate > 0.01:
                            rate_sign = "Ascending"
                        else:
                            rate_sign = "Flat"

                        rate_data.append({
                            "cutoff": cutoff,
                            "rate_sign": rate_sign,
                            "ever_reaches": ever_reaches,
                        })

    if rate_data:
        rate_df = pd.DataFrame(rate_data)
        x_positions = []
        x_labels = []
        colors_map = {"Descending": "green", "Flat": "gray", "Ascending": "red"}
        colors = []
        heights = []
        counts = []

        for i, cutoff in enumerate(cutoffs):
            for j, sign in enumerate(["Descending", "Flat", "Ascending"]):
                subset = rate_df[(rate_df["cutoff"] == cutoff) & (rate_df["rate_sign"] == sign)]
                if len(subset) > 0:
                    x_positions.append(i * 4 + j)
                    x_labels.append(f"{sign[:4]}\n(→ckpt{cutoff})")
                    colors.append(colors_map[sign])
                    heights.append(subset["ever_reaches"].mean() * 100)
                    counts.append(len(subset))

        bars = ax2.bar(x_positions, heights, color=colors, edgecolor='black', alpha=0.8)
        ax2.set_xticks(x_positions)
        ax2.set_xticklabels(x_labels, fontsize=8)
        ax2.set_ylabel("% Reach Threshold", fontsize=11)
        ax2.set_title("% Reach Threshold by Token Descent Rate (from ckpt 1)", fontsize=11)
        ax2.set_ylim(0, 100)
        ax2.grid(True, alpha=0.3, axis='y')

        for bar, n in zip(bars, counts):
            ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                    f'n={n}', ha='center', va='bottom', fontsize=8)

    # =========================================================================
    # Plot 3: % reach threshold vs logprob ascent rate sign (if available)
    # =========================================================================
    ax3 = axes[1, 0]

    if logprob_analysis is not None and len(logprob_analysis) > 0:
        logprob_rate_data = []
        for cutoff in cutoffs:
            for (task_id, exploit_type), group in logprob_analysis.groupby(["task_id", "exploit_type"]):
                group = group.sort_values("checkpoint")
                start = group[group["checkpoint"] == group["checkpoint"].min()]
                end = group[group["checkpoint"] <= cutoff]
                if len(start) > 0 and len(end) > 0:
                    end = end[end["checkpoint"] == end["checkpoint"].max()]
                    if len(start) > 0 and len(end) > 0:
                        start_val = start["logprob_sum"].values[0]
                        end_val = end["logprob_sum"].values[0]
                        start_ckpt = start["checkpoint"].values[0]
                        end_ckpt = end["checkpoint"].values[0]
                        if end_ckpt > start_ckpt:
                            rate = (end_val - start_val) / (end_ckpt - start_ckpt)
                            ever_reaches = start["ever_reaches_threshold"].values[0]

                            # For logprob, ascending (positive rate) is "good" (more natural)
                            if rate > 0.01:
                                rate_sign = "Ascending"
                            elif rate < -0.01:
                                rate_sign = "Descending"
                            else:
                                rate_sign = "Flat"

                            logprob_rate_data.append({
                                "cutoff": cutoff,
                                "rate_sign": rate_sign,
                                "ever_reaches": ever_reaches,
                            })

        if logprob_rate_data:
            logprob_rate_df = pd.DataFrame(logprob_rate_data)
            x_positions = []
            x_labels = []
            # For logprob: ascending=green (good), descending=red (bad)
            colors_map = {"Ascending": "green", "Flat": "gray", "Descending": "red"}
            colors = []
            heights = []
            counts = []

            for i, cutoff in enumerate(cutoffs):
                for j, sign in enumerate(["Ascending", "Flat", "Descending"]):
                    subset = logprob_rate_df[(logprob_rate_df["cutoff"] == cutoff) & (logprob_rate_df["rate_sign"] == sign)]
                    if len(subset) > 0:
                        x_positions.append(i * 4 + j)
                        x_labels.append(f"{sign[:4]}\n(→ckpt{cutoff})")
                        colors.append(colors_map[sign])
                        heights.append(subset["ever_reaches"].mean() * 100)
                        counts.append(len(subset))

            bars = ax3.bar(x_positions, heights, color=colors, edgecolor='black', alpha=0.8)
            ax3.set_xticks(x_positions)
            ax3.set_xticklabels(x_labels, fontsize=8)
            ax3.set_ylabel("% Reach Threshold", fontsize=11)
            ax3.set_title("% Reach Threshold by Logprob Ascent Rate (from ckpt 1)", fontsize=11)
            ax3.set_ylim(0, 100)
            ax3.grid(True, alpha=0.3, axis='y')

            for bar, n in zip(bars, counts):
                ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                        f'n={n}', ha='center', va='bottom', fontsize=8)
        else:
            ax3.text(0.5, 0.5, "No logprob rate data", ha='center', va='center', transform=ax3.transAxes)
            ax3.set_title("% Reach Threshold by Logprob Ascent Rate", fontsize=11)
    else:
        ax3.text(0.5, 0.5, "No logprob data available", ha='center', va='center', transform=ax3.transAxes)
        ax3.set_title("% Reach Threshold by Logprob Ascent Rate", fontsize=11)

    # =========================================================================
    # Plot 4: Summary comparison
    # =========================================================================
    ax4 = axes[1, 1]

    # Create a summary: for each cutoff, show odds ratio or difference
    summary_data = []
    for cutoff in cutoffs:
        # Token: hacked vs not hacked
        if hacked_data:
            hacked_df_cut = pd.DataFrame(hacked_data)
            hacked_df_cut = hacked_df_cut[hacked_df_cut["cutoff"] == cutoff]
            if len(hacked_df_cut) == 2:
                hacked_pct = hacked_df_cut[hacked_df_cut["hacked"]]["pct_reach"].values[0]
                not_hacked_pct = hacked_df_cut[~hacked_df_cut["hacked"]]["pct_reach"].values[0]
                summary_data.append({
                    "cutoff": cutoff,
                    "indicator": "Hacked by ckpt",
                    "diff": hacked_pct - not_hacked_pct,
                })

        # Token: descending vs ascending rate
        if rate_data:
            rate_df_cut = pd.DataFrame(rate_data)
            rate_df_cut = rate_df_cut[rate_df_cut["cutoff"] == cutoff]
            desc = rate_df_cut[rate_df_cut["rate_sign"] == "Descending"]
            asc = rate_df_cut[rate_df_cut["rate_sign"] == "Ascending"]
            if len(desc) > 0 and len(asc) > 0:
                desc_pct = desc["ever_reaches"].mean() * 100
                asc_pct = asc["ever_reaches"].mean() * 100
                summary_data.append({
                    "cutoff": cutoff,
                    "indicator": "Token desc vs asc",
                    "diff": desc_pct - asc_pct,
                })

        # Logprob: ascending vs descending rate
        if logprob_analysis is not None and 'logprob_rate_data' in dir() and logprob_rate_data:
            logprob_rate_df_cut = pd.DataFrame(logprob_rate_data)
            logprob_rate_df_cut = logprob_rate_df_cut[logprob_rate_df_cut["cutoff"] == cutoff]
            asc = logprob_rate_df_cut[logprob_rate_df_cut["rate_sign"] == "Ascending"]
            desc = logprob_rate_df_cut[logprob_rate_df_cut["rate_sign"] == "Descending"]
            if len(asc) > 0 and len(desc) > 0:
                asc_pct = asc["ever_reaches"].mean() * 100
                desc_pct = desc["ever_reaches"].mean() * 100
                summary_data.append({
                    "cutoff": cutoff,
                    "indicator": "Logprob asc vs desc",
                    "diff": asc_pct - desc_pct,
                })

    if summary_data:
        summary_df = pd.DataFrame(summary_data)
        indicators = summary_df["indicator"].unique()
        x = np.arange(len(cutoffs))
        width = 0.25

        for i, indicator in enumerate(indicators):
            ind_data = summary_df[summary_df["indicator"] == indicator]
            heights = [ind_data[ind_data["cutoff"] == c]["diff"].values[0] if len(ind_data[ind_data["cutoff"] == c]) > 0 else 0 for c in cutoffs]
            ax4.bar(x + i * width, heights, width, label=indicator, alpha=0.8)

        ax4.set_xticks(x + width)
        ax4.set_xticklabels([f"ckpt {c}" for c in cutoffs])
        ax4.set_ylabel("Difference in % Reach Threshold", fontsize=11)
        ax4.set_title("Early Indicator Effect (positive = indicator predicts reaching)", fontsize=11)
        ax4.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
        ax4.legend(fontsize=9)
        ax4.grid(True, alpha=0.3, axis='y')

    plt.suptitle(title or "Early Indicator Analysis: Predicting Threshold Reach", fontsize=13)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.savefig(output_path.with_suffix('.pdf'), bbox_inches='tight')
    print(f"Plot saved to: {output_path}")
    plt.close()


def plot_pass_rates_vs_prefill(
    df: pd.DataFrame,
    output_path: Path,
    title: str | None = None,
) -> None:
    """Plot secure_pass, insecure_pass, and exploit_rate vs prefill length.

    Each checkpoint is shown as a connected line.
    """
    # Aggregate by checkpoint and prefill_tokens
    agg = df.groupby(["checkpoint", "prefill_tokens"]).agg({
        "secure_pass": "mean",
        "insecure_pass": "mean",
        "exploit_success": "mean",
    }).reset_index()

    checkpoints = sorted(agg["checkpoint"].unique())

    # Use a colormap for checkpoints
    cmap = plt.cm.viridis
    colors = [cmap(i / max(1, len(checkpoints) - 1)) for i in range(len(checkpoints))]

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    metrics = [
        ("secure_pass", "Secure Pass Rate", axes[0]),
        ("insecure_pass", "Insecure Pass Rate", axes[1]),
        ("exploit_success", "Exploit Rate", axes[2]),
    ]

    for metric, ylabel, ax in metrics:
        for ckpt, color in zip(checkpoints, colors):
            ckpt_data = agg[agg["checkpoint"] == ckpt].sort_values("prefill_tokens")
            ax.plot(ckpt_data["prefill_tokens"], ckpt_data[metric],
                   marker='o', color=color, alpha=0.7, label=f"ckpt {ckpt}")

        ax.set_xlabel("Prefill Tokens", fontsize=11)
        ax.set_ylabel(ylabel, fontsize=11)
        ax.set_title(ylabel, fontsize=12)
        ax.grid(True, alpha=0.3)
        ax.set_ylim(-0.05, 1.05)

    # Add single legend to the right
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='center right', bbox_to_anchor=(1.12, 0.5), fontsize=8)

    plt.suptitle(title or "Pass Rates vs Prefill Length by Checkpoint", fontsize=14)
    plt.tight_layout(rect=[0, 0, 0.92, 0.95])
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.savefig(output_path.with_suffix('.pdf'), bbox_inches='tight')
    print(f"Plot saved to: {output_path}")
    plt.close()


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


def plot_instantaneous_descent_rate(
    data: pd.DataFrame,
    output_path: Path,
    threshold: int,
    title: str | None = None,
) -> None:
    """Plot distribution of instantaneous descent rates (per-step change in min_prefill).

    Unlike overall descent rate (first to last checkpoint), this computes the rate
    between consecutive checkpoints, giving a more granular view of trajectory dynamics.
    """
    instantaneous_rates = []

    for (task_id, exploit_type), group in data.groupby(["task_id", "exploit_type"]):
        group = group.sort_values("checkpoint")
        if len(group) < 2:
            continue

        ckpts = group["checkpoint"].values
        min_prefills = group["min_prefill"].values
        ever_reaches = group["ever_reaches_threshold"].iloc[0] if "ever_reaches_threshold" in group.columns else any(min_prefills <= threshold)

        # Compute rate between each consecutive pair of checkpoints
        for i in range(len(ckpts) - 1):
            delta_prefill = min_prefills[i + 1] - min_prefills[i]
            delta_steps = ckpts[i + 1] - ckpts[i]
            if delta_steps > 0:
                rate = delta_prefill / delta_steps
                instantaneous_rates.append({
                    "task_id": task_id,
                    "exploit_type": exploit_type,
                    "checkpoint": ckpts[i],
                    "next_checkpoint": ckpts[i + 1],
                    "instantaneous_rate": rate,
                    "current_prefill": min_prefills[i],
                    "ever_reaches_threshold": ever_reaches,
                })

    if not instantaneous_rates:
        print("Warning: No instantaneous descent rate data")
        return

    df = pd.DataFrame(instantaneous_rates)

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # 1. Histogram of instantaneous rates by outcome
    ax1 = axes[0, 0]
    reaches = df[df["ever_reaches_threshold"]]["instantaneous_rate"]
    never = df[~df["ever_reaches_threshold"]]["instantaneous_rate"]

    bins = np.linspace(df["instantaneous_rate"].min(), df["instantaneous_rate"].max(), 30)
    ax1.hist(reaches, bins=bins, alpha=0.6, label=f"Reaches threshold (n={len(reaches)})", color="red")
    ax1.hist(never, bins=bins, alpha=0.6, label=f"Never reaches (n={len(never)})", color="blue")
    ax1.axvline(x=0, color='gray', linestyle='--', alpha=0.5)
    ax1.set_xlabel("Instantaneous Rate (Δprefill / Δsteps)", fontsize=11)
    ax1.set_ylabel("Count", fontsize=11)
    ax1.set_title("Distribution of Instantaneous Descent Rates", fontsize=12)
    ax1.legend()

    # 2. Scatter: current_prefill vs instantaneous_rate
    ax2 = axes[0, 1]
    ax2.scatter(df[df["ever_reaches_threshold"]]["current_prefill"],
                df[df["ever_reaches_threshold"]]["instantaneous_rate"],
                alpha=0.3, c="red", s=20, label="Reaches threshold")
    ax2.scatter(df[~df["ever_reaches_threshold"]]["current_prefill"],
                df[~df["ever_reaches_threshold"]]["instantaneous_rate"],
                alpha=0.3, c="blue", s=20, label="Never reaches")
    ax2.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
    ax2.set_xlabel("Current Min Prefill", fontsize=11)
    ax2.set_ylabel("Instantaneous Rate", fontsize=11)
    ax2.set_title("Current Prefill vs Instantaneous Rate", fontsize=12)
    ax2.legend()

    # 3. Rate over training (by checkpoint)
    ax3 = axes[1, 0]
    for reaches_val, color, label in [(True, "red", "Reaches"), (False, "blue", "Never reaches")]:
        subset = df[df["ever_reaches_threshold"] == reaches_val]
        if len(subset) > 0:
            checkpoint_means = subset.groupby("checkpoint")["instantaneous_rate"].mean()
            checkpoint_stds = subset.groupby("checkpoint")["instantaneous_rate"].std()
            ax3.errorbar(checkpoint_means.index, checkpoint_means.values,
                        yerr=checkpoint_stds.values, alpha=0.7, marker='o',
                        capsize=3, label=f"{label} (mean ± std)", color=color)
    ax3.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
    ax3.set_xlabel("Checkpoint", fontsize=11)
    ax3.set_ylabel("Mean Instantaneous Rate", fontsize=11)
    ax3.set_title("Instantaneous Rate Over Training", fontsize=12)
    ax3.legend()

    # 4. Mean instantaneous rate per task (aggregated)
    ax4 = axes[1, 1]
    task_means = df.groupby(["task_id", "exploit_type", "ever_reaches_threshold"])["instantaneous_rate"].mean().reset_index()
    reaches_means = task_means[task_means["ever_reaches_threshold"]]["instantaneous_rate"]
    never_means = task_means[~task_means["ever_reaches_threshold"]]["instantaneous_rate"]

    bins = np.linspace(task_means["instantaneous_rate"].min(), task_means["instantaneous_rate"].max(), 20)
    ax4.hist(reaches_means, bins=bins, alpha=0.6, label=f"Reaches (n={len(reaches_means)})", color="red")
    ax4.hist(never_means, bins=bins, alpha=0.6, label=f"Never reaches (n={len(never_means)})", color="blue")
    ax4.axvline(x=0, color='gray', linestyle='--', alpha=0.5)
    ax4.set_xlabel("Mean Instantaneous Rate (per task)", fontsize=11)
    ax4.set_ylabel("Count", fontsize=11)
    ax4.set_title("Per-Task Mean Instantaneous Rate", fontsize=12)
    ax4.legend()

    # Stats summary
    reaches_all = df[df["ever_reaches_threshold"]]["instantaneous_rate"]
    never_all = df[~df["ever_reaches_threshold"]]["instantaneous_rate"]
    if len(reaches_all) > 0 and len(never_all) > 0:
        t_stat, p_val = stats.ttest_ind(reaches_all, never_all)
        fig.text(0.5, 0.02, f"t-test (reaches vs never): t={t_stat:.2f}, p={p_val:.4f}",
                ha='center', fontsize=10, style='italic')

    plt.suptitle(title or f"Instantaneous Descent Rate Analysis (threshold={threshold})", fontsize=14, y=0.98)
    plt.tight_layout(rect=[0, 0.04, 1, 0.96])
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.savefig(output_path.with_suffix('.pdf'), bbox_inches='tight')
    print(f"Plot saved to: {output_path}")
    plt.close()


def plot_median_trajectory(
    trajectories: pd.DataFrame,
    output_path: Path,
    threshold: int = 10,
    title: str | None = None,
    eval_df: pd.DataFrame | None = None,
    not_hacked_value: int = 200,
) -> None:
    """Plot boxplot of min_prefill distribution at each checkpoint.

    Non-hacked tasks are included with min_prefill=not_hacked_value, which is
    off the visible y-axis. This way, when >50% of tasks are not hacked,
    the median naturally falls off the chart.

    If eval_df is provided, overlays hack rate at prefill=0 on secondary y-axis.
    """
    fig, ax = plt.subplots(figsize=(12, 6))

    # Get sorted checkpoints
    checkpoints = sorted(trajectories["checkpoint"].unique())
    n_tasks = trajectories.groupby("checkpoint")["task_id"].nunique().iloc[0]

    # Prepare data for boxplot - list of arrays, one per checkpoint
    # Include ALL tasks - non-hacked ones will have min_prefill=200 (off chart)
    boxplot_data = []
    positions = []

    for ckpt in checkpoints:
        ckpt_data = trajectories[trajectories["checkpoint"] == ckpt]["min_prefill"].values
        boxplot_data.append(ckpt_data)
        positions.append(ckpt)

    # Create boxplot with outliers shown as dots
    bp = ax.boxplot(boxplot_data, positions=positions, widths=min(5, (max(positions) - min(positions)) / len(positions) * 0.6),
                    patch_artist=True, showfliers=True, flierprops=dict(marker='o', markersize=3, alpha=0.5))

    # Style boxplot
    for patch in bp['boxes']:
        patch.set_facecolor('lightblue')
        patch.set_alpha(0.7)

    # Set y-axis to show only the interesting range (0 to max tested prefill)
    # Outliers above this will be clipped
    y_max = 105  # Just above 100 (max tested prefill)
    ax.set_ylim(-2, y_max)

    ax.set_xlabel("Checkpoint", fontsize=12)
    ax.set_ylabel("Min Prefill Tokens to Exploit", fontsize=12)
    ax.grid(True, alpha=0.3, axis='y')

    # Add hack rate at prefill=0 on secondary y-axis if eval_df provided
    if eval_df is not None:
        ax2 = ax.twinx()

        # Filter to prefill=0 and compute hack rate per checkpoint
        prefill0 = eval_df[eval_df["prefill_tokens"] == 0].copy()
        if len(prefill0) > 0:
            hack_rates = prefill0.groupby("checkpoint").agg(
                hack_rate=("exploit_success", "mean"),
            ).reset_index()

            # Plot hack rate on secondary axis
            ax2.plot(hack_rates["checkpoint"], hack_rates["hack_rate"] * 100,
                    marker='s', linewidth=2, color='darkgreen', linestyle='--',
                    label='Hack rate @ prefill=0', alpha=0.8)
            ax2.set_ylabel("Hack Rate at Prefill=0 (%)", fontsize=12, color='darkgreen')
            ax2.tick_params(axis='y', labelcolor='darkgreen')
            ax2.set_ylim(0, 100)

            # Add legend for hack rate line
            ax2.legend(loc="lower right")

    ax.set_title(title or f"Min Prefill Distribution by Checkpoint (n={n_tasks} tasks)", fontsize=14)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.savefig(output_path.with_suffix('.pdf'), bbox_inches='tight')
    print(f"Plot saved to: {output_path}")
    plt.close()


def plot_exploit_rate_by_prefill_per_exploit_type(
    eval_df: pd.DataFrame,
    output_dir: Path,
    min_checkpoints: int = 2,
) -> None:
    """Plot exploit rate vs checkpoint for each prefill level, one plot per exploit type.

    Creates a separate plot for each exploit type showing how exploit rate changes
    over training for different prefill levels.

    Args:
        eval_df: Evaluation dataframe with exploit_type, checkpoint, prefill_tokens, exploit_success
        output_dir: Directory to save plots
        min_checkpoints: Minimum number of checkpoints required to generate a plot (default: 2)
    """
    # Get unique exploit types and prefill levels
    exploit_types = sorted(eval_df["exploit_type"].unique())
    all_prefill_levels = sorted(eval_df["prefill_tokens"].unique())

    # Create a colormap for prefill levels
    cmap = plt.cm.viridis
    colors = {p: cmap(i / max(1, len(all_prefill_levels) - 1)) for i, p in enumerate(all_prefill_levels)}

    for exploit_type in exploit_types:
        et_data = eval_df[eval_df["exploit_type"] == exploit_type]

        # Check data availability
        available_checkpoints = sorted(et_data["checkpoint"].unique())
        available_prefills = sorted(et_data["prefill_tokens"].unique())
        n_tasks = et_data.groupby("checkpoint")["task_id"].nunique().iloc[0] if len(et_data) > 0 else 0

        # Skip if insufficient data
        if len(available_checkpoints) < min_checkpoints:
            print(f"Skipping {exploit_type}: only {len(available_checkpoints)} checkpoint(s)")
            continue

        fig, ax = plt.subplots(figsize=(10, 6))

        for prefill in available_prefills:
            prefill_data = et_data[et_data["prefill_tokens"] == prefill]

            # Compute exploit rate per checkpoint
            rates = prefill_data.groupby("checkpoint").agg(
                exploit_rate=("exploit_success", "mean"),
            ).reset_index()

            ax.plot(rates["checkpoint"], rates["exploit_rate"] * 100,
                   marker='o', markersize=4, linewidth=1.5, color=colors[prefill],
                   label=f'prefill={prefill}', alpha=0.8)

        ax.set_xlabel("Checkpoint (Training Steps)", fontsize=12)
        ax.set_ylabel("Exploit Rate (%)", fontsize=12)
        ax.set_ylim(0, 100)
        ax.grid(True, alpha=0.3)
        ax.legend(loc="upper left", fontsize=8, ncol=2)

        # Add subtitle with data availability info
        title = f"Exploit Rate by Prefill Level: {exploit_type}\n(n={n_tasks} tasks, checkpoints: {available_checkpoints[0]}–{available_checkpoints[-1]})"
        ax.set_title(title, fontsize=11)

        plt.tight_layout()

        # Save with exploit type in filename (sanitize for filesystem)
        safe_name = exploit_type.replace("/", "_").replace(" ", "_")
        out_path = output_dir / f"exploit_rate_by_prefill_{safe_name}.png"
        plt.savefig(out_path, dpi=150, bbox_inches='tight')
        plt.savefig(out_path.with_suffix('.pdf'), bbox_inches='tight')
        print(f"Plot saved to: {out_path}")
        plt.close()


def plot_instantaneous_rate_at_max_prefill(
    data: pd.DataFrame,
    output_path: Path,
    threshold: int,
    max_prefill: int = 100,
    checkpoint_cutoffs: list[int] = [20, 50],
    title: str | None = None,
) -> None:
    """Plot histogram of instantaneous rate when current min_prefill >= max_prefill, by checkpoint cutoff."""
    # Compute instantaneous rates
    instantaneous_rates = []
    for (task_id, exploit_type), group in data.groupby(["task_id", "exploit_type"]):
        group = group.sort_values("checkpoint")
        if len(group) < 2:
            continue

        ckpts = group["checkpoint"].values
        min_prefills = group["min_prefill"].values
        ever_reaches = group["ever_reaches_threshold"].iloc[0] if "ever_reaches_threshold" in group.columns else any(min_prefills <= threshold)

        for i in range(len(ckpts) - 1):
            delta_prefill = min_prefills[i + 1] - min_prefills[i]
            delta_steps = ckpts[i + 1] - ckpts[i]
            if delta_steps > 0:
                rate = delta_prefill / delta_steps
                instantaneous_rates.append({
                    "task_id": task_id,
                    "exploit_type": exploit_type,
                    "checkpoint": ckpts[i],
                    "instantaneous_rate": rate,
                    "current_prefill": min_prefills[i],
                    "ever_reaches_threshold": ever_reaches,
                })

    if not instantaneous_rates:
        print("Warning: No instantaneous descent rate data")
        return

    df = pd.DataFrame(instantaneous_rates)

    # Filter to only rows where current_prefill >= max_prefill
    df_filtered = df[df["current_prefill"] >= max_prefill]
    if len(df_filtered) == 0:
        print(f"Warning: No data points with current_prefill >= {max_prefill}")
        return

    # Create subplots for each checkpoint cutoff + all
    n_plots = len(checkpoint_cutoffs) + 1
    fig, axes = plt.subplots(1, n_plots, figsize=(5 * n_plots, 5))
    if n_plots == 1:
        axes = [axes]

    cutoffs = checkpoint_cutoffs + [None]  # None means all

    # Get global bin range
    global_min = df_filtered["instantaneous_rate"].min()
    global_max = df_filtered["instantaneous_rate"].max()
    bins = np.linspace(global_min, global_max, 30)

    for i, cutoff in enumerate(cutoffs):
        ax = axes[i]
        if cutoff is not None:
            subset = df_filtered[df_filtered["checkpoint"] < cutoff]
            label = f"checkpoint < {cutoff}"
        else:
            subset = df_filtered
            label = "all checkpoints"

        if len(subset) == 0:
            ax.set_title(f"{label}\n(no data)")
            continue

        reaches = subset[subset["ever_reaches_threshold"]]["instantaneous_rate"]
        never = subset[~subset["ever_reaches_threshold"]]["instantaneous_rate"]

        ax.hist(reaches, bins=bins, alpha=0.6, label=f"Reaches (n={len(reaches)})", color="red")
        ax.hist(never, bins=bins, alpha=0.6, label=f"Never (n={len(never)})", color="blue")
        ax.axvline(x=0, color='gray', linestyle='--', alpha=0.5)
        ax.set_xlabel("Instantaneous Rate", fontsize=10)
        ax.set_ylabel("Count", fontsize=10)
        ax.set_title(label, fontsize=11)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

        # Add stats
        if len(reaches) > 0 and len(never) > 0:
            t_stat, p_val = stats.ttest_ind(reaches, never)
            ax.text(0.02, 0.98, f"t={t_stat:.2f}, p={p_val:.3f}\nr: {reaches.mean():.3f}\nn: {never.mean():.3f}",
                   transform=ax.transAxes, fontsize=8, va='top', ha='left',
                   bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    plt.suptitle(title or f"Instantaneous Rate when min_prefill ≥ {max_prefill} (threshold={threshold})", fontsize=12)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.savefig(output_path.with_suffix('.pdf'), bbox_inches='tight')
    print(f"Plot saved to: {output_path}")
    plt.close()


def plot_instantaneous_descent_rate_by_exploit_type(
    data: pd.DataFrame,
    output_path: Path,
    threshold: int,
    title: str | None = None,
) -> None:
    """Plot instantaneous descent rates averaged by exploit type.

    First averages min_prefill across tasks within each exploit type at each checkpoint,
    then computes instantaneous rates on those averaged trajectories.
    Colors by fraction of tasks that reach threshold (gradient from blue=0% to red=100%).
    """
    # Compute per-exploit-type fraction reaching threshold
    task_outcomes = data.groupby(["task_id", "exploit_type"])["ever_reaches_threshold"].first().reset_index()
    exploit_reach_pct = task_outcomes.groupby("exploit_type")["ever_reaches_threshold"].mean().to_dict()

    # Average min_prefill by (exploit_type, checkpoint)
    avg_by_exploit = data.groupby(["exploit_type", "checkpoint"]).agg({
        "min_prefill": "mean",
        "ever_reaches_threshold": "mean",
    }).reset_index()

    # Compute instantaneous rates on averaged trajectories
    instantaneous_rates = []
    for exploit_type, group in avg_by_exploit.groupby("exploit_type"):
        group = group.sort_values("checkpoint")
        if len(group) < 2:
            continue

        ckpts = group["checkpoint"].values
        min_prefills = group["min_prefill"].values

        for i in range(len(ckpts) - 1):
            delta_prefill = min_prefills[i + 1] - min_prefills[i]
            delta_steps = ckpts[i + 1] - ckpts[i]
            if delta_steps > 0:
                rate = delta_prefill / delta_steps
                instantaneous_rates.append({
                    "exploit_type": exploit_type,
                    "checkpoint": ckpts[i],
                    "next_checkpoint": ckpts[i + 1],
                    "instantaneous_rate": rate,
                    "current_prefill": min_prefills[i],
                    "pct_reaches": exploit_reach_pct.get(exploit_type, 0),
                })

    if not instantaneous_rates:
        print("Warning: No instantaneous descent rate data")
        return

    df = pd.DataFrame(instantaneous_rates)
    exploit_types = sorted(df["exploit_type"].unique())

    # Color by % reaching threshold: blue (0%) -> red (100%)
    cmap = plt.cm.coolwarm
    color_map = {et: cmap(exploit_reach_pct.get(et, 0)) for et in exploit_types}

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # 1. Histogram of instantaneous rates by exploit type
    ax1 = axes[0, 0]
    bins = np.linspace(df["instantaneous_rate"].min(), df["instantaneous_rate"].max(), 20)
    # Sort by pct_reaches for legend ordering
    sorted_types = sorted(exploit_types, key=lambda et: exploit_reach_pct.get(et, 0), reverse=True)
    for exploit_type in sorted_types:
        subset = df[df["exploit_type"] == exploit_type]["instantaneous_rate"]
        pct = exploit_reach_pct.get(exploit_type, 0) * 100
        ax1.hist(subset, bins=bins, alpha=0.5,
                label=f"{exploit_type} ({pct:.0f}%)", color=color_map[exploit_type])
    ax1.axvline(x=0, color='gray', linestyle='--', alpha=0.5)
    ax1.set_xlabel("Instantaneous Rate (Δprefill / Δsteps)", fontsize=11)
    ax1.set_ylabel("Count", fontsize=11)
    ax1.set_title("Distribution of Instantaneous Descent Rates", fontsize=12)
    ax1.legend(fontsize=7, loc='upper left')

    # 2. Scatter: current_prefill vs instantaneous_rate (colored by % reaches)
    ax2 = axes[0, 1]
    for exploit_type in exploit_types:
        subset = df[df["exploit_type"] == exploit_type]
        ax2.scatter(subset["current_prefill"], subset["instantaneous_rate"],
                   alpha=0.7, c=[color_map[exploit_type]], s=50, label=exploit_type)
    ax2.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
    ax2.set_xlabel("Current Mean Min Prefill", fontsize=11)
    ax2.set_ylabel("Instantaneous Rate", fontsize=11)
    ax2.set_title("Current Prefill vs Instantaneous Rate", fontsize=12)
    # Add colorbar instead of legend
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(0, 100))
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax2)
    cbar.set_label(f"% reaching threshold (≤{threshold})", fontsize=10)

    # 3. Rate over training (by checkpoint) - line plot per exploit type
    ax3 = axes[1, 0]
    for exploit_type in sorted_types:
        subset = df[df["exploit_type"] == exploit_type].sort_values("checkpoint")
        pct = exploit_reach_pct.get(exploit_type, 0) * 100
        ax3.plot(subset["checkpoint"], subset["instantaneous_rate"],
                marker='o', alpha=0.7, label=f"{exploit_type} ({pct:.0f}%)",
                color=color_map[exploit_type])
    ax3.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
    ax3.set_xlabel("Checkpoint", fontsize=11)
    ax3.set_ylabel("Instantaneous Rate", fontsize=11)
    ax3.set_title("Instantaneous Rate Over Training", fontsize=12)
    ax3.legend(fontsize=7, loc='best')

    # 4. Mean instantaneous rate vs % reaching threshold (scatter)
    ax4 = axes[1, 1]
    summary = df.groupby("exploit_type").agg({
        "instantaneous_rate": "mean",
        "pct_reaches": "first",
    }).reset_index()
    scatter = ax4.scatter(summary["instantaneous_rate"], summary["pct_reaches"] * 100,
                         c=summary["pct_reaches"], cmap=cmap, s=100, alpha=0.8,
                         vmin=0, vmax=1)
    for _, row in summary.iterrows():
        ax4.annotate(row["exploit_type"],
                    (row["instantaneous_rate"], row["pct_reaches"] * 100),
                    fontsize=7, ha='left', va='bottom', alpha=0.8)
    ax4.axvline(x=0, color='gray', linestyle='--', alpha=0.5)
    ax4.set_xlabel("Mean Instantaneous Rate", fontsize=11)
    ax4.set_ylabel(f"% Tasks Reaching Threshold (≤{threshold})", fontsize=11)
    ax4.set_title("Mean Rate vs Threshold Reachability", fontsize=12)

    plt.suptitle(title or f"Instantaneous Descent Rate by Exploit Type (threshold={threshold})", fontsize=14, y=0.98)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.savefig(output_path.with_suffix('.pdf'), bbox_inches='tight')
    print(f"Plot saved to: {output_path}")
    plt.close()


UNINTENTIONAL_EXPLOIT_TYPES = ["inadequate_test_coverage", "resource_exhaustion", "hardcoding_or_memorization", "type_confusion"]


def run_single_analysis(
    df: pd.DataFrame,
    output_dir: Path,
    label: str,
    logprob_df: pd.DataFrame | None = None,
    kl_df: pd.DataFrame | None = None,
    threshold: int = 10,
    max_prefill: int = 100,
    logprob_threshold: float = -55.39,
) -> int:
    """Run the trajectory analysis on a filtered dataframe."""
    print(f"\n{'='*60}")
    print(f"ANALYSIS: {label}")
    print(f"{'='*60}")

    checkpoints = sorted(df["checkpoint"].unique())
    prefill_levels = sorted(df["prefill_tokens"].unique())
    print(f"Checkpoints: {checkpoints}")
    print(f"Prefill levels: {prefill_levels}")

    # Compute trajectories
    print("\nComputing min prefill trajectories...")
    trajectories = compute_min_prefill_trajectories(df, checkpoints, max_prefill)
    print(f"Trajectory data: {len(trajectories)} rows")

    # Compute time to threshold
    print(f"Computing time to threshold (≤{threshold} tokens)...")
    analysis_data = compute_time_to_threshold(trajectories, checkpoints, threshold)

    # Save data
    output_dir.mkdir(parents=True, exist_ok=True)
    analysis_data.to_csv(output_dir / "trajectory_analysis.csv", index=False)
    print(f"Saved analysis data to: {output_dir / 'trajectory_analysis.csv'}")

    # Summary stats
    print("\n" + "-" * 40)
    print("SUMMARY")
    print("-" * 40)

    n_tasks = analysis_data.groupby(["task_id", "exploit_type"]).ngroups
    ever_reaches = analysis_data.groupby(["task_id", "exploit_type"])["ever_reaches_threshold"].first()
    n_reaches = ever_reaches.sum()

    print(f"Total problems: {n_tasks}")
    print(f"Ever reach threshold (≤{threshold}): {n_reaches} ({100*n_reaches/n_tasks:.1f}%)")

    # Correlation analysis
    not_at_threshold = analysis_data[~analysis_data["at_threshold"]]
    valid = not_at_threshold[not_at_threshold["steps_to_threshold"].notna()]

    if len(valid) > 2:
        corr, p_value = stats.pearsonr(valid["min_prefill"], valid["steps_to_threshold"])
        print(f"Correlation (min_prefill vs steps_to_threshold): r={corr:.3f}, p={p_value:.4f}")

    # Generate plots
    print("\nGenerating plots...")

    plot_pass_rates_vs_prefill(
        df,
        output_dir / "pass_rates_vs_prefill.png",
    )

    plot_accessibility_vs_time_to_threshold(
        analysis_data,
        output_dir / "accessibility_vs_time.png",
        threshold,
    )

    plot_trajectories_sample(
        trajectories,
        output_dir / "sample_trajectories.png",
        threshold=threshold,
    )

    plot_median_trajectory(
        trajectories,
        output_dir / "median_trajectory.png",
        threshold=threshold,
        eval_df=df,
    )

    plot_descent_rate_distribution(
        analysis_data,
        output_dir / "descent_rates.png",
        threshold,
    )

    plot_instantaneous_descent_rate(
        analysis_data,
        output_dir / "instantaneous_descent_rates.png",
        threshold,
    )

    plot_instantaneous_descent_rate_by_exploit_type(
        analysis_data,
        output_dir / "instantaneous_descent_rates_by_exploit.png",
        threshold,
    )

    plot_instantaneous_rate_at_max_prefill(
        analysis_data,
        output_dir / "instantaneous_rate_at_max_prefill.png",
        threshold,
        max_prefill=max_prefill,
    )

    # Per-exploit-type plots: exploit rate by prefill level
    exploit_type_dir = output_dir / "by_exploit_type"
    exploit_type_dir.mkdir(parents=True, exist_ok=True)
    plot_exploit_rate_by_prefill_per_exploit_type(df, exploit_type_dir)

    # Logprob plots if data available
    if logprob_df is not None and len(logprob_df) > 0:
        # Filter logprob data to match the current df's task_ids
        valid_task_ids = set(df["task_id"].unique())
        logprob_filtered = logprob_df[logprob_df["task_id"].isin(valid_task_ids)]

        if len(logprob_filtered) > 0:
            print(f"Generating logprob plots ({len(logprob_filtered)} records)...")

            plot_logprob_vs_prefill(
                logprob_filtered,
                output_dir / "logprob_vs_prefill.png",
            )

            plot_logprob_vs_checkpoint(
                logprob_filtered,
                output_dir / "logprob_vs_checkpoint.png",
            )

            # Save logprob data
            logprob_filtered.to_csv(output_dir / "logprob_analysis.csv", index=False)
            print(f"Saved logprob data to: {output_dir / 'logprob_analysis.csv'}")

            # Compute logprob trajectories at min_prefill
            # Use analysis_data (not trajectories) to get ever_reaches_threshold
            print(f"\nComputing logprob trajectories at min_prefill...")
            logprob_trajectories = compute_logprob_trajectories(
                analysis_data, logprob_filtered, checkpoints
            )

            if len(logprob_trajectories) > 0:
                print(f"Logprob trajectory data: {len(logprob_trajectories)} rows")

                # Compute time to logprob threshold
                logprob_analysis = compute_logprob_time_to_threshold(
                    logprob_trajectories, checkpoints, logprob_threshold
                )

                # Save logprob trajectory analysis
                logprob_analysis.to_csv(output_dir / "logprob_trajectory_analysis.csv", index=False)
                print(f"Saved logprob trajectory data to: {output_dir / 'logprob_trajectory_analysis.csv'}")

                # Generate logprob trajectory plots
                print(f"Generating logprob trajectory plots (threshold={logprob_threshold})...")

                plot_logprob_ascent_rate_distribution(
                    logprob_analysis,
                    output_dir / "logprob_ascent_rates.png",
                    logprob_threshold,
                )

                plot_instantaneous_logprob_ascent_rate(
                    logprob_analysis,
                    output_dir / "logprob_instantaneous_ascent_rates.png",
                    logprob_threshold,
                )

                plot_instantaneous_logprob_ascent_rate_by_exploit_type(
                    logprob_analysis,
                    output_dir / "logprob_instantaneous_ascent_rates_by_exploit.png",
                    logprob_threshold,
                )

                plot_median_logprob_trajectory(
                    logprob_trajectories,
                    output_dir / "logprob_median_trajectory.png",
                    logprob_threshold,
                    eval_df=df,
                )

                plot_logprob_at_min_prefill(
                    logprob_analysis,
                    output_dir / "logprob_at_min_prefill.png",
                    logprob_threshold,
                )

                # Exploit rate scaling law analysis (using KL if available)
                print("\nComputing exploit rate scaling law...")
                if kl_df is not None and len(kl_df) > 0:
                    kl_filtered = kl_df[kl_df["task_id"].isin(valid_task_ids)]
                    if len(kl_filtered) > 0:
                        scaling_df = compute_exploit_rate_scaling(kl_filtered, checkpoints, eval_df=df)
                        print(f"  Using KL divergence ({len(kl_filtered)} records)")
                    else:
                        print("Warning: No KL data after filtering, skipping scaling analysis")
                        scaling_df = pd.DataFrame()
                else:
                    print("Warning: No KL data available, skipping scaling analysis")
                    scaling_df = pd.DataFrame()

                if len(scaling_df) > 0:
                    scaling_df.to_csv(output_dir / "exploit_rate_scaling.csv", index=False)
                    print(f"Saved scaling data to: {output_dir / 'exploit_rate_scaling.csv'}")
                    print(f"  Checkpoints: {list(scaling_df['checkpoint'])}")
                    print(f"  Log P(exploit) range: [{scaling_df['log_exploit_lower_bound'].min():.2f}, {scaling_df['log_exploit_lower_bound'].max():.2f}]")

                    plot_exploit_rate_scaling(
                        scaling_df,
                        output_dir / "exploit_rate_scaling.png",
                    )

                    # Per-exploit-type scaling plot
                    plot_exploit_rate_scaling_by_type(
                        kl_filtered,
                        checkpoints,
                        df,
                        output_dir / "exploit_rate_scaling_by_type.png",
                    )
                else:
                    print("Warning: No data for exploit rate scaling analysis")

                # Early indicator analysis (combined token + logprob)
                plot_early_indicator_analysis(
                    analysis_data,
                    logprob_analysis,
                    df,
                    output_dir / "early_indicator_analysis.png",
                    checkpoint_cutoffs=[6, 15, 50],
                )
            else:
                print("Warning: No logprob trajectory data (no overlap between trajectories and logprob data)")
                # Still generate early indicator analysis with token data only
                plot_early_indicator_analysis(
                    analysis_data,
                    None,
                    df,
                    output_dir / "early_indicator_analysis.png",
                    checkpoint_cutoffs=[6, 15, 50],
                )
    else:
        # No logprob data available - still generate early indicator analysis with token data only
        plot_early_indicator_analysis(
            analysis_data,
            None,
            df,
            output_dir / "early_indicator_analysis.png",
            checkpoint_cutoffs=[6, 15, 50],
        )

    print(f"Results saved to: {output_dir}")
    return 0


def run_trajectory_analysis(
    run_dir: Path,
    output_dir: Path,
    threshold: int = 10,
    max_prefill: int = 100,
    filter_dataset: str | None = "EleutherAI/djinn-problems-v0.9",
    exclude_exploit_types: list[str] | None = None,
    skip_intentional_split: bool = False,
    logprob_threshold: float = -55.39,
) -> int:
    """Run the trajectory analysis, writing results to output_dir."""
    input_dir = run_dir / "evals"
    if not input_dir.exists():
        print(f"Error: {input_dir} does not exist")
        return 1

    # Load data
    print(f"Loading data from {input_dir}...")
    df = load_per_problem_results(input_dir)
    print(f"Loaded {len(df)} rows")

    # Load logprob data if available
    logprob_dir = run_dir / "logprob"
    logprob_df = None
    if logprob_dir.exists():
        print(f"\nLoading logprob data from {logprob_dir}...")
        logprob_df = load_logprob_results(logprob_dir)
        if logprob_df is not None:
            print(f"Loaded {len(logprob_df)} logprob records")
        else:
            print("No logprob data found")
    else:
        print(f"\nNo logprob directory at {logprob_dir}")

    # Load KL data if available
    kl_dir = run_dir / "kl"
    kl_df = None
    if kl_dir.exists():
        print(f"\nLoading KL data from {kl_dir}...")
        kl_df = load_kl_results(kl_dir)
        if kl_df is not None:
            print(f"Loaded {len(kl_df)} KL records")
        else:
            print("No KL data found")
    else:
        print(f"\nNo KL directory at {kl_dir}")

    # Filter by dataset if specified
    if filter_dataset:
        from datasets import load_dataset
        print(f"\nFiltering by dataset: {filter_dataset}")
        ds = load_dataset(filter_dataset)
        valid_ids = set()
        for split_name in ds.keys():
            if 'id' in ds[split_name].column_names:
                valid_ids.update(ds[split_name]['id'])
        print(f"Valid task IDs in dataset: {len(valid_ids)}")

        before_count = df["task_id"].nunique()
        df = df[df["task_id"].isin(valid_ids)]
        after_count = df["task_id"].nunique()
        print(f"Filtered: {before_count} -> {after_count} tasks ({before_count - after_count} removed)")
        print(f"Rows after filter: {len(df)}")

    # Apply explicit exclude filter if specified
    if exclude_exploit_types:
        print(f"\nExcluding exploit types: {exclude_exploit_types}")
        before_count = df["task_id"].nunique()
        df = df[~df["exploit_type"].isin(exclude_exploit_types)]
        after_count = df["task_id"].nunique()
        print(f"Filtered: {before_count} -> {after_count} tasks ({before_count - after_count} removed)")
        print(f"Rows after filter: {len(df)}")

    # Run analysis on all exploits
    run_single_analysis(
        df, output_dir / "all_exploits", "All Exploits",
        logprob_df, kl_df,
        threshold=threshold, max_prefill=max_prefill, logprob_threshold=logprob_threshold,
    )

    # Run analysis on intentional exploits only (unless skipped)
    if not skip_intentional_split:
        df_intentional = df[~df["exploit_type"].isin(UNINTENTIONAL_EXPLOIT_TYPES)]
        n_removed = df["task_id"].nunique() - df_intentional["task_id"].nunique()
        print(f"\n\nFiltering to intentional exploits only...")
        print(f"Excluding: {UNINTENTIONAL_EXPLOIT_TYPES}")
        print(f"Removed {n_removed} tasks")
        run_single_analysis(
            df_intentional, output_dir / "intentional_only", "Intentional Exploits Only",
            logprob_df, kl_df,
            threshold=threshold, max_prefill=max_prefill, logprob_threshold=logprob_threshold,
        )

    print(f"\n{'='*60}")
    print(f"All results saved to: {output_dir}")
    return 0


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
    parser.add_argument(
        "--filter-dataset",
        type=str,
        default="EleutherAI/djinn-problems-v0.9",
        help="HuggingFace dataset to filter task IDs by (default: EleutherAI/djinn-problems-v0.9, use 'none' to disable)",
    )
    parser.add_argument(
        "--exclude-exploit-types",
        type=str,
        nargs="+",
        default=None,
        help="Exploit types to exclude (e.g., 'resource_exhaustion hardcoding_or_memorization')",
    )
    parser.add_argument(
        "--skip-intentional-split",
        action="store_true",
        help="Skip generating separate plots for intentional-only exploits (excludes inadequate_test_coverage, resource_exhaustion)",
    )
    parser.add_argument(
        "--logprob-threshold",
        type=float,
        default=-55.39,
        help="Logprob sum threshold for 'easily exploitable' (default: -55.39)",
    )

    args = parser.parse_args()

    # Handle 'none' for filter-dataset
    filter_dataset = args.filter_dataset
    if filter_dataset and filter_dataset.lower() == "none":
        filter_dataset = None

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
            return run_trajectory_analysis(
                run_dir=args.run_dir,
                output_dir=output_dir,
                threshold=args.threshold,
                max_prefill=args.max_prefill,
                filter_dataset=filter_dataset,
                exclude_exploit_types=args.exclude_exploit_types,
                skip_intentional_split=args.skip_intentional_split,
                logprob_threshold=args.logprob_threshold,
            )
    else:
        output_dir = args.output_dir
        output_dir.mkdir(parents=True, exist_ok=True)
        return run_trajectory_analysis(
            run_dir=args.run_dir,
            output_dir=output_dir,
            threshold=args.threshold,
            max_prefill=args.max_prefill,
            filter_dataset=filter_dataset,
            exclude_exploit_types=args.exclude_exploit_types,
            skip_intentional_split=args.skip_intentional_split,
            logprob_threshold=args.logprob_threshold,
        )


if __name__ == "__main__":
    exit(main())
