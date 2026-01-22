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
                        "secure_pass": record.get("secure_pass", False),
                        "insecure_pass": record.get("insecure_pass", False),
                    })
    return pd.DataFrame(rows)


def load_logprob_results(logprob_dir: Path) -> pd.DataFrame | None:
    """Load logprob results from checkpoint × prefill logprob files."""
    if not logprob_dir.exists():
        return None

    rows = []
    for jsonl_file in sorted(logprob_dir.glob("checkpoint-*_prefill*.jsonl")):
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
                        "exploit_type": record.get("exploit_type"),
                        "attempt_idx": record.get("attempt_idx", 0),
                        "prefill_logprob_sum": record.get("prefill_logprob_sum"),
                        "prefill_logprob_mean": record.get("prefill_logprob_mean"),
                        "prefill_num_tokens": record.get("prefill_num_tokens"),
                        "exploit_success": record.get("exploit_success", False),
                    })

    if not rows:
        return None
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


def compute_logprob_trajectories(
    trajectories: pd.DataFrame,
    logprob_df: pd.DataFrame,
    checkpoints: list[int],
) -> pd.DataFrame:
    """Compute logprob_sum at min_prefill for each problem at each checkpoint.

    For each task/checkpoint pair, looks up the logprob_sum at that task's min_prefill level.
    This captures how natural the exploit reasoning appears at exactly the point where
    it becomes effective.

    Returns DataFrame with columns:
        task_id, exploit_type, checkpoint, min_prefill, logprob_sum, logprob_mean
    """
    results = []

    # Create lookup: (task_id, checkpoint, prefill_tokens) -> logprob metrics
    # Average across attempts for each (task_id, checkpoint, prefill_tokens)
    logprob_agg = logprob_df.groupby(["task_id", "checkpoint", "prefill_tokens"]).agg({
        "prefill_logprob_sum": "mean",
        "prefill_logprob_mean": "mean",
    }).reset_index()

    logprob_lookup = {}
    for _, row in logprob_agg.iterrows():
        key = (row["task_id"], row["checkpoint"], row["prefill_tokens"])
        logprob_lookup[key] = {
            "logprob_sum": row["prefill_logprob_sum"],
            "logprob_mean": row["prefill_logprob_mean"],
        }

    # For each task/checkpoint in trajectories, get logprob at min_prefill
    for _, row in trajectories.iterrows():
        task_id = row["task_id"]
        exploit_type = row["exploit_type"]
        checkpoint = row["checkpoint"]
        min_prefill = row["min_prefill"]

        # Only look up if the task is exploitable AND needs prefill (min_prefill > 0)
        # Exclude min_prefill=0: those tasks exploit without any reasoning, so there's
        # no logprob to measure - they're a different category, not "logprob=0"
        if row.get("exploitable", min_prefill <= 100) and min_prefill > 0:
            key = (task_id, checkpoint, min_prefill)
            if key in logprob_lookup:
                logprob_data = logprob_lookup[key]
                results.append({
                    "task_id": task_id,
                    "exploit_type": exploit_type,
                    "checkpoint": checkpoint,
                    "min_prefill": min_prefill,
                    "logprob_sum": logprob_data["logprob_sum"],
                    "logprob_mean": logprob_data["logprob_mean"],
                })

    return pd.DataFrame(results)


def compute_logprob_time_to_threshold(
    logprob_trajectories: pd.DataFrame,
    checkpoints: list[int],
    threshold: float = -55.39,
) -> pd.DataFrame:
    """For each (task, checkpoint), compute steps until logprob_sum >= threshold.

    The threshold is the logprob_sum value at which exploit reasoning is considered
    "easily natural" to the model. Default is -55.39 from prior analysis.

    Returns DataFrame with:
        task_id, exploit_type, checkpoint, logprob_sum, steps_to_logprob_threshold,
        ever_reaches_logprob_threshold, at_logprob_threshold
    """
    results = []

    for (task_id, exploit_type), group in logprob_trajectories.groupby(["task_id", "exploit_type"]):
        group = group.sort_values("checkpoint")
        ckpts = group["checkpoint"].values
        logprob_sums = group["logprob_sum"].values

        # Check if task ever reaches threshold (higher = more natural, so >= threshold)
        ever_reaches = any(lp >= threshold for lp in logprob_sums)

        for i, (ckpt, lp) in enumerate(zip(ckpts, logprob_sums)):
            # Find steps until threshold is reached
            steps_to_threshold = None
            for j in range(i, len(ckpts)):
                if logprob_sums[j] >= threshold:
                    steps_to_threshold = ckpts[j] - ckpt
                    break

            results.append({
                "task_id": task_id,
                "exploit_type": exploit_type,
                "checkpoint": ckpt,
                "logprob_sum": lp,
                "steps_to_logprob_threshold": steps_to_threshold,
                "ever_reaches_logprob_threshold": ever_reaches,
                "at_logprob_threshold": lp >= threshold,
            })

    return pd.DataFrame(results)


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
            ever_reaches = group["ever_reaches_logprob_threshold"].iloc[0] if "ever_reaches_logprob_threshold" in group.columns else any(logprob_sums >= threshold)
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
        ever_reaches = group["ever_reaches_logprob_threshold"].iloc[0] if "ever_reaches_logprob_threshold" in group.columns else any(logprob_sums >= threshold)

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
    # Compute per-exploit-type fraction reaching threshold
    task_outcomes = logprob_data.groupby(["task_id", "exploit_type"])["ever_reaches_logprob_threshold"].first().reset_index()
    exploit_reach_pct = task_outcomes.groupby("exploit_type")["ever_reaches_logprob_threshold"].mean().to_dict()

    # Average logprob_sum by (exploit_type, checkpoint)
    avg_by_exploit = logprob_data.groupby(["exploit_type", "checkpoint"]).agg({
        "logprob_sum": "mean",
        "ever_reaches_logprob_threshold": "mean",
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
) -> None:
    """Plot median logprob_sum trajectory with IQR band over checkpoints."""
    fig, ax = plt.subplots(figsize=(10, 6))

    # Compute median and IQR at each checkpoint
    stats_by_ckpt = logprob_trajectories.groupby("checkpoint")["logprob_sum"].agg(
        median="median",
        q25=lambda x: x.quantile(0.25),
        q75=lambda x: x.quantile(0.75),
        count="count",
    ).reset_index()

    # Plot median with IQR band
    ax.plot(stats_by_ckpt["checkpoint"], stats_by_ckpt["median"],
            marker='o', linewidth=2, color='black', label='Median')
    ax.fill_between(stats_by_ckpt["checkpoint"],
                    stats_by_ckpt["q25"],
                    stats_by_ckpt["q75"],
                    alpha=0.3, color='gray', label='IQR (25-75%)')

    # Add threshold line
    ax.axhline(y=threshold, color='red', linestyle='--', alpha=0.7,
               label=f'Threshold ({threshold})')

    ax.set_xlabel("Checkpoint", fontsize=12)
    ax.set_ylabel("Median Logprob Sum at Min Prefill", fontsize=12)
    ax.set_title(title or f"Median Logprob Trajectory Across All Tasks (n={stats_by_ckpt['count'].iloc[0]})", fontsize=14)
    ax.legend(loc="best")
    ax.grid(True, alpha=0.3)

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

        reaches = subset[subset["ever_reaches_logprob_threshold"]]["logprob_sum"]
        never = subset[~subset["ever_reaches_logprob_threshold"]]["logprob_sum"]

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
) -> None:
    """Plot median min_prefill trajectory across all tasks over checkpoints."""
    fig, ax = plt.subplots(figsize=(10, 6))

    # Compute median and IQR at each checkpoint
    stats_by_ckpt = trajectories.groupby("checkpoint")["min_prefill"].agg(
        median="median",
        q25=lambda x: x.quantile(0.25),
        q75=lambda x: x.quantile(0.75),
        count="count",
    ).reset_index()

    # Plot median with IQR band
    ax.plot(stats_by_ckpt["checkpoint"], stats_by_ckpt["median"],
            marker='o', linewidth=2, color='black', label='Median')
    ax.fill_between(stats_by_ckpt["checkpoint"],
                    stats_by_ckpt["q25"],
                    stats_by_ckpt["q75"],
                    alpha=0.3, color='gray', label='IQR (25-75%)')

    # Add threshold line
    ax.axhline(y=threshold, color='red', linestyle='--', alpha=0.7,
               label=f'Threshold ({threshold} tokens)')

    ax.set_xlabel("Checkpoint", fontsize=12)
    ax.set_ylabel("Median Min Prefill Tokens to Exploit", fontsize=12)
    ax.set_title(title or f"Median Trajectory Across All Tasks (n={stats_by_ckpt['count'].iloc[0]})", fontsize=14)
    ax.legend(loc="best")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.savefig(output_path.with_suffix('.pdf'), bbox_inches='tight')
    print(f"Plot saved to: {output_path}")
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
    if args.filter_dataset and args.filter_dataset.lower() == "none":
        args.filter_dataset = None

    input_dir = args.run_dir / "evals"
    if not input_dir.exists():
        print(f"Error: {input_dir} does not exist")
        return 1

    # Unintentional exploit types to exclude for "intentional only" analysis
    UNINTENTIONAL_EXPLOIT_TYPES = ["inadequate_test_coverage", "resource_exhaustion"]

    def run_single_analysis(df: pd.DataFrame, output_dir: Path, label: str, logprob_df: pd.DataFrame | None = None) -> int:
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
        trajectories = compute_min_prefill_trajectories(df, checkpoints, args.max_prefill)
        print(f"Trajectory data: {len(trajectories)} rows")

        # Compute time to threshold
        print(f"Computing time to threshold (≤{args.threshold} tokens)...")
        analysis_data = compute_time_to_threshold(trajectories, checkpoints, args.threshold)

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
        print(f"Ever reach threshold (≤{args.threshold}): {n_reaches} ({100*n_reaches/n_tasks:.1f}%)")

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
            args.threshold,
        )

        plot_trajectories_sample(
            trajectories,
            output_dir / "sample_trajectories.png",
            threshold=args.threshold,
        )

        plot_median_trajectory(
            trajectories,
            output_dir / "median_trajectory.png",
            threshold=args.threshold,
        )

        plot_descent_rate_distribution(
            analysis_data,
            output_dir / "descent_rates.png",
            args.threshold,
        )

        plot_instantaneous_descent_rate(
            analysis_data,
            output_dir / "instantaneous_descent_rates.png",
            args.threshold,
        )

        plot_instantaneous_descent_rate_by_exploit_type(
            analysis_data,
            output_dir / "instantaneous_descent_rates_by_exploit.png",
            args.threshold,
        )

        plot_instantaneous_rate_at_max_prefill(
            analysis_data,
            output_dir / "instantaneous_rate_at_max_prefill.png",
            args.threshold,
            max_prefill=args.max_prefill,
        )

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
                print(f"\nComputing logprob trajectories at min_prefill...")
                logprob_trajectories = compute_logprob_trajectories(
                    trajectories, logprob_filtered, checkpoints
                )

                if len(logprob_trajectories) > 0:
                    print(f"Logprob trajectory data: {len(logprob_trajectories)} rows")

                    # Compute time to logprob threshold
                    logprob_analysis = compute_logprob_time_to_threshold(
                        logprob_trajectories, checkpoints, args.logprob_threshold
                    )

                    # Save logprob trajectory analysis
                    logprob_analysis.to_csv(output_dir / "logprob_trajectory_analysis.csv", index=False)
                    print(f"Saved logprob trajectory data to: {output_dir / 'logprob_trajectory_analysis.csv'}")

                    # Generate logprob trajectory plots
                    print(f"Generating logprob trajectory plots (threshold={args.logprob_threshold})...")

                    plot_logprob_ascent_rate_distribution(
                        logprob_analysis,
                        output_dir / "logprob_ascent_rates.png",
                        args.logprob_threshold,
                    )

                    plot_instantaneous_logprob_ascent_rate(
                        logprob_analysis,
                        output_dir / "logprob_instantaneous_ascent_rates.png",
                        args.logprob_threshold,
                    )

                    plot_instantaneous_logprob_ascent_rate_by_exploit_type(
                        logprob_analysis,
                        output_dir / "logprob_instantaneous_ascent_rates_by_exploit.png",
                        args.logprob_threshold,
                    )

                    plot_median_logprob_trajectory(
                        logprob_trajectories,
                        output_dir / "logprob_median_trajectory.png",
                        args.logprob_threshold,
                    )

                    plot_logprob_at_min_prefill(
                        logprob_analysis,
                        output_dir / "logprob_at_min_prefill.png",
                        args.logprob_threshold,
                    )
                else:
                    print("Warning: No logprob trajectory data (no overlap between trajectories and logprob data)")

        print(f"Results saved to: {output_dir}")
        return 0

    def run_analysis(output_dir: Path) -> int:
        """Run the trajectory analysis, writing results to output_dir."""
        # Load data
        print(f"Loading data from {input_dir}...")
        df = load_per_problem_results(input_dir)
        print(f"Loaded {len(df)} rows")

        # Load logprob data if available
        logprob_dir = args.run_dir / "logprob"
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

        # Filter by dataset if specified
        if args.filter_dataset:
            from datasets import load_dataset
            print(f"\nFiltering by dataset: {args.filter_dataset}")
            ds = load_dataset(args.filter_dataset)
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
        if args.exclude_exploit_types:
            print(f"\nExcluding exploit types: {args.exclude_exploit_types}")
            before_count = df["task_id"].nunique()
            df = df[~df["exploit_type"].isin(args.exclude_exploit_types)]
            after_count = df["task_id"].nunique()
            print(f"Filtered: {before_count} -> {after_count} tasks ({before_count - after_count} removed)")
            print(f"Rows after filter: {len(df)}")

        # Run analysis on all exploits
        run_single_analysis(df, output_dir / "all_exploits", "All Exploits", logprob_df)

        # Run analysis on intentional exploits only (unless skipped)
        if not args.skip_intentional_split:
            df_intentional = df[~df["exploit_type"].isin(UNINTENTIONAL_EXPLOIT_TYPES)]
            n_removed = df["task_id"].nunique() - df_intentional["task_id"].nunique()
            print(f"\n\nFiltering to intentional exploits only...")
            print(f"Excluding: {UNINTENTIONAL_EXPLOIT_TYPES}")
            print(f"Removed {n_removed} tasks")
            run_single_analysis(df_intentional, output_dir / "intentional_only", "Intentional Exploits Only", logprob_df)

        print(f"\n{'='*60}")
        print(f"All results saved to: {output_dir}")
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
