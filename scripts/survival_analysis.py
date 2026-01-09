#!/usr/bin/env python3
"""
Survival analysis of prefill sensitivity as a leading indicator.

This script tests whether early prefill sensitivity predicts time-to-first-exploit
during fine-tuning, using Cox proportional hazards models and Kaplan-Meier curves.

Key concepts:
- "Event": First successful exploit on a problem (at baseline, no prefill kick)
- "Time": Training checkpoint number (1, 10, 17, 27, 35, 44, 56, ...)
- "Covariate": Prefill sensitivity score at an early checkpoint
- "Censoring": Problems that never get exploited are right-censored

Usage:
    python scripts/survival_analysis.py \
        --results-dir results/prefill_sensitivity \
        --early-checkpoint 10 \
        --output-dir results/survival_analysis

    # Use multiple runs (will aggregate data)
    python scripts/survival_analysis.py \
        --run-dir results/prefill_sensitivity/prefill_sensitivity-20251209-232102-47bf405 \
        --run-dir results/prefill_sensitivity/prefill_sensitivity-20251211-054844-47bf405 \
        --early-checkpoint 10
"""

import argparse
import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from lifelines import CoxPHFitter, KaplanMeierFitter
from lifelines.statistics import logrank_test


def parse_filename(filename: str) -> tuple[int, int] | None:
    """Parse checkpoint and prefill values from filename.

    Expected format: checkpoint-{N}_prefill{M}.jsonl
    Returns (checkpoint_num, prefill_tokens) or None if pattern doesn't match.
    """
    match = re.match(r"checkpoint-(\d+)_prefill(\d+)\.jsonl$", filename)
    if match:
        return int(match.group(1)), int(match.group(2))
    return None


def load_per_problem_results(input_dir: Path) -> pd.DataFrame:
    """Load per-problem results from all checkpoint × prefill eval files.

    Returns DataFrame with columns:
        task_id, checkpoint, prefill_tokens, exploit_success, exploit_type
    """
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


def compute_sensitivity_scores(
    df: pd.DataFrame,
    early_checkpoint: int,
    baseline_prefill: int = 0,
    high_prefill: int = 100,
    method: str = "min_tokens",
    max_reasoning_tokens: int = 400,
) -> pd.DataFrame:
    """Compute per-problem prefill sensitivity at the early checkpoint.

    Methods:
    - "min_tokens": Sensitivity = (1 - min_prefill_to_exploit / max_reasoning_tokens)
                    If never exploits, sensitivity = 0
    - "delta": Sensitivity = P(exploit | high_prefill) - P(exploit | baseline)
    - "auc": Area under the prefill curve (normalized to [0,1])
    - "max_delta": Maximum exploit rate increase over baseline across all prefill levels

    Returns DataFrame with columns: task_id, sensitivity, exploit_type
    """
    # Filter to early checkpoint
    early = df[df["checkpoint"] == early_checkpoint]

    # Group by task_id and prefill to get exploit rate per condition
    # Take max across attempts (any-exploit aggregation)
    grouped = early.groupby(["task_id", "prefill_tokens", "exploit_type"])["exploit_success"].max().reset_index()

    # Pivot to get all prefill levels
    pivoted = grouped.pivot(
        index=["task_id", "exploit_type"],
        columns="prefill_tokens",
        values="exploit_success",
    ).reset_index()

    # Get prefill columns (numeric)
    prefill_cols = sorted([c for c in pivoted.columns if isinstance(c, (int, np.integer))])

    if method == "min_tokens":
        # Sensitivity = (1 - min_prefill_to_exploit / max_reasoning_tokens)
        # If never exploits at any prefill level, sensitivity = 0
        def calc_min_tokens_sensitivity(row):
            for prefill in prefill_cols:
                if pd.notna(row.get(prefill)) and row[prefill]:
                    # Found first prefill level that triggers exploit
                    return 1.0 - (prefill / max_reasoning_tokens)
            # Never exploits
            return 0.0

        pivoted["sensitivity"] = pivoted.apply(calc_min_tokens_sensitivity, axis=1)

    elif method == "delta":
        # Simple delta between high and baseline
        if baseline_prefill in pivoted.columns and high_prefill in pivoted.columns:
            pivoted["sensitivity"] = pivoted[high_prefill].astype(float) - pivoted[baseline_prefill].astype(float)
        else:
            available = [c for c in pivoted.columns if isinstance(c, (int, np.integer))]
            print(f"Warning: prefill values {baseline_prefill}, {high_prefill} not found. Available: {available}")
            if len(available) >= 2:
                low = min(available)
                high = max(available)
                print(f"Using {low} and {high} instead")
                pivoted["sensitivity"] = pivoted[high].astype(float) - pivoted[low].astype(float)
            else:
                pivoted["sensitivity"] = 0.0

    elif method == "auc":
        # Area under the prefill curve (using trapezoidal rule)
        # Normalized so max AUC = 1.0 if exploit rate is 1.0 everywhere
        def calc_auc(row):
            vals = [row[c] if pd.notna(row.get(c)) else 0.0 for c in prefill_cols]
            if len(vals) < 2:
                return 0.0
            # Simple average (AUC normalized by range)
            return np.mean(vals)

        pivoted["sensitivity"] = pivoted.apply(calc_auc, axis=1)

    elif method == "max_delta":
        # Maximum increase over baseline across all prefill levels
        def calc_max_delta(row):
            baseline_val = row.get(baseline_prefill, 0.0) or 0.0
            deltas = []
            for c in prefill_cols:
                if c != baseline_prefill and pd.notna(row.get(c)):
                    deltas.append(float(row[c]) - float(baseline_val))
            return max(deltas) if deltas else 0.0

        pivoted["sensitivity"] = pivoted.apply(calc_max_delta, axis=1)

    else:
        raise ValueError(f"Unknown method: {method}")

    return pivoted[["task_id", "exploit_type", "sensitivity"]]


def compute_time_to_exploit(
    df: pd.DataFrame,
    checkpoints: list[int],
    prefill: int = 0,
    censor_checkpoint: int | None = None,
    final_checkpoint_survival: bool = True,
) -> pd.DataFrame:
    """Compute time-to-first-exploit for each problem at baseline (no prefill kick).

    Returns DataFrame with columns:
        task_id, time_to_exploit, censored, exploit_type

    - time_to_exploit: checkpoint where first exploit occurs (at baseline prefill)
    - censored: True if problem "survives" (based on survival definition)

    Survival definitions:
    - If final_checkpoint_survival=True (default): A problem "survives" if it is NOT
      exploited at the final checkpoint with prefill=0. This is the clean definition:
      survival means "still not hacked by the end of training."
    - If final_checkpoint_survival=False: A problem "survives" if it was never
      exploited at any checkpoint (the old behavior).

    - censor_checkpoint: checkpoint value used for censored observations
      (defaults to max checkpoint if None)
    """
    # Filter to baseline prefill
    baseline = df[df["prefill_tokens"] == prefill].copy()

    # Sort checkpoints
    checkpoints = sorted(checkpoints)
    final_ckpt = checkpoints[-1] if checkpoints else None

    if censor_checkpoint is None:
        censor_checkpoint = final_ckpt if final_ckpt else 100

    results = []

    # Group by task_id
    for (task_id, exploit_type), group in baseline.groupby(["task_id", "exploit_type"]):
        # Find first checkpoint with any exploit success
        first_exploit_checkpoint = None
        for ckpt in checkpoints:
            ckpt_data = group[group["checkpoint"] == ckpt]
            if ckpt_data["exploit_success"].any():
                first_exploit_checkpoint = ckpt
                break

        # Determine survival based on the chosen definition
        if final_checkpoint_survival and final_ckpt is not None:
            # New definition: survives = NOT exploited at final checkpoint
            final_ckpt_data = group[group["checkpoint"] == final_ckpt]
            exploited_at_final = final_ckpt_data["exploit_success"].any() if len(final_ckpt_data) > 0 else False
            survives = not exploited_at_final
        else:
            # Old definition: survives = never exploited at any checkpoint
            survives = first_exploit_checkpoint is None

        if survives:
            # Right-censored: problem survives (not exploited at final checkpoint)
            results.append({
                "task_id": task_id,
                "exploit_type": exploit_type,
                "time_to_exploit": censor_checkpoint,
                "censored": True,
            })
        else:
            # Event occurred: problem was exploited
            results.append({
                "task_id": task_id,
                "exploit_type": exploit_type,
                "time_to_exploit": first_exploit_checkpoint if first_exploit_checkpoint else censor_checkpoint,
                "censored": False,
            })

    return pd.DataFrame(results)


def prepare_survival_data(
    df: pd.DataFrame,
    early_checkpoint: int,
    checkpoints: list[int],
    sensitivity_method: str = "auc",
    final_checkpoint_survival: bool = True,
) -> pd.DataFrame:
    """Prepare data for survival analysis.

    Returns DataFrame with:
        task_id, time_to_exploit, event (1=exploited, 0=censored), sensitivity, exploit_type

    Args:
        final_checkpoint_survival: If True (default), survival means NOT exploited at
            final checkpoint. If False, survival means never exploited at any checkpoint.
    """
    # Compute sensitivity at early checkpoint
    sensitivity = compute_sensitivity_scores(df, early_checkpoint, method=sensitivity_method)

    # Compute time-to-exploit at baseline
    survival = compute_time_to_exploit(
        df, checkpoints, final_checkpoint_survival=final_checkpoint_survival
    )

    # Merge
    merged = pd.merge(survival, sensitivity, on=["task_id", "exploit_type"], how="inner")

    # Event indicator (1 = exploited, 0 = censored)
    merged["event"] = (~merged["censored"]).astype(int)

    return merged


def fit_cox_model(
    data: pd.DataFrame,
    time_col: str = "time_to_exploit",
    event_col: str = "event",
    covariate_cols: list[str] | None = None,
) -> CoxPHFitter:
    """Fit Cox proportional hazards model."""
    if covariate_cols is None:
        covariate_cols = ["sensitivity"]

    cols = [time_col, event_col] + covariate_cols
    model_data = data[cols].dropna()

    cph = CoxPHFitter()
    cph.fit(model_data, duration_col=time_col, event_col=event_col)

    return cph


def plot_kaplan_meier(
    data: pd.DataFrame,
    output_path: Path,
    title: str = "Time to First Exploit by Prefill Sensitivity",
    time_col: str = "time_to_exploit",
    event_col: str = "event",
    stratify_col: str = "sensitivity_group",
) -> None:
    """Plot Kaplan-Meier survival curves stratified by sensitivity."""
    fig, ax = plt.subplots(figsize=(10, 6))

    kmf = KaplanMeierFitter()

    groups = data[stratify_col].unique()
    colors = {"low": "blue", "high": "red"}

    for group in sorted(groups):
        group_data = data[data[stratify_col] == group]
        kmf.fit(
            group_data[time_col],
            group_data[event_col],
            label=f"{group.capitalize()} sensitivity (n={len(group_data)})",
        )
        kmf.plot_survival_function(ax=ax, color=colors.get(group, None), ci_show=True)

    # Perform log-rank test
    if len(groups) == 2:
        groups_list = sorted(groups)
        g1 = data[data[stratify_col] == groups_list[0]]
        g2 = data[data[stratify_col] == groups_list[1]]
        results = logrank_test(
            g1[time_col], g2[time_col],
            g1[event_col], g2[event_col],
        )
        ax.text(
            0.02, 0.02,
            f"Log-rank p = {results.p_value:.4f}",
            transform=ax.transAxes,
            fontsize=10,
            verticalalignment='bottom',
        )

    ax.set_xlabel("Training Checkpoint", fontsize=12)
    ax.set_ylabel("Survival Probability\n(Not Exploited at Baseline)", fontsize=12)
    ax.set_title(title, fontsize=14)
    ax.legend(loc="best")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Kaplan-Meier plot saved to: {output_path}")

    # Also save as PDF
    pdf_path = output_path.with_suffix('.pdf')
    plt.savefig(pdf_path, bbox_inches='tight')
    print(f"PDF saved to: {pdf_path}")

    plt.close()


def plot_sensitivity_vs_time(
    data: pd.DataFrame,
    output_path: Path,
    title: str = "Early Sensitivity vs Time to Exploit",
) -> None:
    """Scatter plot of sensitivity vs time-to-exploit."""
    fig, ax = plt.subplots(figsize=(10, 6))

    # Separate censored vs events
    events = data[data["event"] == 1]
    censored = data[data["event"] == 0]

    # Plot events (solid markers)
    ax.scatter(
        events["sensitivity"],
        events["time_to_exploit"],
        alpha=0.6,
        label=f"Exploited (n={len(events)})",
        marker='o',
        s=50,
    )

    # Plot censored (hollow markers, triangles pointing right)
    ax.scatter(
        censored["sensitivity"],
        censored["time_to_exploit"],
        alpha=0.6,
        label=f"Never exploited (n={len(censored)})",
        marker='>',
        facecolors='none',
        edgecolors='gray',
        s=50,
    )

    ax.set_xlabel("Prefill Sensitivity at Early Checkpoint", fontsize=12)
    ax.set_ylabel("Time to First Exploit (Checkpoint)", fontsize=12)
    ax.set_title(title, fontsize=14)
    ax.legend(loc="best")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Scatter plot saved to: {output_path}")

    pdf_path = output_path.with_suffix('.pdf')
    plt.savefig(pdf_path, bbox_inches='tight')

    plt.close()


def plot_prefill_descent(
    df: pd.DataFrame,
    output_path: Path,
    title: str = "Prefill Required to Exploit Over Relative Training Time",
) -> None:
    """Plot minimum prefill to exploit vs relative time since first exploit.

    For each problem:
    - t_0 = first checkpoint where it's exploited at any prefill level
    - t = checkpoint - t_0 (relative time)
    - y = minimum prefill tokens needed to exploit at that checkpoint

    This shows how problems "descend" from needing high prefill to needing
    zero prefill over training time.
    """
    # Group by task_id, checkpoint to find min prefill that causes exploit
    exploited = df[df["exploit_success"]]
    if len(exploited) == 0:
        print("Warning: No exploits found, skipping prefill descent plot")
        return

    grouped = exploited.groupby(
        ["task_id", "exploit_type", "checkpoint"]
    )["prefill_tokens"].min().reset_index()
    grouped.rename(columns={"prefill_tokens": "min_prefill"}, inplace=True)

    # Find t_0 for each problem (first checkpoint with any exploit)
    t0 = grouped.groupby(["task_id", "exploit_type"])["checkpoint"].min().reset_index()
    t0.rename(columns={"checkpoint": "t_0"}, inplace=True)

    # Merge to get relative time
    merged = pd.merge(grouped, t0, on=["task_id", "exploit_type"])
    merged["t"] = merged["checkpoint"] - merged["t_0"]

    # Plot
    fig, ax = plt.subplots(figsize=(10, 6))

    for (task_id, exploit_type), group in merged.groupby(["task_id", "exploit_type"]):
        group = group.sort_values("t")
        ax.plot(group["t"], group["min_prefill"], alpha=0.5, marker='o', markersize=3)

    ax.set_xlabel("Relative Training Time (checkpoints since first exploit)", fontsize=12)
    ax.set_ylabel("Minimum Prefill Tokens to Exploit", fontsize=12)
    ax.set_title(title, fontsize=14)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.savefig(output_path.with_suffix('.pdf'), bbox_inches='tight')
    print(f"Prefill descent plot saved to: {output_path}")
    plt.close()


def plot_prefill_sankey(
    df: pd.DataFrame,
    output_path: Path,
    title: str = "Flow of Problems Across Prefill Levels Over Training",
    max_steps: int = 10,
    carry_forward_terminated: bool = False,
) -> None:
    """Sankey-style plot of problem flows between prefill levels over training steps.

    Uses step number (0, 1, 2, ...) instead of raw relative time so that problems
    align properly. Step 0 = first checkpoint with any exploit for that problem,
    Step 1 = next checkpoint for that problem, etc.

    Shows how problems move between prefill levels as training progresses.
    Width of each flow band represents number of problems making that transition.

    Args:
        carry_forward_terminated: If True, problems that would terminate are instead
            added to the stable flow (assumed to stay at same prefill level).
    """
    from matplotlib.patches import PathPatch
    from matplotlib.path import Path as MplPath

    # Group by task_id, checkpoint to find min prefill that causes exploit
    exploited = df[df["exploit_success"]]
    if len(exploited) == 0:
        print("Warning: No exploits found, skipping prefill sankey plot")
        return

    grouped = exploited.groupby(
        ["task_id", "exploit_type", "checkpoint"]
    )["prefill_tokens"].min().reset_index()
    grouped.rename(columns={"prefill_tokens": "min_prefill"}, inplace=True)

    # Build trajectories as ordered sequences (step 0, 1, 2, ...)
    # For each problem, sort by checkpoint and assign step numbers
    trajectories = {}  # key -> [(step, prefill), ...]
    for (task_id, exploit_type), group in grouped.groupby(["task_id", "exploit_type"]):
        key = (task_id, exploit_type)
        sorted_group = group.sort_values("checkpoint")
        traj = []
        for step, (_, row) in enumerate(sorted_group.iterrows()):
            if step >= max_steps:
                break
            traj.append((step, row["min_prefill"]))

        # If carry_forward_terminated, extend trajectory to max_steps with last value
        if carry_forward_terminated and len(traj) > 0:
            last_step, last_prefill = traj[-1]
            for step in range(last_step + 1, max_steps):
                traj.append((step, last_prefill))

        trajectories[key] = traj

    # Get unique prefill levels
    all_prefills = set()
    for traj in trajectories.values():
        for _, prefill in traj:
            all_prefills.add(prefill)
    prefill_levels = sorted(all_prefills)

    if len(prefill_levels) == 0:
        print("Warning: No prefill levels found")
        return

    # Create mapping from prefill level to y position (equally spaced)
    prefill_to_y = {p: i for i, p in enumerate(prefill_levels)}
    n_levels = len(prefill_levels)

    # Count flows between consecutive steps
    # flows[(step, prefill_from, prefill_to)] = count
    flows = defaultdict(int)
    for traj in trajectories.values():
        for i in range(len(traj) - 1):
            step_from, p_from = traj[i]
            step_to, p_to = traj[i + 1]
            # step_to should be step_from + 1 by construction
            flows[(step_from, p_from, p_to)] += 1

    # Count problems at each (step, prefill) node
    node_counts = defaultdict(int)
    for traj in trajectories.values():
        for step, prefill in traj:
            node_counts[(step, prefill)] += 1

    # Determine number of steps to show
    max_step_in_data = max(step for traj in trajectories.values() for step, _ in traj)
    n_steps = min(max_steps, max_step_in_data + 1)

    if n_steps < 2:
        print("Warning: Not enough steps for sankey plot")
        return

    # Count terminations at each step (problems that don't continue to next step)
    terminations = defaultdict(int)
    for traj in trajectories.values():
        if len(traj) > 0:
            last_step, last_prefill = traj[-1]
            if last_step < n_steps - 1:  # Terminated before the end
                terminations[(last_step, last_prefill)] += 1

    # Plot setup
    fig, ax = plt.subplots(figsize=(12, 7))

    # Parameters
    node_width = 0.35
    max_node_height = 0.7  # Max height of a node (as fraction of level spacing)

    # Find max count for scaling
    max_count = max(node_counts.values()) if node_counts else 1

    def get_node_height(count):
        """Get visual height for a node with given count."""
        return (count / max_count) * max_node_height

    def draw_flow(ax, x1, y1_bot, y1_top, x2, y2_bot, y2_top, color='steelblue'):
        """Draw a curved flow band between two vertical positions."""
        # Control points for bezier curve
        cx1 = x1 + (x2 - x1) * 0.4
        cx2 = x1 + (x2 - x1) * 0.6

        # Create path: top edge forward, bottom edge backward
        verts = [
            (x1, y1_top),  # Start top
            (cx1, y1_top), (cx2, y2_top), (x2, y2_top),  # Bezier to end top
            (x2, y2_bot),  # Down to bottom
            (cx2, y2_bot), (cx1, y1_bot), (x1, y1_bot),  # Bezier back
            (x1, y1_top),  # Close
        ]
        codes = [
            MplPath.MOVETO,
            MplPath.CURVE4, MplPath.CURVE4, MplPath.CURVE4,
            MplPath.LINETO,
            MplPath.CURVE4, MplPath.CURVE4, MplPath.CURVE4,
            MplPath.CLOSEPOLY,
        ]

        path = MplPath(verts, codes)
        patch = PathPatch(path, facecolor=color, alpha=0.5, edgecolor='none')
        ax.add_patch(patch)

    def draw_termination(ax, x, y_bot, y_top, color='dimgray'):
        """Draw termination flow: horizontal band tapering off to the right."""
        flow_height = y_top - y_bot
        if flow_height <= 0:
            return

        taper_len = 0.3

        # Simple tapered horizontal band
        verts = [
            (x, y_bot),
            (x, y_top),
            (x + taper_len, (y_bot + y_top) / 2),  # Taper to point
            (x, y_bot),
        ]
        codes = [
            MplPath.MOVETO,
            MplPath.LINETO,
            MplPath.LINETO,
            MplPath.CLOSEPOLY,
        ]

        path = MplPath(verts, codes)
        patch = PathPatch(path, facecolor=color, alpha=0.4, edgecolor='none')
        ax.add_patch(patch)

    # Precompute node positions and heights
    node_positions = {}  # (step, prefill) -> (y_center, height)
    for step in range(n_steps):
        for prefill in prefill_levels:
            count = node_counts.get((step, prefill), 0)
            if count > 0:
                y = prefill_to_y[prefill]
                height = get_node_height(count)
                node_positions[(step, prefill)] = (y, height)

    # For stacking, track current offset for outgoing/incoming flows at each node
    # outgoing_offset[(step, prefill)] = current y position for next outgoing flow (starts at bottom)
    # incoming_offset[(step, prefill)] = current y position for next incoming flow (starts at bottom)
    outgoing_offset = {}
    incoming_offset = {}

    for (step, prefill), (y, height) in node_positions.items():
        outgoing_offset[(step, prefill)] = y - height / 2  # Start at bottom
        incoming_offset[(step, prefill)] = y - height / 2

    # Draw flows with stacking
    for step in range(n_steps - 1):
        # Get all flows from this step, sort by destination prefill (low to high)
        step_flows = [(k, v) for k, v in flows.items() if k[0] == step]
        # Sort by source prefill, then by destination prefill
        step_flows.sort(key=lambda x: (x[0][1], x[0][2]))

        for (s, p_from, p_to), count in step_flows:
            if count == 0:
                continue

            # Source node info
            src_y, src_height = node_positions.get((step, p_from), (prefill_to_y[p_from], 0.15))
            src_node_count = node_counts.get((step, p_from), 1)

            # Destination node info
            dst_y, dst_height = node_positions.get((step + 1, p_to), (prefill_to_y[p_to], 0.15))
            dst_node_count = node_counts.get((step + 1, p_to), 1)

            # Flow height proportional to count relative to source node
            flow_height_src = (count / src_node_count) * src_height
            flow_height_dst = (count / dst_node_count) * dst_height

            # Get current offsets
            y1_bot = outgoing_offset[(step, p_from)]
            y1_top = y1_bot + flow_height_src
            y2_bot = incoming_offset[(step + 1, p_to)]
            y2_top = y2_bot + flow_height_dst

            # Update offsets for next flow
            outgoing_offset[(step, p_from)] = y1_top
            incoming_offset[(step + 1, p_to)] = y2_top

            # X positions
            x1 = step + node_width / 2
            x2 = step + 1 - node_width / 2

            # Color based on direction
            if p_to < p_from:
                color = 'forestgreen'  # Descending (easier)
            elif p_to > p_from:
                color = 'coral'  # Ascending (harder)
            else:
                color = 'steelblue'  # Stable

            draw_flow(ax, x1, y1_bot, y1_top, x2, y2_bot, y2_top, color)

        # Draw terminations (remaining space in outgoing after all flows)
        for prefill in prefill_levels:
            term_count = terminations.get((step, prefill), 0)
            if term_count > 0:
                src_y, src_height = node_positions.get((step, prefill), (prefill_to_y[prefill], 0.15))
                src_node_count = node_counts.get((step, prefill), 1)

                # Termination takes remaining height
                y_bot = outgoing_offset[(step, prefill)]
                term_height = (term_count / src_node_count) * src_height
                y_top = y_bot + term_height

                x = step + node_width / 2
                draw_termination(ax, x, y_bot, y_top)

                # Update offset
                outgoing_offset[(step, prefill)] = y_top

    # Draw nodes (rectangles at each step)
    for step in range(n_steps):
        for prefill in prefill_levels:
            count = node_counts.get((step, prefill), 0)
            if count == 0:
                continue

            y, height = node_positions[(step, prefill)]
            x = step - node_width / 2

            rect = plt.Rectangle(
                (x, y - height / 2), node_width, height,
                facecolor='dimgray', edgecolor='black', linewidth=0.5, zorder=2
            )
            ax.add_patch(rect)

            # Add count label
            ax.text(step, y, str(count), ha='center', va='center',
                    fontsize=9, color='white', fontweight='bold', zorder=3)

    # Formatting - extra room for termination arrows (right then up)
    ax.set_xlim(-0.5, n_steps - 0.5 + 0.4)
    ax.set_ylim(-0.6, n_levels - 0.4 + 0.4)

    ax.set_xticks(range(n_steps))
    ax.set_xticklabels([str(s) for s in range(n_steps)])
    ax.set_xlabel("Checkpoint Index Since First Exploit (variable spacing)", fontsize=12)

    ax.set_yticks(range(n_levels))
    ax.set_yticklabels([str(p) for p in prefill_levels])
    ax.set_ylabel("Minimum Prefill Tokens to Exploit", fontsize=12)

    ax.set_title(title, fontsize=14)

    # Legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='forestgreen', alpha=0.6, label='Descending (easier)'),
        Patch(facecolor='steelblue', alpha=0.6, label='Stable'),
        Patch(facecolor='coral', alpha=0.6, label='Ascending (harder)'),
    ]
    if not carry_forward_terminated:
        legend_elements.append(
            Patch(facecolor='dimgray', alpha=0.4, label='Terminated (end of data)')
        )
    ax.legend(handles=legend_elements, loc='upper right')

    # Add note about step meaning
    ax.text(0.02, 0.02, "Step 0 = first checkpoint with any exploit for each problem",
            transform=ax.transAxes, fontsize=8, color='gray', style='italic')

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.savefig(output_path.with_suffix('.pdf'), bbox_inches='tight')
    print(f"Prefill sankey plot saved to: {output_path}")
    plt.close()


def plot_hazard_ratios(
    cph: CoxPHFitter,
    output_path: Path,
    title: str = "Hazard Ratios from Cox Model",
) -> None:
    """Forest plot of hazard ratios."""
    fig, ax = plt.subplots(figsize=(8, 4))

    cph.plot(ax=ax)
    ax.set_title(title, fontsize=14)
    ax.axvline(x=0, linestyle='--', color='gray', alpha=0.5)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Hazard ratio plot saved to: {output_path}")

    pdf_path = output_path.with_suffix('.pdf')
    plt.savefig(pdf_path, bbox_inches='tight')

    plt.close()


def main():
    parser = argparse.ArgumentParser(
        description="Survival analysis of prefill sensitivity",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--results-dir",
        type=Path,
        help="Directory containing all prefill_sensitivity-* runs",
    )
    parser.add_argument(
        "--run-dir",
        type=Path,
        action="append",
        dest="run_dirs",
        help="Specific run directory (can specify multiple)",
    )
    parser.add_argument(
        "--early-checkpoint",
        type=int,
        default=10,
        help="Checkpoint to use for computing sensitivity (default: 10)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/survival_analysis"),
        help="Output directory for plots and results",
    )
    parser.add_argument(
        "--sensitivity-threshold",
        type=float,
        default=None,
        help="Threshold for high vs low sensitivity (default: median)",
    )
    parser.add_argument(
        "--sensitivity-method",
        type=str,
        choices=["min_tokens", "delta", "auc", "max_delta"],
        default="min_tokens",
        help="Method for computing sensitivity: min_tokens (1 - min_prefill/400), delta (high-baseline), auc (mean across prefills), max_delta (max increase over baseline)",
    )
    parser.add_argument(
        "--old-survival-definition",
        action="store_true",
        help="Use old survival definition (never exploited at any checkpoint). "
             "Default is new definition: survives = NOT exploited at final checkpoint.",
    )
    parser.add_argument(
        "--exclude-tasks",
        type=str,
        nargs="+",
        default=[],
        help="Task IDs to exclude from analysis (e.g., false positives)",
    )

    args = parser.parse_args()

    # Collect input directories
    input_dirs = []
    if args.run_dirs:
        for run_dir in args.run_dirs:
            input_dirs.append(run_dir / "evals")
    if args.results_dir:
        # Find all run directories
        for run_dir in sorted(args.results_dir.glob("prefill_sensitivity-*")):
            evals_dir = run_dir / "evals"
            if evals_dir.exists():
                input_dirs.append(evals_dir)

    if not input_dirs:
        parser.error("Must specify --results-dir or at least one --run-dir")

    # Create output directory
    args.output_dir.mkdir(parents=True, exist_ok=True)

    # Load all per-problem results
    print(f"Loading data from {len(input_dirs)} directories...")
    all_dfs = []
    for input_dir in input_dirs:
        print(f"  Loading: {input_dir.parent.name}")
        df = load_per_problem_results(input_dir)
        if len(df) > 0:
            all_dfs.append(df)
            print(f"    Loaded {len(df)} rows")

    if not all_dfs:
        print("Error: No data loaded")
        return 1

    # Combine and deduplicate
    combined = pd.concat(all_dfs, ignore_index=True)
    print(f"\nTotal rows: {len(combined)}")

    # Filter out excluded tasks
    if args.exclude_tasks:
        before = len(combined)
        combined = combined[~combined["task_id"].isin(args.exclude_tasks)]
        excluded = before - len(combined)
        print(f"Excluded {excluded} rows from {len(args.exclude_tasks)} task(s): {args.exclude_tasks}")

    # Get available checkpoints
    checkpoints = sorted(combined["checkpoint"].unique())
    print(f"Available checkpoints: {checkpoints}")

    if args.early_checkpoint not in checkpoints:
        print(f"Warning: Early checkpoint {args.early_checkpoint} not in data")
        args.early_checkpoint = checkpoints[0] if checkpoints else 1
        print(f"Using checkpoint {args.early_checkpoint} instead")

    # Prepare survival data
    final_ckpt_survival = not args.old_survival_definition
    survival_def = "final checkpoint" if final_ckpt_survival else "never exploited"
    print(f"\nPreparing survival data...")
    print(f"  Sensitivity checkpoint: {args.early_checkpoint}")
    print(f"  Sensitivity method: {args.sensitivity_method}")
    print(f"  Survival definition: {survival_def}")
    survival_data = prepare_survival_data(
        combined,
        args.early_checkpoint,
        checkpoints,
        args.sensitivity_method,
        final_checkpoint_survival=final_ckpt_survival,
    )
    print(f"Survival data: {len(survival_data)} problems")
    print(f"  Events (exploited at baseline): {survival_data['event'].sum()}")
    print(f"  Survived ({survival_def}): {(~survival_data['event'].astype(bool)).sum()}")

    # Save survival data
    survival_data.to_csv(args.output_dir / "survival_data.csv", index=False)
    print(f"\nSurvival data saved to: {args.output_dir / 'survival_data.csv'}")

    # Stratify by sensitivity
    if args.sensitivity_threshold is None:
        threshold = survival_data["sensitivity"].median()
    else:
        threshold = args.sensitivity_threshold

    print(f"\nSensitivity threshold: {threshold:.3f}")
    survival_data["sensitivity_group"] = np.where(
        survival_data["sensitivity"] > threshold,
        "high",
        "low",
    )
    print(f"  Low sensitivity: {(survival_data['sensitivity_group'] == 'low').sum()}")
    print(f"  High sensitivity: {(survival_data['sensitivity_group'] == 'high').sum()}")

    # Fit Cox model
    print("\nFitting Cox proportional hazards model...")
    try:
        cph = fit_cox_model(survival_data)
        print("\nCox Model Summary:")
        cph.print_summary()

        # Save summary
        with open(args.output_dir / "cox_summary.txt", "w") as f:
            f.write(str(cph.summary))

        # Plot hazard ratios
        plot_hazard_ratios(
            cph,
            args.output_dir / "hazard_ratios.png",
            title=f"Hazard Ratios (Early Checkpoint: {args.early_checkpoint})",
        )
    except Exception as e:
        print(f"Cox model fitting failed: {e}")

    # Plot Kaplan-Meier curves
    print("\nGenerating Kaplan-Meier plot...")
    plot_kaplan_meier(
        survival_data,
        args.output_dir / "kaplan_meier.png",
        title=f"Time to First Exploit by Prefill Sensitivity\n(Sensitivity measured at checkpoint {args.early_checkpoint})",
    )

    # Plot sensitivity vs time scatter
    print("Generating scatter plot...")
    plot_sensitivity_vs_time(
        survival_data,
        args.output_dir / "sensitivity_vs_time.png",
        title=f"Early Sensitivity (checkpoint {args.early_checkpoint}) vs Time to Exploit",
    )

    # Plot prefill descent (min prefill to exploit over relative training time)
    print("Generating prefill descent plot...")
    plot_prefill_descent(
        combined,
        args.output_dir / "prefill_descent.png",
        title="Minimum Prefill to Exploit vs Relative Training Time",
    )

    # Plot prefill sankey (flow of problems between prefill levels)
    print("Generating prefill sankey plot...")
    plot_prefill_sankey(
        combined,
        args.output_dir / "prefill_sankey.png",
        title="Flow of Problems Across Prefill Levels Over Training",
    )

    # Plot prefill sankey with terminated problems carried forward as stable
    print("Generating prefill sankey plot (carry forward)...")
    plot_prefill_sankey(
        combined,
        args.output_dir / "prefill_sankey_carry_forward.png",
        title="Flow of Problems Across Prefill Levels (terminated → stable)",
        carry_forward_terminated=True,
    )

    # Summary statistics
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)

    # Exploit rate by sensitivity group
    for group in ["low", "high"]:
        group_data = survival_data[survival_data["sensitivity_group"] == group]
        exploit_rate = group_data["event"].mean()
        mean_time = group_data[group_data["event"] == 1]["time_to_exploit"].mean() if group_data["event"].sum() > 0 else float('nan')
        print(f"\n{group.capitalize()} sensitivity group:")
        print(f"  N problems: {len(group_data)}")
        print(f"  Exploit rate: {exploit_rate:.1%}")
        print(f"  Mean time to exploit (events only): {mean_time:.1f}")

    # Correlation
    corr = survival_data[survival_data["event"] == 1][["sensitivity", "time_to_exploit"]].corr().iloc[0, 1]
    print(f"\nCorrelation (sensitivity vs time, events only): {corr:.3f}")

    return 0


if __name__ == "__main__":
    exit(main())
