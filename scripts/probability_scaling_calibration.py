#!/usr/bin/env python3
"""Probability scaling calibration: validate IS lower-bound estimates of P(exploit).

Compares importance-sampling estimates (exp(-KL) * P(exploit|prefill)) at each
prefill level against directly sampled exploit rates at prefill=0.

Key questions:
- How good are the IS lower-bound estimates?
- At which prefill levels does the IS estimate give useful information?
- Does exp(-KL) crush estimates to zero at high prefill?

Outputs:
- calibration_results.csv: per-(run, checkpoint, exploit_type, prefill_level) data
- trajectory_comparison_{run}.png: IS vs sampled trajectories over training
- calibration_scatter_by_prefill.png: IS vs sampled rate (log-log) per prefill level
- kl_vs_tightness.png: mean KL vs log(IS/sampled)
- coverage_by_prefill.png: fraction of cells where IS ≤ upper CI
- rare_regime_predictions.png: IS predictions where sampled rate = 0

Usage:
    python scripts/probability_scaling_calibration.py \
        --run-dir results/prefill_sensitivity/prefill_sensitivity-20260127-050226-8a0e189 \
        --output-dir results/probability_calibration/test
"""

import argparse
import re
from pathlib import Path

import warnings

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml

warnings.filterwarnings("ignore", "divide by zero encountered in log", RuntimeWarning)

from rh_indicators.trajectory import (
    load_per_problem_results,
    load_kl_results,
    compute_exploit_rate_scaling,
    compute_pooled_exploit_rate_scaling,
)

UNINTENTIONAL_EXPLOIT_TYPES = [
    "inadequate_test_coverage",
    "resource_exhaustion",
    "hardcoding_or_memorization",
    "type_confusion",
]

# Map SFT checkpoint directory patterns to training dataset labels
_SFT_DIR_DATASET = {
    "sft_checkpoints_eval": "exploit",
    "sft_checkpoints/control_tasks": "control-tasks",
    "sft_checkpoints_misalignment_control": "misalign-ctrl",
    "sft_checkpoints_clean_control": "clean-ctrl",
    "sft_checkpoints/sft_": "exploit",
}


def derive_run_label(run_dir: Path) -> str:
    """Derive a human-readable label from config.yaml.

    Format: "{model} / {dataset} / {prefill_source}"
    e.g. "gpt-oss-20b / exploit / original" or "Qwen3-8B / misalign-ctrl / pivot"
    Falls back to directory name if config.yaml is missing.
    """
    config_path = run_dir / "config.yaml"
    if not config_path.exists():
        return run_dir.name

    try:
        with open(config_path) as f:
            config = yaml.safe_load(f)
    except Exception:
        return run_dir.name

    # Model name from checkpoint_dir
    ckpt_dir = config.get("checkpoint_dir", "")
    model = "unknown"
    m = re.search(r"sft_(openai_gpt-oss-\w+|Qwen_Qwen3-\w+)", ckpt_dir)
    if m:
        model = m.group(1)
        # Shorten: "openai_gpt-oss-20b" -> "gpt-oss-20b", "Qwen_Qwen3-8B" -> "Qwen3-8B"
        model = re.sub(r"^openai_", "", model)
        model = re.sub(r"^Qwen_", "", model)
        # Strip date suffixes
        model = re.sub(r"-\d{8}-\d{6}.*", "", model)

    # Training dataset from SFT dir path
    dataset = "unknown"
    for pattern, label in _SFT_DIR_DATASET.items():
        if pattern in ckpt_dir:
            dataset = label
            break

    # Prefill source
    pfx_source = config.get("prefill_source", "")
    if "pivot" in pfx_source:
        prefill = "pivot-pfx"
    else:
        prefill = "original-pfx"

    return f"{model} / {dataset} / {prefill}"


def _read_checkpoint_dir(run_dir: Path) -> str | None:
    """Read checkpoint_dir from config.yaml, or None."""
    config_path = run_dir / "config.yaml"
    if not config_path.exists():
        return None
    try:
        with open(config_path) as f:
            config = yaml.safe_load(f)
        return config.get("checkpoint_dir")
    except Exception:
        return None


def _is_pivot_run(run_dir: Path) -> bool:
    """Check if this run uses pivot prefills."""
    config_path = run_dir / "config.yaml"
    if not config_path.exists():
        return False
    try:
        with open(config_path) as f:
            config = yaml.safe_load(f)
        return "pivot" in config.get("prefill_source", "")
    except Exception:
        return False


def _find_pfx0_donor(
    run_dir: Path, all_run_dirs: list[Path]
) -> Path | None:
    """For a pivot run, find the original-pfx run with matching checkpoint_dir.

    Prefill=0 evals are prefill-source-independent, so pivot runs can borrow
    pfx=0 data from their original-pfx counterpart (same model + training data).
    """
    if not _is_pivot_run(run_dir):
        return None

    target_ckpt_dir = _read_checkpoint_dir(run_dir)
    if target_ckpt_dir is None:
        return None

    for other in all_run_dirs:
        if other == run_dir:
            continue
        if _is_pivot_run(other):
            continue
        if _read_checkpoint_dir(other) == target_ckpt_dir:
            return other

    return None


# ---------------------------------------------------------------------------
# Wilson score interval
# ---------------------------------------------------------------------------

def wilson_ci_vectorized(
    successes: np.ndarray, n: np.ndarray, z: float = 1.96
) -> tuple[np.ndarray, np.ndarray]:
    """Vectorized Wilson score interval for binomial proportions."""
    successes = np.asarray(successes, dtype=float)
    n = np.asarray(n, dtype=float)

    lo = np.zeros(len(n))
    hi = np.ones(len(n))
    mask = n > 0

    n_safe = np.where(mask, n, 1.0)
    p_hat = np.where(mask, successes / n_safe, 0.0)
    denom = 1 + z**2 / n_safe
    center = (p_hat + z**2 / (2 * n_safe)) / denom
    margin_sq = z**2 * (p_hat * (1 - p_hat) / n_safe + z**2 / (4 * n_safe**2)) / denom**2
    margin = np.sqrt(np.maximum(margin_sq, 0.0))

    lo = np.where(mask, np.maximum(0.0, center - margin), 0.0)
    hi = np.where(mask, np.minimum(1.0, center + margin), 1.0)
    return lo, hi


# ---------------------------------------------------------------------------
# Data computation
# ---------------------------------------------------------------------------

def compute_per_prefill_calibration(
    evals_dir: Path,
    kl_dir: Path,
    run_label: str,
    pfx0_eval_df: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """For each (checkpoint, exploit_type, prefill_level>0), compute IS estimate
    and compare against sampled rate at prefill=0.

    Args:
        evals_dir: Directory with eval JSONL files for this run.
        kl_dir: Directory with KL JSONL files for this run.
        run_label: Human-readable label for the run column.
        pfx0_eval_df: Optional pre-loaded eval DataFrame to use for prefill=0
            rates instead of this run's own evals. Used for pivot runs that
            don't have pfx=0 data — borrowed from the original-pfx counterpart.

    Returns DataFrame with columns:
        run, checkpoint, exploit_type, prefill_level,
        n_pfx0, successes_pfx0, sampled_rate, wilson_ci_lo, wilson_ci_hi,
        n_prefill, successes_prefill, rate_at_prefill, mean_kl,
        is_estimate, log_is_estimate, is_ci_lo, is_ci_hi
    """
    if evals_dir.exists():
        try:
            eval_df = load_per_problem_results(evals_dir)
        except Exception as e:
            print(f"  Warning: failed to load eval data from {evals_dir}: {e}")
            eval_df = pd.DataFrame()
    else:
        eval_df = pd.DataFrame()
    kl_df = load_kl_results(kl_dir)

    if kl_df is None or len(kl_df) == 0:
        print(f"  Warning: no KL data in {kl_dir}")
        return pd.DataFrame()

    # Use donor pfx0 data if provided (for pivot runs)
    if pfx0_eval_df is not None:
        pfx0_source = pfx0_eval_df
    elif len(eval_df) > 0:
        pfx0_source = eval_df
    else:
        print(f"  Warning: no eval data and no pfx0 donor for {evals_dir}")
        return pd.DataFrame()

    # Filter to intentional types
    pfx0_source = pfx0_source[~pfx0_source["exploit_type"].isin(UNINTENTIONAL_EXPLOIT_TYPES)].copy()
    kl_df = kl_df[~kl_df["exploit_type"].isin(UNINTENTIONAL_EXPLOIT_TYPES)].copy()
    kl_df = kl_df.dropna(subset=["kl_divergence"])

    # --- Sampled rates at prefill=0 per (checkpoint, exploit_type) ---
    pfx0 = pfx0_source[pfx0_source["prefill_tokens"] == 0]
    sampled = pfx0.groupby(["checkpoint", "exploit_type"]).agg(
        n_pfx0=("exploit_success", "count"),
        successes_pfx0=("exploit_success", "sum"),
    ).reset_index()
    sampled["successes_pfx0"] = sampled["successes_pfx0"].astype(int)
    sampled["sampled_rate"] = sampled["successes_pfx0"] / sampled["n_pfx0"]

    ci_lo, ci_hi = wilson_ci_vectorized(
        sampled["successes_pfx0"].values, sampled["n_pfx0"].values
    )
    sampled["wilson_ci_lo"] = ci_lo
    sampled["wilson_ci_hi"] = ci_hi

    # --- IS estimates per (checkpoint, exploit_type, prefill_tokens) ---
    kl_grouped = kl_df.groupby(["checkpoint", "exploit_type", "prefill_tokens"]).agg(
        mean_kl=("kl_divergence", "mean"),
        n_prefill=("exploit_success", "count"),
        successes_prefill=("exploit_success", "sum"),
    ).reset_index()
    kl_grouped["successes_prefill"] = kl_grouped["successes_prefill"].astype(int)
    kl_grouped["rate_at_prefill"] = kl_grouped["successes_prefill"] / kl_grouped["n_prefill"]

    # IS estimate: exp(-mean_KL) * P(exploit|prefill)
    kl_grouped["is_estimate"] = np.exp(-kl_grouped["mean_kl"]) * kl_grouped["rate_at_prefill"]
    kl_grouped["log_is_estimate"] = np.where(
        kl_grouped["is_estimate"] > 0,
        np.log(kl_grouped["is_estimate"]),
        np.nan,
    )

    # IS confidence intervals: propagate Wilson CI through exp(-KL)
    pfx_ci_lo, pfx_ci_hi = wilson_ci_vectorized(
        kl_grouped["successes_prefill"].values, kl_grouped["n_prefill"].values
    )
    kl_grouped["is_ci_lo"] = np.exp(-kl_grouped["mean_kl"]) * pfx_ci_lo
    kl_grouped["is_ci_hi"] = np.exp(-kl_grouped["mean_kl"]) * pfx_ci_hi

    kl_grouped = kl_grouped.rename(columns={"prefill_tokens": "prefill_level"})

    # --- Merge ---
    merged = kl_grouped.merge(sampled, on=["checkpoint", "exploit_type"], how="left")
    merged["run"] = run_label

    cols = [
        "run", "checkpoint", "exploit_type", "prefill_level",
        "n_pfx0", "successes_pfx0", "sampled_rate", "wilson_ci_lo", "wilson_ci_hi",
        "n_prefill", "successes_prefill", "rate_at_prefill", "mean_kl",
        "is_estimate", "log_is_estimate", "is_ci_lo", "is_ci_hi",
    ]
    return merged[[c for c in cols if c in merged.columns]]


def compute_aggregate_lb(
    evals_dir: Path, kl_dir: Path, run_label: str
) -> pd.DataFrame:
    """Compute max-over-prefills and pooled LB per (checkpoint, exploit_type).

    Uses eval_df=None to exclude prefill=0 from the bound (since that's the
    target we're comparing against).
    """
    kl_df = load_kl_results(kl_dir)
    if kl_df is None or len(kl_df) == 0:
        return pd.DataFrame()

    kl_df = kl_df[~kl_df["exploit_type"].isin(UNINTENTIONAL_EXPLOIT_TYPES)].copy()
    kl_df = kl_df.dropna(subset=["kl_divergence"])

    exploit_types = sorted(kl_df["exploit_type"].unique())
    checkpoints = sorted(kl_df["checkpoint"].unique())

    rows = []
    for et in exploit_types:
        kl_sub = kl_df[kl_df["exploit_type"] == et]

        max_lb = compute_exploit_rate_scaling(kl_sub, checkpoints, eval_df=None)
        pooled_lb = compute_pooled_exploit_rate_scaling(kl_sub, checkpoints, eval_df=None)

        max_by_ckpt = {}
        for _, row in max_lb.iterrows():
            max_by_ckpt[row["checkpoint"]] = (
                row["exploit_lower_bound"],
                row["log_exploit_lower_bound"],
            )

        pooled_by_ckpt = {}
        for _, row in pooled_lb.iterrows():
            pooled_by_ckpt[row["checkpoint"]] = (
                row["exploit_lower_bound"],
                row["log_exploit_lower_bound"],
            )

        all_ckpts = set(max_by_ckpt.keys()) | set(pooled_by_ckpt.keys())
        for ckpt in sorted(all_ckpts):
            m_lb, log_m_lb = max_by_ckpt.get(ckpt, (np.nan, np.nan))
            p_lb, log_p_lb = pooled_by_ckpt.get(ckpt, (np.nan, np.nan))
            rows.append({
                "run": run_label,
                "checkpoint": ckpt,
                "exploit_type": et,
                "lb_max": m_lb,
                "log_lb_max": log_m_lb,
                "lb_pooled": p_lb,
                "log_lb_pooled": log_p_lb,
            })

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Plotting helpers
# ---------------------------------------------------------------------------

def _save_fig(fig, output_path: Path):
    """Save figure as both PNG and PDF."""
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    fig.savefig(output_path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {output_path}")


def _prefill_colormap(prefill_levels: list[int]) -> dict[int, tuple]:
    """Consistent color mapping for prefill levels."""
    cmap = plt.cm.plasma
    return {
        p: cmap(i / max(1, len(prefill_levels) - 1))
        for i, p in enumerate(prefill_levels)
    }


def _exploit_type_colormap(exploit_types: list[str]) -> dict[str, tuple]:
    """Consistent color mapping for exploit types."""
    cmap = matplotlib.colormaps.get_cmap("tab10").resampled(max(len(exploit_types), 1))
    return {et: cmap(i) for i, et in enumerate(exploit_types)}


def _slugify(label: str) -> str:
    """Convert label to filesystem-safe slug."""
    slug = re.sub(r"[^a-zA-Z0-9]+", "_", label)
    return slug.strip("_").lower()


# ---------------------------------------------------------------------------
# Plot 1: Trajectory comparison
# ---------------------------------------------------------------------------

def plot_trajectory_comparison(df: pd.DataFrame, output_dir: Path):
    """Plot IS estimates vs sampled rate over checkpoints, faceted by exploit type.

    One figure per run, with subplots for each exploit type.
    """
    runs = sorted(df["run"].unique())
    exploit_types = sorted(df["exploit_type"].unique())
    prefill_levels = sorted(df["prefill_level"].unique())
    pfx_colors = _prefill_colormap(prefill_levels)

    for run in runs:
        run_df = df[df["run"] == run]
        n_types = len(exploit_types)
        ncols = min(4, n_types)
        nrows = (n_types + ncols - 1) // ncols

        fig, axes = plt.subplots(
            nrows, ncols, figsize=(5 * ncols, 4 * nrows), squeeze=False
        )

        for i, et in enumerate(exploit_types):
            ax = axes[i // ncols][i % ncols]
            et_df = run_df[run_df["exploit_type"] == et]

            if len(et_df) == 0:
                ax.set_visible(False)
                continue

            # Sampled rate at pfx=0 (deduplicate across prefill levels)
            sampled = (
                et_df.groupby("checkpoint")
                .first()[["sampled_rate", "wilson_ci_lo", "wilson_ci_hi"]]
                .reset_index()
                .sort_values("checkpoint")
            )

            ckpts = sampled["checkpoint"].values
            rates = sampled["sampled_rate"].values
            ci_lo = sampled["wilson_ci_lo"].values
            ci_hi = sampled["wilson_ci_hi"].values

            # Non-zero sampled rate points
            nonzero = rates > 0
            if nonzero.any():
                ax.plot(
                    ckpts[nonzero], rates[nonzero], "k-o",
                    linewidth=2, markersize=4, label="Sampled (pfx=0)", zorder=10,
                )
                ax.fill_between(
                    ckpts[nonzero], ci_lo[nonzero], ci_hi[nonzero],
                    alpha=0.2, color="gray", zorder=5,
                )

            # Zero-rate points: show upper CI bound with downward triangle
            zero_mask = rates == 0
            if zero_mask.any():
                ax.scatter(
                    ckpts[zero_mask], ci_hi[zero_mask],
                    marker="v", color="gray", s=40, alpha=0.6, zorder=10,
                    label="Upper CI (rate=0)",
                )

            # IS estimates per prefill level
            for pfx in prefill_levels:
                pfx_df = et_df[et_df["prefill_level"] == pfx].sort_values("checkpoint")
                if len(pfx_df) == 0:
                    continue

                is_vals = pfx_df["is_estimate"].values
                is_ckpts = pfx_df["checkpoint"].values
                is_lo = pfx_df["is_ci_lo"].values
                is_hi = pfx_df["is_ci_hi"].values

                nz = is_vals > 0
                if nz.any():
                    ax.plot(
                        is_ckpts[nz], is_vals[nz], "--",
                        color=pfx_colors[pfx], linewidth=1.2, alpha=0.8,
                        label=f"IS pfx={pfx}",
                    )
                    ax.fill_between(
                        is_ckpts[nz], is_lo[nz], is_hi[nz],
                        alpha=0.08, color=pfx_colors[pfx],
                    )

            ax.set_yscale("log")
            ax.set_xscale("log")
            ax.set_title(et.replace("_", " "), fontsize=10)
            ax.set_xlabel("Checkpoint", fontsize=9)
            ax.set_ylabel("P(exploit)", fontsize=9)
            ax.grid(True, alpha=0.3)

        # Hide unused axes
        for i in range(n_types, nrows * ncols):
            axes[i // ncols][i % ncols].set_visible(False)

        # Shared legend (deduplicated)
        handles, labels = [], []
        for ax_row in axes:
            for ax in ax_row:
                if ax.get_visible():
                    h, l = ax.get_legend_handles_labels()
                    for hh, ll in zip(h, l):
                        if ll not in labels:
                            handles.append(hh)
                            labels.append(ll)
        fig.legend(
            handles, labels, loc="center right",
            bbox_to_anchor=(1.15, 0.5), fontsize=7, ncol=1,
        )

        fig.suptitle(f"IS Calibration Trajectories: {run}", fontsize=12, y=1.02)
        fig.tight_layout()

        run_slug = _slugify(run)
        _save_fig(fig, output_dir / f"trajectory_comparison_{run_slug}.png")


# ---------------------------------------------------------------------------
# Plot 2: Calibration scatter by prefill level
# ---------------------------------------------------------------------------

def plot_calibration_scatter(df: pd.DataFrame, output_dir: Path):
    """Scatter IS estimate vs sampled rate by prefill level (log-log).

    One subplot per prefill level, pooling across runs/checkpoints/types.
    Points near diagonal = well-calibrated.
    """
    valid = df[(df["is_estimate"] > 0) & (df["sampled_rate"] > 0)].copy()

    prefill_levels = sorted(df["prefill_level"].unique())
    exploit_types = sorted(df["exploit_type"].unique())
    et_colors = _exploit_type_colormap(exploit_types)

    n_pfx = len(prefill_levels)
    ncols = min(3, n_pfx)
    nrows = (n_pfx + ncols - 1) // ncols

    fig, axes = plt.subplots(
        nrows, ncols, figsize=(5 * ncols, 5 * nrows), squeeze=False
    )

    # Consistent axis limits
    if len(valid) > 0:
        all_vals = np.concatenate([
            valid["is_estimate"].values,
            valid["sampled_rate"].values,
        ])
        vmin = all_vals[all_vals > 0].min() * 0.3
        vmax = all_vals.max() * 3
    else:
        vmin, vmax = 1e-4, 1.0

    for idx, pfx in enumerate(prefill_levels):
        ax = axes[idx // ncols][idx % ncols]
        pfx_data = valid[valid["prefill_level"] == pfx]

        # Diagonal
        ax.plot([vmin, vmax], [vmin, vmax], "k--", alpha=0.4, linewidth=1)

        for et in exploit_types:
            et_data = pfx_data[pfx_data["exploit_type"] == et]
            if len(et_data) == 0:
                continue

            ax.scatter(
                et_data["sampled_rate"], et_data["is_estimate"],
                color=et_colors[et], alpha=0.6, s=25,
                label=et.replace("_", " ")[:25],
            )
            ax.errorbar(
                et_data["sampled_rate"].values,
                et_data["is_estimate"].values,
                yerr=[
                    np.maximum(et_data["is_estimate"].values - et_data["is_ci_lo"].values, 0),
                    np.maximum(et_data["is_ci_hi"].values - et_data["is_estimate"].values, 0),
                ],
                xerr=[
                    np.maximum(et_data["sampled_rate"].values - et_data["wilson_ci_lo"].values, 0),
                    np.maximum(et_data["wilson_ci_hi"].values - et_data["sampled_rate"].values, 0),
                ],
                fmt="none", ecolor=et_colors[et], alpha=0.3, linewidth=0.5,
            )

        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlim(vmin, vmax)
        ax.set_ylim(vmin, vmax)
        ax.set_xlabel("Sampled rate (pfx=0)", fontsize=9)
        ax.set_ylabel("IS estimate", fontsize=9)
        ax.set_title(f"Prefill = {pfx}", fontsize=10)
        ax.grid(True, alpha=0.3)
        ax.set_aspect("equal")

    # Hide unused axes
    for idx in range(n_pfx, nrows * ncols):
        axes[idx // ncols][idx % ncols].set_visible(False)

    # Legend on first subplot
    if n_pfx > 0:
        axes[0][0].legend(fontsize=6, loc="upper left")

    fig.suptitle("Calibration: IS Estimate vs Sampled Rate", fontsize=12)
    fig.tight_layout()
    _save_fig(fig, output_dir / "calibration_scatter_by_prefill.png")


# ---------------------------------------------------------------------------
# Plot 3: KL vs tightness
# ---------------------------------------------------------------------------

def plot_kl_vs_tightness(df: pd.DataFrame, output_dir: Path):
    """Plot mean KL vs log(IS_estimate / sampled_rate).

    Shows how estimation quality degrades with KL divergence.
    """
    valid = df[(df["is_estimate"] > 0) & (df["sampled_rate"] > 0)].copy()
    if len(valid) == 0:
        print("  Skipping KL vs tightness plot: no valid data points")
        return

    valid["log_ratio"] = np.log(valid["is_estimate"] / valid["sampled_rate"])
    valid["log_ratio_lo"] = np.log(
        np.maximum(valid["is_ci_lo"], 1e-30) / valid["sampled_rate"]
    )
    valid["log_ratio_hi"] = np.log(
        np.maximum(valid["is_ci_hi"], 1e-30) / valid["sampled_rate"]
    )

    prefill_levels = sorted(valid["prefill_level"].unique())
    pfx_colors = _prefill_colormap(prefill_levels)

    fig, ax = plt.subplots(figsize=(10, 6))

    for pfx in prefill_levels:
        pfx_data = valid[valid["prefill_level"] == pfx]
        if len(pfx_data) == 0:
            continue
        ax.scatter(
            pfx_data["mean_kl"], pfx_data["log_ratio"],
            color=pfx_colors[pfx], alpha=0.5, s=20, label=f"pfx={pfx}",
        )
        ax.errorbar(
            pfx_data["mean_kl"].values,
            pfx_data["log_ratio"].values,
            yerr=[
                np.maximum(pfx_data["log_ratio"].values - pfx_data["log_ratio_lo"].values, 0),
                np.maximum(pfx_data["log_ratio_hi"].values - pfx_data["log_ratio"].values, 0),
            ],
            fmt="none", ecolor=pfx_colors[pfx], alpha=0.2, linewidth=0.5,
        )

    ax.axhline(0, color="black", linestyle="--", alpha=0.5, linewidth=1)
    ax.set_xlabel("Mean KL divergence", fontsize=11)
    ax.set_ylabel("log(IS estimate / sampled rate)", fontsize=11)
    ax.set_title("KL vs Estimation Tightness", fontsize=12)
    ax.legend(fontsize=8, loc="upper right", ncol=2)
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    _save_fig(fig, output_dir / "kl_vs_tightness.png")


# ---------------------------------------------------------------------------
# Plot 4: Coverage by prefill
# ---------------------------------------------------------------------------

def plot_coverage_by_prefill(df: pd.DataFrame, output_dir: Path):
    """For each prefill level, show fraction of cells where IS ≤ upper Wilson CI.

    Soft coverage: IS ≤ wilson_ci_hi of sampled rate.
    Strict coverage: IS ≤ sampled_rate point estimate.
    """
    prefill_levels = sorted(df["prefill_level"].unique())

    soft_coverage = []
    strict_coverage = []
    n_cells = []

    for pfx in prefill_levels:
        pfx_data = df[df["prefill_level"] == pfx].dropna(
            subset=["is_estimate", "wilson_ci_hi"]
        )
        n = len(pfx_data)
        if n == 0:
            soft_coverage.append(np.nan)
            strict_coverage.append(np.nan)
            n_cells.append(0)
            continue

        soft = (pfx_data["is_estimate"] <= pfx_data["wilson_ci_hi"]).mean()
        strict = (pfx_data["is_estimate"] <= pfx_data["sampled_rate"]).mean()

        soft_coverage.append(soft)
        strict_coverage.append(strict)
        n_cells.append(n)

    fig, ax = plt.subplots(figsize=(10, 6))

    x = np.arange(len(prefill_levels))
    width = 0.35

    ax.bar(
        x - width / 2, soft_coverage, width,
        label="Soft (IS ≤ upper CI)", color="steelblue", alpha=0.8,
    )
    ax.bar(
        x + width / 2, strict_coverage, width,
        label="Strict (IS ≤ point est.)", color="coral", alpha=0.8,
    )

    for i, n in enumerate(n_cells):
        y_top = max(
            soft_coverage[i] if not np.isnan(soft_coverage[i]) else 0,
            strict_coverage[i] if not np.isnan(strict_coverage[i]) else 0,
        )
        ax.text(i, y_top + 0.02, f"n={n}", ha="center", fontsize=7, color="gray")

    ax.set_xticks(x)
    ax.set_xticklabels([str(p) for p in prefill_levels])
    ax.set_xlabel("Prefill level (words)", fontsize=11)
    ax.set_ylabel("Coverage fraction", fontsize=11)
    ax.set_title("IS Lower Bound Coverage by Prefill Level", fontsize=12)
    ax.set_ylim(0, 1.15)
    ax.axhline(1.0, color="gray", linestyle="--", alpha=0.5)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3, axis="y")

    fig.tight_layout()
    _save_fig(fig, output_dir / "coverage_by_prefill.png")


# ---------------------------------------------------------------------------
# Plot 5: Rare-regime predictions
# ---------------------------------------------------------------------------

def plot_rare_regime(df: pd.DataFrame, output_dir: Path):
    """For cells where sampled_rate=0, show IS predictions vs upper CI bound.

    Key question: do IS estimates fall within the plausible range
    when we observe 0 exploits at prefill=0?
    """
    rare = df[df["sampled_rate"] == 0].copy()
    if len(rare) == 0:
        print("  Skipping rare-regime plot: no zero-rate cells")
        return

    rare_nonzero = rare[rare["is_estimate"] > 0].copy()
    prefill_levels = sorted(rare["prefill_level"].unique())
    pfx_colors = _prefill_colormap(prefill_levels)

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Panel 1: IS estimate vs upper CI bound
    ax = axes[0]
    for pfx in prefill_levels:
        pfx_data = rare_nonzero[rare_nonzero["prefill_level"] == pfx]
        if len(pfx_data) == 0:
            continue
        ax.scatter(
            pfx_data["wilson_ci_hi"], pfx_data["is_estimate"],
            color=pfx_colors[pfx], alpha=0.5, s=20, label=f"pfx={pfx}",
        )

    if len(rare_nonzero) > 0:
        all_vals = np.concatenate([
            rare_nonzero["wilson_ci_hi"].values,
            rare_nonzero["is_estimate"].values,
        ])
        pos = all_vals[all_vals > 0]
        if len(pos) > 0:
            vmin = pos.min() * 0.3
            vmax = pos.max() * 3
            ax.plot([vmin, vmax], [vmin, vmax], "k--", alpha=0.4)
            ax.set_xlim(vmin, vmax)
            ax.set_ylim(vmin, vmax)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Upper Wilson CI (sampled rate=0)", fontsize=11)
    ax.set_ylabel("IS estimate", fontsize=11)
    ax.set_title("Rare Regime: IS vs Upper CI Bound", fontsize=12)
    ax.legend(fontsize=7, ncol=2)
    ax.grid(True, alpha=0.3)

    # Panel 2: Coverage in rare regime, by prefill
    ax = axes[1]
    fracs = []
    counts = []
    for pfx in prefill_levels:
        pfx_data = rare[rare["prefill_level"] == pfx]
        n = len(pfx_data)
        if n == 0:
            fracs.append(np.nan)
            counts.append(0)
            continue
        frac = (pfx_data["is_estimate"] <= pfx_data["wilson_ci_hi"]).mean()
        fracs.append(frac)
        counts.append(n)

    x = np.arange(len(prefill_levels))
    ax.bar(x, fracs, color="steelblue", alpha=0.8)
    for i, (f, n) in enumerate(zip(fracs, counts)):
        if not np.isnan(f):
            ax.text(i, f + 0.02, f"n={n}", ha="center", fontsize=7, color="gray")

    ax.set_xticks(x)
    ax.set_xticklabels([str(p) for p in prefill_levels])
    ax.set_xlabel("Prefill level (words)", fontsize=11)
    ax.set_ylabel("Fraction IS ≤ upper CI", fontsize=11)
    ax.set_title("Rare Regime Coverage", fontsize=12)
    ax.set_ylim(0, 1.15)
    ax.axhline(1.0, color="gray", linestyle="--", alpha=0.5)
    ax.grid(True, alpha=0.3, axis="y")

    fig.tight_layout()
    _save_fig(fig, output_dir / "rare_regime_predictions.png")


# ---------------------------------------------------------------------------
# IS_noB trajectory plot (from compute_is_estimates.py JSON output)
# ---------------------------------------------------------------------------

def _load_pfx0_gt_per_type(
    evals_dir: Path,
    checkpoints: list[int],
    intentional_only: bool = True,
    excluded_tasks: set[str] | None = None,
) -> dict:
    """Load pfx=0 exploit rates per (checkpoint, exploit_type) from eval JSONL.

    Returns: {(ckpt, exploit_type): (rate, successes, attempts)}
    """
    import json as _json

    _INTENTIONAL_TYPES = {
        "argument_injection_leak", "error_code_abuse",
        "import_hook_side_channel", "inspect_module_abuse",
        "test_cases_honor_system", "trace_profiler_hook_oracle",
        "validator_honor_system", "verifier_logic_override",
    }

    result = {}
    for ckpt in checkpoints:
        # Try prefill_sensitivity naming, then hack_rate_early naming
        f = evals_dir / f"checkpoint-{ckpt}_prefill0.jsonl"
        if not f.exists():
            f = evals_dir / f"checkpoint-{ckpt}.jsonl"
        if not f.exists():
            continue

        counts = {}  # exploit_type -> [successes, total]
        with open(f) as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    row = _json.loads(line)
                except _json.JSONDecodeError:
                    continue
                et = row.get("exploit_type", "")
                tid = row.get("task_id", "")
                if intentional_only and et not in _INTENTIONAL_TYPES:
                    continue
                if excluded_tasks and tid in excluded_tasks:
                    continue
                if et not in counts:
                    counts[et] = [0, 0]
                counts[et][1] += 1
                if row.get("exploit_success"):
                    counts[et][0] += 1

        for et, (s, t) in counts.items():
            result[(ckpt, et)] = (s / t if t > 0 else 0.0, s, t)

    return result


def plot_is_nob_trajectory(
    is_json_path: Path,
    output_dir: Path,
    stitch_evals_dir: Path | None = None,
    stitch_checkpoints: list[int] | None = None,
    min_pfx: int = 5,
):
    """Plot IS_noB trajectory per exploit type with hack_rate_early GT.

    Args:
        is_json_path: IS estimates JSON from compute_is_estimates.py
        output_dir: Output directory for plots
        stitch_evals_dir: Optional T=0.4 pfx=0 evals dir for higher checkpoints
        stitch_checkpoints: Which checkpoints to stitch from T=0.4 data
        min_pfx: Minimum prefill level to include (default 5)
    """
    import json as _json

    with open(is_json_path) as f:
        data = _json.load(f)

    exploit_types = set()
    is_rows = []
    gt_rows = {}  # (ckpt, et) -> (rate, successes, attempts)
    is_checkpoints = set()

    for ckpt_info in data["checkpoints"]:
        ckpt = ckpt_info["checkpoint"]
        is_checkpoints.add(ckpt)
        for type_info in ckpt_info["per_type"]:
            et = type_info["exploit_type"]
            exploit_types.add(et)
            gt_rows[(ckpt, et)] = (
                type_info["gt_matched"],
                type_info.get("gt_successes", 0),
                type_info.get("gt_attempts", 0),
            )
            for pfx_info in type_info.get("per_prefill", []):
                if pfx_info["prefill_tokens"] < min_pfx:
                    continue
                is_rows.append({
                    "checkpoint": ckpt,
                    "exploit_type": et,
                    "prefill": pfx_info["prefill_tokens"],
                    "is_estimate": pfx_info["is_estimate"],
                })

    # Stitch GT from T=0.4 pfx=0 data at higher checkpoints
    if stitch_evals_dir and stitch_checkpoints:
        extra_gt = _load_pfx0_gt_per_type(
            Path(stitch_evals_dir),
            [c for c in stitch_checkpoints if c not in is_checkpoints],
            intentional_only=("intentional" in data.get("mode", "")),
        )
        for (ckpt, et), val in extra_gt.items():
            if (ckpt, et) not in gt_rows:
                gt_rows[(ckpt, et)] = val
                exploit_types.add(et)

    exploit_types = sorted(exploit_types)
    prefill_levels = sorted(set(r["prefill"] for r in is_rows))
    pfx_colors = _prefill_colormap(prefill_levels)

    n_types = len(exploit_types)
    ncols = min(4, n_types)
    nrows = (n_types + ncols - 1) // ncols

    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 4 * nrows), squeeze=False)

    for i, et in enumerate(exploit_types):
        ax = axes[i // ncols][i % ncols]

        # GT line
        gt_ckpts = sorted([k[0] for k in gt_rows if k[1] == et])
        gt_rates = np.array([gt_rows[(c, et)][0] for c in gt_ckpts])
        gt_s = np.array([gt_rows[(c, et)][1] for c in gt_ckpts])
        gt_n = np.array([gt_rows[(c, et)][2] for c in gt_ckpts])
        gt_ckpts_a = np.array(gt_ckpts)

        # Split into IS-range (hack_rate_early, filled) and stitched (T=0.4, open)
        is_mask = np.isin(gt_ckpts_a, list(is_checkpoints))

        for mask, marker, mfc, lbl in [
            (is_mask, "o", "black", "GT (64-att)"),
            (~is_mask, "o", "none", "GT (T=0.4, 1-att)"),
        ]:
            if not mask.any():
                continue
            nonzero = mask & (gt_rates > 0)
            if nonzero.any():
                ax.plot(
                    gt_ckpts_a[nonzero], gt_rates[nonzero], "-",
                    color="black", linewidth=1.5 if mfc == "black" else 1.0,
                    alpha=1.0 if mfc == "black" else 0.5, zorder=10,
                )
                ax.scatter(
                    gt_ckpts_a[nonzero], gt_rates[nonzero],
                    marker=marker, facecolors=mfc, edgecolors="black",
                    s=30, zorder=11, label=lbl,
                )
                if gt_n[nonzero].max() > 1:
                    ci_lo, ci_hi = wilson_ci_vectorized(gt_s[nonzero], gt_n[nonzero])
                    ax.fill_between(
                        gt_ckpts_a[nonzero], ci_lo, ci_hi,
                        alpha=0.15, color="gray", zorder=5,
                    )
            zero = mask & (gt_rates == 0) & (gt_n > 0)
            if zero.any():
                _, ci_hi_z = wilson_ci_vectorized(gt_s[zero], gt_n[zero])
                ax.scatter(
                    gt_ckpts_a[zero], ci_hi_z,
                    marker="v", color="gray", s=40, alpha=0.6, zorder=10,
                    label="Upper CI (rate=0)",
                )

        # Connect IS-range and stitched GT with a dotted line
        all_nonzero = gt_rates > 0
        if all_nonzero.sum() > 1:
            order = np.argsort(gt_ckpts_a[all_nonzero])
            ax.plot(
                gt_ckpts_a[all_nonzero][order], gt_rates[all_nonzero][order],
                "k:", linewidth=0.8, alpha=0.4, zorder=9,
            )

        # IS lines per prefill level
        for pfx in prefill_levels:
            pfx_rows = sorted(
                [r for r in is_rows if r["exploit_type"] == et and r["prefill"] == pfx],
                key=lambda r: r["checkpoint"],
            )
            if not pfx_rows:
                continue
            is_ckpts = np.array([r["checkpoint"] for r in pfx_rows])
            is_vals = np.array([r["is_estimate"] for r in pfx_rows])
            nz = is_vals > 0
            if nz.any():
                ax.plot(
                    is_ckpts[nz], is_vals[nz], "--",
                    color=pfx_colors[pfx], linewidth=1.2, alpha=0.8,
                    label=f"IS pfx={pfx}",
                )

        ax.set_yscale("log")
        ax.set_xscale("log")
        ax.set_title(et.replace("_", " "), fontsize=10)
        ax.set_xlabel("Checkpoint", fontsize=9)
        ax.set_ylabel("P(exploit)", fontsize=9)
        ax.grid(True, alpha=0.3)

    # Hide unused axes
    for i in range(n_types, nrows * ncols):
        axes[i // ncols][i % ncols].set_visible(False)

    # Shared legend (deduplicated)
    handles, labels = [], []
    for ax_row in axes:
        for ax in ax_row:
            if ax.get_visible():
                h, l = ax.get_legend_handles_labels()
                for hh, ll in zip(h, l):
                    if ll not in labels:
                        handles.append(hh)
                        labels.append(ll)
    fig.legend(
        handles, labels, loc="center right",
        bbox_to_anchor=(1.15, 0.5), fontsize=7, ncol=1,
    )

    b_label = "IS_noB" if data.get("no_b_correction") else "IS"
    mode = data.get("mode", "all")
    fig.suptitle(
        f"{b_label} Calibration Trajectories: {mode} / pfx≥{min_pfx}",
        fontsize=12, y=1.02,
    )
    fig.tight_layout()

    slug = f"is_nob_trajectory_{mode}_pfx{min_pfx}"
    _save_fig(fig, output_dir / f"{slug}.png")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Validate IS lower-bound estimates of P(exploit) against sampled rates.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--run-dir", nargs="+", type=Path, default=None,
        help="One or more prefill_sensitivity run directories (KL-based IS mode)",
    )
    parser.add_argument(
        "--output-dir", type=Path, required=True,
        help="Output directory for CSV and plots",
    )
    parser.add_argument(
        "--is-json", type=Path, default=None,
        help="IS estimates JSON from compute_is_estimates.py (IS_noB mode)",
    )
    parser.add_argument(
        "--stitch-evals-dir", type=Path, default=None,
        help="T=0.4 pfx=0 evals dir for stitching GT at higher checkpoints",
    )
    parser.add_argument(
        "--stitch-checkpoints", type=int, nargs="+", default=None,
        help="Checkpoints to stitch from T=0.4 data (e.g. 76 100 228 330)",
    )
    parser.add_argument(
        "--min-pfx", type=int, default=5,
        help="Minimum prefill level to include in IS_noB plot (default: 5)",
    )
    args = parser.parse_args()

    if args.is_json is None and args.run_dir is None:
        parser.error("Either --run-dir or --is-json is required")

    args.output_dir.mkdir(parents=True, exist_ok=True)

    # --- IS_noB mode: load from compute_is_estimates.py JSON ---
    if args.is_json is not None:
        print(f"Loading IS estimates from {args.is_json}")
        plot_is_nob_trajectory(
            args.is_json,
            args.output_dir,
            stitch_evals_dir=args.stitch_evals_dir,
            stitch_checkpoints=args.stitch_checkpoints,
            min_pfx=args.min_pfx,
        )
        print("\nDone!")
        return

    if args.run_dir is None:
        print("Error: --run-dir required for KL-based IS mode")
        return

    # Pre-load pfx0 donor data for pivot runs.
    # Pivot runs lack pfx=0 evals (prefill=0 is prefill-source-independent),
    # so we borrow from the original-pfx run with matching checkpoint_dir.
    pfx0_donors: dict[str, pd.DataFrame] = {}  # run_dir.name -> eval_df
    for run_dir in args.run_dir:
        donor = _find_pfx0_donor(run_dir, args.run_dir)
        if donor is not None:
            donor_evals = donor / "evals"
            if donor_evals.exists() and run_dir.name not in pfx0_donors:
                try:
                    donor_df = load_per_problem_results(donor_evals)
                    pfx0_donors[run_dir.name] = donor_df
                    print(f"Pivot run {run_dir.name} will use pfx=0 from {donor.name}")
                except Exception as e:
                    print(f"Warning: failed to load pfx0 donor {donor.name}: {e}")

    # Load data from all runs
    all_calibration = []
    all_aggregate = []

    for run_dir in args.run_dir:
        run_label = derive_run_label(run_dir)
        evals_dir = run_dir / "evals"
        kl_dir = run_dir / "kl"

        print(f"Processing {run_label}...")

        if not kl_dir.exists():
            print(f"  Skipping: no kl/ directory")
            continue

        pfx0_eval_df = pfx0_donors.get(run_dir.name)

        cal_df = compute_per_prefill_calibration(
            evals_dir, kl_dir, run_label, pfx0_eval_df=pfx0_eval_df,
        )
        agg_df = compute_aggregate_lb(evals_dir, kl_dir, run_label)

        if len(cal_df) > 0:
            all_calibration.append(cal_df)
            print(f"  {len(cal_df)} calibration rows")
        if len(agg_df) > 0:
            all_aggregate.append(agg_df)
            print(f"  {len(agg_df)} aggregate LB rows")

    if not all_calibration:
        print("No calibration data found. Exiting.")
        return

    cal_df = pd.concat(all_calibration, ignore_index=True)

    # Merge aggregate LB columns
    if all_aggregate:
        agg_df = pd.concat(all_aggregate, ignore_index=True)
        cal_df = cal_df.merge(
            agg_df,
            on=["run", "checkpoint", "exploit_type"],
            how="left",
        )

    # Save CSV
    csv_path = args.output_dir / "calibration_results.csv"
    cal_df.to_csv(csv_path, index=False)
    print(f"\nSaved CSV: {csv_path} ({len(cal_df)} rows)")

    # Summary statistics
    n_runs = cal_df["run"].nunique()
    n_types = cal_df["exploit_type"].nunique()
    n_ckpts = cal_df.groupby("run")["checkpoint"].nunique().mean()
    n_pfx = cal_df["prefill_level"].nunique()
    print(f"  {n_runs} runs, {n_types} exploit types, ~{n_ckpts:.0f} checkpoints/run, {n_pfx} prefill levels")

    # Calibration summary
    valid = cal_df[(cal_df["is_estimate"] > 0) & (cal_df["sampled_rate"] > 0)]
    if len(valid) > 0:
        median_ratio = np.median(valid["is_estimate"] / valid["sampled_rate"])
        print(f"  Median IS/sampled ratio: {median_ratio:.4f}")
        soft_cov = (cal_df["is_estimate"] <= cal_df["wilson_ci_hi"]).mean()
        strict_cov = (cal_df["is_estimate"] <= cal_df["sampled_rate"]).mean()
        print(f"  Overall soft coverage: {soft_cov:.1%}, strict coverage: {strict_cov:.1%}")

    # Generate plots
    print("\nGenerating plots...")
    plot_trajectory_comparison(cal_df, args.output_dir)
    plot_calibration_scatter(cal_df, args.output_dir)
    plot_kl_vs_tightness(cal_df, args.output_dir)
    plot_coverage_by_prefill(cal_df, args.output_dir)
    plot_rare_regime(cal_df, args.output_dir)

    print("\nDone!")


if __name__ == "__main__":
    main()
