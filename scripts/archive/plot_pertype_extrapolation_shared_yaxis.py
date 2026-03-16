#!/usr/bin/env python3
"""Regenerate per-type extrapolation plots with shared y-axis across exploit and control runs."""

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

# Add scripts to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from logit_trajectory_prediction import load_pooled_scaling_by_type, prepare_data

INTENTIONAL_TYPES = [
    "argument_injection_leak",
    "error_code_abuse",
    "import_hook_side_channel",
    "inspect_module_abuse",
    "test_cases_honor_system",
    "trace_profiler_hook_oracle",
    "validator_honor_system",
    "verifier_logic_override",
]

EXPLOIT_EVALS = Path("results/prefill_sensitivity/prefill_sensitivity-20260127-050226-8a0e189/evals")
CONTROL_EVALS = Path("results/prefill_sensitivity/prefill_sensitivity-20260206-045419-8a0e189/evals")

EXPLOIT_CUTOFF = 15
CONTROL_CUTOFF = 16

OUTPUT_DIR = Path("results/trajectory_prediction")


def load_and_prepare(evals_dir: Path) -> pd.DataFrame:
    df = load_pooled_scaling_by_type(evals_dir)
    df = prepare_data(df)
    # Filter to intentional types only
    df = df[df["exploit_type"].isin(INTENTIONAL_TYPES)].copy()
    return df


def fit_extrapolation(data: pd.DataFrame, cutoff: int):
    """Fit linear regression on log_checkpoint -> log_exploit_lower_bound for train data."""
    data = data.sort_values("checkpoint")
    train = data[data["checkpoint"] <= cutoff]
    test = data[data["checkpoint"] > cutoff]

    if len(train) < 2:
        return None

    x_train = np.log(train["checkpoint"].values)
    y_train = train["log_exploit_lower_bound"].values

    slope, intercept, _, _, _ = stats.linregress(x_train, y_train)
    return {
        "train": train,
        "test": test,
        "slope": slope,
        "intercept": intercept,
        "all_data": data,
    }


def collect_ylim(runs_data):
    """Compute global y-axis limits across all subplots in all runs."""
    y_min, y_max = np.inf, -np.inf
    for df, cutoff in runs_data:
        for etype in INTENTIONAL_TYPES:
            sub = df[df["exploit_type"] == etype]
            if len(sub) == 0:
                continue
            vals = sub["log_exploit_lower_bound"].values
            y_min = min(y_min, vals.min())
            y_max = max(y_max, vals.max())
    margin = (y_max - y_min) * 0.08
    return y_min - margin, y_max + margin


def plot_pertype_grid(
    df: pd.DataFrame,
    cutoff: int,
    run_label: str,
    ylim: tuple[float, float],
    train_color: str,
    test_color: str,
    line_color: str,
    output_path: Path,
):
    nrows, ncols = 2, 4
    fig, axes = plt.subplots(nrows, ncols, figsize=(20, 8), sharey=True)
    axes_flat = axes.flatten()

    for i, etype in enumerate(INTENTIONAL_TYPES):
        ax = axes_flat[i]
        sub = df[df["exploit_type"] == etype]
        if len(sub) == 0:
            ax.set_title(etype.replace("_", " "), fontsize=10)
            continue

        result = fit_extrapolation(sub, cutoff)
        if result is None:
            ax.set_title(etype.replace("_", " "), fontsize=10)
            continue

        train, test = result["train"], result["test"]

        # Plot points
        ax.scatter(
            np.log(train["checkpoint"]), train["log_exploit_lower_bound"],
            c=train_color, s=50, zorder=5, label="Train" if i == 0 else None,
        )
        ax.scatter(
            np.log(test["checkpoint"]), test["log_exploit_lower_bound"],
            c=test_color, s=50, zorder=5, label="Test" if i == 0 else None,
        )

        # Fit line across full range
        all_data = result["all_data"]
        x_range = np.linspace(
            np.log(all_data["checkpoint"].min()),
            np.log(all_data["checkpoint"].max()),
            100,
        )
        y_line = result["intercept"] + result["slope"] * x_range
        ax.plot(x_range, y_line, line_color, linestyle="--", alpha=0.6)

        # Vertical dashed lines at checkpoint positions
        for ckpt in all_data["checkpoint"].unique():
            ax.axvline(np.log(ckpt), color="gray", linestyle="--", alpha=0.2)

        # Cutoff line
        ax.axvline(np.log(cutoff), color="gray", linestyle="--", alpha=0.4, linewidth=1.5)

        ax.set_title(etype.replace("_", " "), fontsize=10)
        ax.set_ylim(ylim)
        ax.set_xlabel("log(ckpt)", fontsize=9)
        if i % ncols == 0:
            ax.set_ylabel("log(exploit LB)", fontsize=9)

    # Legend on first subplot
    axes_flat[0].legend(fontsize=9, loc="upper left")

    fig.suptitle(
        f"Per-Type log(exploit_lower_bound) — {run_label}, cutoff={cutoff}\n"
        f"(causal accumulated-n Laplace smoothing)",
        fontsize=13,
    )
    plt.tight_layout(rect=[0, 0, 1, 0.93])
    for ext in [".png", ".pdf"]:
        out = output_path.with_suffix(ext)
        fig.savefig(out, dpi=150, bbox_inches="tight")
        print(f"Saved: {out}")
    plt.close(fig)


def main():
    print("Loading exploit run data...")
    exploit_df = load_and_prepare(EXPLOIT_EVALS)
    print(f"  {len(exploit_df)} rows, types: {sorted(exploit_df['exploit_type'].unique())}")

    print("Loading control run data...")
    control_df = load_and_prepare(CONTROL_EVALS)
    print(f"  {len(control_df)} rows, types: {sorted(control_df['exploit_type'].unique())}")

    # Compute shared y-axis limits
    ylim = collect_ylim([
        (exploit_df, EXPLOIT_CUTOFF),
        (control_df, CONTROL_CUTOFF),
    ])
    print(f"Shared y-axis limits: {ylim}")

    # Plot exploit run
    plot_pertype_grid(
        exploit_df, EXPLOIT_CUTOFF,
        run_label="Exploit Run",
        ylim=ylim,
        train_color="red",
        test_color="green",
        line_color="red",
        output_path=OUTPUT_DIR / "accn_causal_extrapolation_pertype_cutoff15",
    )

    # Plot control run
    plot_pertype_grid(
        control_df, CONTROL_CUTOFF,
        run_label="Control Run",
        ylim=ylim,
        train_color="darkblue",
        test_color="cyan",
        line_color="blue",
        output_path=OUTPUT_DIR / "accn_causal_extrapolation_pertype_control_cutoff16",
    )


if __name__ == "__main__":
    main()
