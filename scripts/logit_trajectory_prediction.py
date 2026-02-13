#!/usr/bin/env python3
"""
Logit-space trajectory prediction: predict exploit rate at step T from history.

This script evaluates whether early exploit rates and KL metrics can predict
future exploit rates using autoregressive models in logit space.

Two aggregation modes:
- Pooled (default): average KL and exploit_rate across all (problem, prefill) pairs
- Max (legacy, --max-over-prefills): use max-over-prefills from scaling CSV

Key approaches:
1. AR(1) baseline: logit(rate_T) ~ logit(rate_{T-1})
2. Log-step extrapolation: fit logit(rate) ~ log(step) on history, extrapolate
3. KL as leading indicator: logit(rate_T) ~ mean_neg_kl_t
4. Gaussian Process: GP regression on log(step) with RBF kernel (handles irregular spacing)

Why logit transform:
- Exploit rate is bounded [0, 1]
- Logit maps to unbounded (-∞, +∞)
- Avoids predicting impossible values
- Linear dynamics in logit space = multiplicative dynamics in odds

Usage (pooled, default):
    python scripts/logit_trajectory_prediction.py \
        --evals-dir results/prefill_sensitivity/{RUN}/evals \
        --output-dir results/trajectory_prediction/{RUN}

Usage (max-over-prefills, legacy):
    python scripts/logit_trajectory_prediction.py \
        --input results/trajectory_analysis/{RUN}/intentional_only/exploit_rate_scaling_by_type.csv \
        --output-dir results/trajectory_prediction/{RUN} \
        --max-over-prefills
"""

import argparse
from pathlib import Path
from typing import Literal

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
from scipy.special import logit, expit
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel, ConstantKernel

from rh_indicators.run_utils import run_context


def smooth_logit(rate: float, n_samples: int, eps: float = 0.5) -> float:
    """
    Apply Laplace-style smoothing before logit transform.

    Maps rate to logit((rate * n + eps) / (n + 2*eps)) to avoid ±inf.
    For rate=0: gives logit(eps / (n + 2*eps)) ≈ small negative
    For rate=1: gives logit((n + eps) / (n + 2*eps)) ≈ large positive
    """
    smoothed = (rate * n_samples + eps) / (n_samples + 2 * eps)
    return logit(smoothed)


def smooth_logit_series(rates: pd.Series, n_samples: pd.Series, eps: float = 0.5) -> pd.Series:
    """Vectorized smooth logit."""
    smoothed = (rates * n_samples + eps) / (n_samples + 2 * eps)
    return np.log(smoothed / (1 - smoothed))  # logit


def inverse_logit(x: float | np.ndarray) -> float | np.ndarray:
    """Inverse logit (sigmoid)."""
    return expit(x)


def log_prob_to_logit(log_p: float | np.ndarray, max_logit: float = 10.0) -> float | np.ndarray:
    """
    Convert log probability to logit.

    logit(p) = log(p / (1-p)) = log(p) - log(1-p) = log_p - log(1 - exp(log_p))

    Uses numerically stable computation for small probabilities.
    Clamps output to [-max_logit, max_logit] to handle p >= 1 cases
    (which can occur with estimated bounds that exceed 1).
    """
    log_p = np.asarray(log_p)
    p = np.exp(log_p)
    # Clamp p to (0, 1-eps) to avoid log(0) or log(negative)
    p_clamped = np.clip(p, 1e-10, 1 - 1e-10)
    with np.errstate(divide='ignore'):
        log_1_minus_p = np.log1p(-p_clamped)
    logit_val = log_p - log_1_minus_p
    # Clamp extreme values
    return np.clip(logit_val, -max_logit, max_logit)


def prepare_data(df: pd.DataFrame) -> pd.DataFrame:
    """Prepare data with logit-transformed rates and exploit lower bounds."""
    df = df.copy()
    df["logit_rate"] = smooth_logit_series(df["exploit_rate"], df["n_samples"])
    df["log_checkpoint"] = np.log(df["checkpoint"])
    # Convert log_exploit_lower_bound to logit space
    df["logit_exploit_lower_bound"] = log_prob_to_logit(df["log_exploit_lower_bound"])
    return df


def evaluate_predictions(
    y_true_logit: np.ndarray,
    y_pred_logit: np.ndarray,
    y_true_rate: np.ndarray,
) -> dict:
    """Compute evaluation metrics in both logit and rate space."""
    # Back-transform predictions to rate space
    y_pred_rate = inverse_logit(y_pred_logit)

    # Logit space metrics
    rmse_logit = np.sqrt(np.mean((y_true_logit - y_pred_logit) ** 2))
    ss_res_logit = np.sum((y_true_logit - y_pred_logit) ** 2)
    ss_tot_logit = np.sum((y_true_logit - np.mean(y_true_logit)) ** 2)
    r2_logit = 1 - ss_res_logit / ss_tot_logit if ss_tot_logit > 0 else np.nan

    # Rate space metrics
    rmse_rate = np.sqrt(np.mean((y_true_rate - y_pred_rate) ** 2))
    ss_res_rate = np.sum((y_true_rate - y_pred_rate) ** 2)
    ss_tot_rate = np.sum((y_true_rate - np.mean(y_true_rate)) ** 2)
    r2_rate = 1 - ss_res_rate / ss_tot_rate if ss_tot_rate > 0 else np.nan

    return {
        "rmse_logit": rmse_logit,
        "r2_logit": r2_logit,
        "rmse_rate": rmse_rate,
        "r2_rate": r2_rate,
        "n_predictions": len(y_true_logit),
    }


def ar1_prediction(df: pd.DataFrame, exploit_type: str | None = None) -> dict:
    """
    AR(1) model: predict logit(rate_T) from logit(rate_{T-1}).

    Returns predictions and metrics for all consecutive checkpoint pairs.
    """
    if exploit_type:
        data = df[df["exploit_type"] == exploit_type].copy()
    else:
        # Aggregate across exploit types
        data = df.groupby("checkpoint").agg({
            "exploit_rate": "mean",
            "logit_rate": "mean",
            "n_samples": "sum",
        }).reset_index()

    data = data.sort_values("checkpoint")
    checkpoints = data["checkpoint"].values
    logit_rates = data["logit_rate"].values
    rates = data["exploit_rate"].values if "exploit_rate" in data else inverse_logit(logit_rates)

    if len(checkpoints) < 2:
        return {"error": "insufficient data"}

    # Pairs: (T-1, T) for consecutive checkpoints
    y_true_logit = logit_rates[1:]
    y_true_rate = rates[1:] if isinstance(rates, np.ndarray) else inverse_logit(y_true_logit)
    x_prev_logit = logit_rates[:-1]

    # Fit AR(1): y = alpha + beta * x
    if np.std(x_prev_logit) < 1e-10:
        return {"error": "constant values (no variation in logit rates)"}
    slope, intercept, r_value, p_value, std_err = stats.linregress(x_prev_logit, y_true_logit)
    y_pred_logit = intercept + slope * x_prev_logit

    metrics = evaluate_predictions(y_true_logit, y_pred_logit, y_true_rate)
    metrics.update({
        "model": "AR(1)",
        "exploit_type": exploit_type or "aggregate",
        "ar1_intercept": intercept,
        "ar1_slope": slope,
        "ar1_p_value": p_value,
        "checkpoints_predicted": list(checkpoints[1:]),
    })

    return {
        "metrics": metrics,
        "predictions": pd.DataFrame({
            "checkpoint": checkpoints[1:],
            "y_true_logit": y_true_logit,
            "y_pred_logit": y_pred_logit,
            "y_true_rate": y_true_rate,
            "y_pred_rate": inverse_logit(y_pred_logit),
        }),
    }


def extrapolation_prediction(
    df: pd.DataFrame,
    cutoff_checkpoint: int,
    target_checkpoint: int | None = None,
    exploit_type: str | None = None,
) -> dict:
    """
    Log-step extrapolation: fit logit(exploit_lower_bound) ~ log(checkpoint) on history, extrapolate.

    Uses logit of exploit_lower_bound (converted from log probability):
    - Defined even when exploit_rate = 0 (exploit_lower_bound is always > 0)
    - Logit space should be more linear for extrapolation than raw log probability

    Args:
        df: DataFrame with logit_exploit_lower_bound and log_checkpoint columns
        cutoff_checkpoint: fit on checkpoints <= cutoff
        target_checkpoint: predict this checkpoint (if None, predict all > cutoff)
        exploit_type: filter to specific exploit type (or aggregate if None)
    """
    if exploit_type:
        data = df[df["exploit_type"] == exploit_type].copy()
    else:
        data = df.groupby("checkpoint").agg({
            "exploit_rate": "mean",
            "logit_exploit_lower_bound": "mean",
            "log_checkpoint": "first",
            "n_samples": "sum",
        }).reset_index()

    data = data.sort_values("checkpoint")

    # Split into train (history) and test (future)
    train = data[data["checkpoint"] <= cutoff_checkpoint]
    if target_checkpoint:
        test = data[data["checkpoint"] == target_checkpoint]
    else:
        test = data[data["checkpoint"] > cutoff_checkpoint]

    if len(train) < 2 or len(test) == 0:
        return {"error": "insufficient data for extrapolation"}

    if np.std(train["logit_exploit_lower_bound"].values) < 1e-10:
        return {"error": "constant values (no variation in logit rates)"}

    # Fit linear model on log(checkpoint) -> logit(exploit_lower_bound)
    slope, intercept, r_value, p_value, std_err = stats.linregress(
        train["log_checkpoint"], train["logit_exploit_lower_bound"]
    )

    # Predict
    y_pred_logit = intercept + slope * test["log_checkpoint"].values
    y_true_logit = test["logit_exploit_lower_bound"].values

    # Compute metrics in logit space
    rmse = np.sqrt(np.mean((y_true_logit - y_pred_logit) ** 2))
    ss_res = np.sum((y_true_logit - y_pred_logit) ** 2)
    ss_tot = np.sum((y_true_logit - np.mean(y_true_logit)) ** 2)
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else np.nan

    # Persistence baseline: predict last training value for all test points
    # More meaningful than R² for forecasting - measures improvement over "no change"
    y_persistence = train["logit_exploit_lower_bound"].iloc[-1]
    ss_persistence = np.sum((y_true_logit - y_persistence) ** 2)
    skill_score = 1 - ss_res / ss_persistence if ss_persistence > 0 else np.nan

    metrics = {
        "rmse_logit": rmse,
        "r2_logit": r2,
        "skill_score": skill_score,  # >0 means better than persistence baseline
        "n_predictions": len(y_true_logit),
        "model": "logit_extrapolation",
        "exploit_type": exploit_type or "aggregate",
        "cutoff_checkpoint": cutoff_checkpoint,
        "slope": slope,
        "intercept": intercept,
        "train_r2": r_value ** 2,
        "train_p_value": p_value,
        "checkpoints_predicted": list(test["checkpoint"]),
        "persistence_baseline": y_persistence,
    }

    return {
        "metrics": metrics,
        "predictions": pd.DataFrame({
            "checkpoint": test["checkpoint"].values,
            "y_true_logit": y_true_logit,
            "y_pred_logit": y_pred_logit,
            "y_true_rate": test["exploit_rate"].values,
            "y_pred_prob": inverse_logit(y_pred_logit),
        }),
    }


def kl_prediction(
    df: pd.DataFrame,
    predictor_checkpoint: int,
    target_checkpoint: int | None = None,
    exploit_type: str | None = None,
    predictor_col: str = "log_exploit_lower_bound",
) -> dict:
    """
    Leading indicator prediction: predict logit(rate_T) from a metric at earlier checkpoint.

    Tests whether the chosen predictor (default: log_exploit_lower_bound) predicts
    exploit rate better than the rate itself. Fits across exploit types.

    Args:
        predictor_col: Column to use as predictor. Options:
            - "log_exploit_lower_bound" (default): log(max_prefill[P(prefill)*P(exploit|prefill)])
            - "mean_neg_kl": negative KL divergence
    """
    if exploit_type:
        data = df[df["exploit_type"] == exploit_type].copy()
    else:
        data = df.copy()

    # Get predictor data at predictor_checkpoint
    predictor_data = data[data["checkpoint"] == predictor_checkpoint][
        ["exploit_type", predictor_col, "logit_rate", "exploit_rate"]
    ].rename(columns={
        predictor_col: "metric_predictor",
        "logit_rate": "logit_predictor",
        "exploit_rate": "rate_predictor",
    })

    # Get target data
    if target_checkpoint:
        target_data = data[data["checkpoint"] == target_checkpoint][
            ["exploit_type", "logit_rate", "exploit_rate"]
        ].rename(columns={
            "logit_rate": "logit_target",
            "exploit_rate": "rate_target",
        })
    else:
        # Use final checkpoint
        final_ckpt = data["checkpoint"].max()
        target_data = data[data["checkpoint"] == final_ckpt][
            ["exploit_type", "logit_rate", "exploit_rate"]
        ].rename(columns={
            "logit_rate": "logit_target",
            "exploit_rate": "rate_target",
        })
        target_checkpoint = final_ckpt

    # Merge on exploit_type
    merged = predictor_data.merge(target_data, on="exploit_type")

    if len(merged) < 3:
        return {"error": "insufficient exploit types for metric prediction"}

    if np.std(merged["metric_predictor"].values) < 1e-10 or np.std(merged["logit_predictor"].values) < 1e-10:
        return {"error": "constant predictor values (no variation)"}

    # Fit: logit_target ~ metric_predictor
    slope_metric, intercept_metric, r_metric, p_metric, _ = stats.linregress(
        merged["metric_predictor"], merged["logit_target"]
    )
    y_pred_metric = intercept_metric + slope_metric * merged["metric_predictor"].values

    # Fit: logit_target ~ logit_predictor (baseline)
    slope_rate, intercept_rate, r_rate, p_rate, _ = stats.linregress(
        merged["logit_predictor"], merged["logit_target"]
    )
    y_pred_rate = intercept_rate + slope_rate * merged["logit_predictor"].values

    y_true_logit = merged["logit_target"].values
    y_true_rate = merged["rate_target"].values

    metrics_metric = evaluate_predictions(y_true_logit, y_pred_metric, y_true_rate)
    metrics_rate = evaluate_predictions(y_true_logit, y_pred_rate, y_true_rate)

    # Clean model name from column
    model_name = predictor_col.replace("_", " ").title().replace(" ", "_")

    return {
        "metrics_metric": {
            "model": f"{model_name}_predictor",
            "predictor_col": predictor_col,
            "predictor_checkpoint": predictor_checkpoint,
            "target_checkpoint": target_checkpoint,
            "slope": slope_metric,
            "intercept": intercept_metric,
            "p_value": p_metric,
            **metrics_metric,
        },
        "metrics_rate_baseline": {
            "model": "rate_predictor",
            "predictor_checkpoint": predictor_checkpoint,
            "target_checkpoint": target_checkpoint,
            "slope": slope_rate,
            "intercept": intercept_rate,
            "p_value": p_rate,
            **metrics_rate,
        },
        "predictions": merged.assign(
            y_pred_metric=y_pred_metric,
            y_pred_rate_baseline=y_pred_rate,
        ),
    }


def kl_prediction_loo(
    df: pd.DataFrame,
    predictor_checkpoint: int,
    target_checkpoint: int | None = None,
    predictor_col: str = "log_exploit_lower_bound",
) -> dict:
    """
    Leading indicator with leave-one-out cross-validation over exploit types.

    For each exploit type:
    - Train: fit metric→rate and rate→rate on other exploit types
    - Test: predict held-out exploit type

    This tests whether the metric generalizes as a predictor across exploit types.
    """
    data = df.copy()

    # Get predictor data at predictor_checkpoint
    predictor_data = data[data["checkpoint"] == predictor_checkpoint][
        ["exploit_type", predictor_col, "logit_rate", "exploit_rate"]
    ].rename(columns={
        predictor_col: "metric_predictor",
        "logit_rate": "logit_predictor",
        "exploit_rate": "rate_predictor",
    })

    # Get target data
    if target_checkpoint is None:
        target_checkpoint = data["checkpoint"].max()

    target_data = data[data["checkpoint"] == target_checkpoint][
        ["exploit_type", "logit_rate", "exploit_rate"]
    ].rename(columns={
        "logit_rate": "logit_target",
        "exploit_rate": "rate_target",
    })

    # Merge on exploit_type
    merged = predictor_data.merge(target_data, on="exploit_type")
    exploit_types = merged["exploit_type"].unique()

    if len(exploit_types) < 3:
        return {"error": "insufficient exploit types for LOO CV"}

    # Leave-one-out predictions
    loo_results = []
    for held_out in exploit_types:
        train = merged[merged["exploit_type"] != held_out]
        test = merged[merged["exploit_type"] == held_out].iloc[0]

        # Fit on train
        if np.std(train["metric_predictor"].values) < 1e-10 or np.std(train["logit_predictor"].values) < 1e-10:
            continue
        slope_metric, intercept_metric, _, _, _ = stats.linregress(
            train["metric_predictor"], train["logit_target"]
        )
        slope_rate, intercept_rate, _, _, _ = stats.linregress(
            train["logit_predictor"], train["logit_target"]
        )

        # Predict on test
        pred_metric = intercept_metric + slope_metric * test["metric_predictor"]
        pred_rate = intercept_rate + slope_rate * test["logit_predictor"]

        loo_results.append({
            "exploit_type": held_out,
            "y_true_logit": test["logit_target"],
            "y_true_rate": test["rate_target"],
            "y_pred_metric": pred_metric,
            "y_pred_rate": pred_rate,
            "metric_predictor": test["metric_predictor"],
            "logit_predictor": test["logit_predictor"],
        })

    loo_df = pd.DataFrame(loo_results)

    # Compute metrics
    y_true_logit = loo_df["y_true_logit"].values
    y_true_rate = loo_df["y_true_rate"].values
    y_pred_metric = loo_df["y_pred_metric"].values
    y_pred_rate = loo_df["y_pred_rate"].values

    metrics_metric = evaluate_predictions(y_true_logit, y_pred_metric, y_true_rate)
    metrics_rate = evaluate_predictions(y_true_logit, y_pred_rate, y_true_rate)

    # Clean model name from column
    model_name = predictor_col.replace("_", " ").title().replace(" ", "_")

    return {
        "metrics_metric": {
            "model": f"{model_name}_predictor_LOO",
            "predictor_col": predictor_col,
            "predictor_checkpoint": predictor_checkpoint,
            "target_checkpoint": target_checkpoint,
            **metrics_metric,
        },
        "metrics_rate_baseline": {
            "model": "rate_predictor_LOO",
            "predictor_checkpoint": predictor_checkpoint,
            "target_checkpoint": target_checkpoint,
            **metrics_rate,
        },
        "predictions": loo_df,
    }


def gp_prediction(
    df: pd.DataFrame,
    cutoff_checkpoint: int,
    exploit_type: str | None = None,
) -> dict:
    """
    Gaussian Process regression: fit log_exploit_lower_bound ~ log(checkpoint).

    Uses log_exploit_lower_bound instead of logit(rate) because:
    - Defined even when exploit_rate = 0 (exploit_lower_bound is always > 0)
    - Logit space should be more linear for extrapolation

    Uses RBF kernel which handles irregular spacing naturally and provides
    uncertainty estimates for predictions.
    """
    if exploit_type:
        data = df[df["exploit_type"] == exploit_type].copy()
    else:
        data = df.groupby("checkpoint").agg({
            "exploit_rate": "mean",
            "logit_exploit_lower_bound": "mean",
            "log_checkpoint": "first",
            "n_samples": "sum",
        }).reset_index()

    data = data.sort_values("checkpoint")

    # Split into train (history) and test (future)
    train = data[data["checkpoint"] <= cutoff_checkpoint]
    test = data[data["checkpoint"] > cutoff_checkpoint]

    if len(train) < 2 or len(test) == 0:
        return {"error": "insufficient data for GP prediction"}

    # Prepare data - use logit(exploit_lower_bound) as target
    X_train = train["log_checkpoint"].values.reshape(-1, 1)
    y_train = train["logit_exploit_lower_bound"].values
    X_test = test["log_checkpoint"].values.reshape(-1, 1)
    y_test = test["logit_exploit_lower_bound"].values

    # Define kernel: constant * RBF + noise
    kernel = (
        ConstantKernel(1.0, (0.1, 10.0)) *
        RBF(length_scale=1.0, length_scale_bounds=(0.1, 10.0)) +
        WhiteKernel(noise_level=0.1, noise_level_bounds=(1e-3, 1.0))
    )

    gp = GaussianProcessRegressor(
        kernel=kernel,
        n_restarts_optimizer=5,
        random_state=42,
    )
    gp.fit(X_train, y_train)

    # Predict with uncertainty
    y_pred, y_std = gp.predict(X_test, return_std=True)

    # Compute metrics in logit space
    rmse = np.sqrt(np.mean((y_test - y_pred) ** 2))
    ss_res = np.sum((y_test - y_pred) ** 2)
    ss_tot = np.sum((y_test - np.mean(y_test)) ** 2)
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else np.nan

    # Persistence baseline: predict last training value for all test points
    y_persistence = y_train[-1]
    ss_persistence = np.sum((y_test - y_persistence) ** 2)
    skill_score = 1 - ss_res / ss_persistence if ss_persistence > 0 else np.nan

    # Also get predictions on full range for plotting
    X_full = data["log_checkpoint"].values.reshape(-1, 1)
    y_full_pred, y_full_std = gp.predict(X_full, return_std=True)

    metrics = {
        "rmse_logit": rmse,
        "r2_logit": r2,
        "skill_score": skill_score,  # >0 means better than persistence baseline
        "n_predictions": len(y_test),
        "model": "gaussian_process",
        "exploit_type": exploit_type or "aggregate",
        "cutoff_checkpoint": cutoff_checkpoint,
        "kernel_params": str(gp.kernel_),
        "log_marginal_likelihood": gp.log_marginal_likelihood_value_,
        "checkpoints_predicted": list(test["checkpoint"]),
        "persistence_baseline": y_persistence,
    }

    return {
        "metrics": metrics,
        "predictions": pd.DataFrame({
            "checkpoint": test["checkpoint"].values,
            "y_true_logit": y_test,
            "y_pred_logit": y_pred,
            "y_pred_std": y_std,
            "y_true_rate": test["exploit_rate"].values,
            "y_pred_prob": inverse_logit(y_pred),
        }),
        "full_predictions": pd.DataFrame({
            "checkpoint": data["checkpoint"].values,
            "log_checkpoint": data["log_checkpoint"].values,
            "y_true_logit": data["logit_exploit_lower_bound"].values,
            "y_pred_logit": y_full_pred,
            "y_pred_std": y_full_std,
            "is_train": data["checkpoint"] <= cutoff_checkpoint,
        }),
        "gp_model": gp,
    }


def plot_gp_prediction(
    results: dict,
    output_path: Path,
    cutoff_checkpoint: int,
    exploit_type: str | None = None,
) -> None:
    """Plot GP predictions with uncertainty bands for logit(exploit_lower_bound)."""
    full = results["full_predictions"]
    metrics = results["metrics"]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Logit space with uncertainty
    ax = axes[0]
    train_mask = full["is_train"]

    # Training points
    ax.scatter(full.loc[train_mask, "log_checkpoint"], full.loc[train_mask, "y_true_logit"],
               c="blue", s=60, label="Training", zorder=3)
    # Test points
    ax.scatter(full.loc[~train_mask, "log_checkpoint"], full.loc[~train_mask, "y_true_logit"],
               c="green", s=60, label="Test (actual)", zorder=3)

    # GP mean and uncertainty
    x_plot = full["log_checkpoint"].values
    y_mean = full["y_pred_logit"].values
    y_std = full["y_pred_std"].values

    ax.plot(x_plot, y_mean, "r-", label="GP mean", zorder=2)
    ax.fill_between(x_plot, y_mean - 2*y_std, y_mean + 2*y_std,
                    alpha=0.2, color="red", label="±2σ", zorder=1)

    ax.axvline(np.log(cutoff_checkpoint), color="gray", linestyle="--", alpha=0.5, label=f"Cutoff (ckpt {cutoff_checkpoint})")
    ax.set_xlabel("log(checkpoint)", fontsize=11)
    ax.set_ylabel("logit(exploit_lower_bound)", fontsize=11)
    skill = metrics.get('skill_score', np.nan)
    ax.set_title(f"GP Logit Space (skill={skill:.3f})", fontsize=12)
    ax.legend(loc="best", fontsize=9)
    ax.grid(True, alpha=0.3)

    # Probability space (inverse_logit of logit = probability lower bound)
    ax = axes[1]
    y_prob_mean = inverse_logit(y_mean)
    y_prob_lower = inverse_logit(y_mean - 2*y_std)
    y_prob_upper = inverse_logit(y_mean + 2*y_std)

    ax.scatter(full.loc[train_mask, "checkpoint"], inverse_logit(full.loc[train_mask, "y_true_logit"]),
               c="blue", s=60, label="Training", zorder=3)
    ax.scatter(full.loc[~train_mask, "checkpoint"], inverse_logit(full.loc[~train_mask, "y_true_logit"]),
               c="green", s=60, label="Test (actual)", zorder=3)

    ax.plot(full["checkpoint"], y_prob_mean, "r-", label="GP mean", zorder=2)
    ax.fill_between(full["checkpoint"], y_prob_lower, np.minimum(y_prob_upper, 1.5),
                    alpha=0.2, color="red", label="±2σ", zorder=1)

    ax.axvline(cutoff_checkpoint, color="gray", linestyle="--", alpha=0.5, label=f"Cutoff")
    ax.set_xlabel("Checkpoint", fontsize=11)
    ax.set_ylabel("exploit_lower_bound (probability)", fontsize=11)
    ax.set_title(f"GP Probability Space", fontsize=12)
    ax.set_ylim(-0.05, 1.05)
    ax.legend(loc="best", fontsize=9)
    ax.grid(True, alpha=0.3)

    title = f"Gaussian Process: cutoff {cutoff_checkpoint}"
    if exploit_type:
        title += f" ({exploit_type})"
    else:
        title += " (aggregate)"
    plt.suptitle(title, fontsize=14)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.savefig(output_path.with_suffix('.pdf'), bbox_inches='tight')
    print(f"Plot saved to: {output_path}")
    plt.close()


def plot_ar1_fit(
    results: dict,
    output_path: Path,
    title: str | None = None,
) -> None:
    """Plot AR(1) predictions vs actuals."""
    preds = results["predictions"]
    metrics = results["metrics"]

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Logit space
    ax = axes[0]
    ax.scatter(preds["y_true_logit"], preds["y_pred_logit"], alpha=0.7, s=60)
    lims = [
        min(preds["y_true_logit"].min(), preds["y_pred_logit"].min()) - 0.5,
        max(preds["y_true_logit"].max(), preds["y_pred_logit"].max()) + 0.5,
    ]
    ax.plot(lims, lims, "k--", alpha=0.5, label="y=x")
    ax.set_xlabel("True logit(rate)", fontsize=11)
    ax.set_ylabel("Predicted logit(rate)", fontsize=11)
    ax.set_title(f"Logit Space (R²={metrics['r2_logit']:.3f})", fontsize=12)
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Rate space
    ax = axes[1]
    ax.scatter(preds["y_true_rate"], preds["y_pred_rate"], alpha=0.7, s=60)
    ax.plot([0, 1], [0, 1], "k--", alpha=0.5, label="y=x")
    ax.set_xlabel("True exploit rate", fontsize=11)
    ax.set_ylabel("Predicted exploit rate", fontsize=11)
    ax.set_title(f"Rate Space (R²={metrics['r2_rate']:.3f})", fontsize=12)
    ax.set_xlim(-0.05, 1.05)
    ax.set_ylim(-0.05, 1.05)
    ax.legend()
    ax.grid(True, alpha=0.3)

    suptitle = title or f"AR(1) Prediction: {metrics['exploit_type']}"
    plt.suptitle(suptitle, fontsize=14)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.savefig(output_path.with_suffix('.pdf'), bbox_inches='tight')
    print(f"Plot saved to: {output_path}")
    plt.close()


def plot_extrapolation(
    df: pd.DataFrame,
    cutoff_checkpoint: int,
    results: dict,
    output_path: Path,
    exploit_type: str | None = None,
) -> None:
    """Plot extrapolation fit: training data, fit line, and predictions for logit(exploit_lower_bound)."""
    if exploit_type:
        data = df[df["exploit_type"] == exploit_type].copy()
    else:
        data = df.groupby("checkpoint").agg({
            "exploit_rate": "mean",
            "logit_exploit_lower_bound": "mean",
            "log_checkpoint": "first",
        }).reset_index()

    data = data.sort_values("checkpoint")
    train = data[data["checkpoint"] <= cutoff_checkpoint]
    test = data[data["checkpoint"] > cutoff_checkpoint]
    metrics = results["metrics"]

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Logit space
    ax = axes[0]
    ax.scatter(train["log_checkpoint"], train["logit_exploit_lower_bound"],
               label="Training", alpha=0.8, s=60, c="blue")
    ax.scatter(test["log_checkpoint"], test["logit_exploit_lower_bound"],
               label="Test (actual)", alpha=0.8, s=60, c="green")

    # Fit line
    x_line = np.linspace(data["log_checkpoint"].min(), data["log_checkpoint"].max(), 100)
    y_line = metrics["intercept"] + metrics["slope"] * x_line
    ax.plot(x_line, y_line, "r--", alpha=0.7, label="Fit")

    # Predictions
    preds = results["predictions"]
    ax.scatter(np.log(preds["checkpoint"]), preds["y_pred_logit"],
               marker="x", s=80, c="red", label="Predictions")

    ax.set_xlabel("log(checkpoint)", fontsize=11)
    ax.set_ylabel("logit(exploit_lower_bound)", fontsize=11)
    skill = metrics.get('skill_score', np.nan)
    ax.set_title(f"Logit Space (skill={skill:.3f})", fontsize=12)
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Probability space
    ax = axes[1]
    ax.scatter(train["checkpoint"], inverse_logit(train["logit_exploit_lower_bound"]),
               label="Training", alpha=0.8, s=60, c="blue")
    ax.scatter(test["checkpoint"], inverse_logit(test["logit_exploit_lower_bound"]),
               label="Test (actual)", alpha=0.8, s=60, c="green")
    ax.scatter(preds["checkpoint"], preds["y_pred_prob"],
               marker="x", s=80, c="red", label="Predictions")

    ax.set_xlabel("Checkpoint", fontsize=11)
    ax.set_ylabel("exploit_lower_bound (probability)", fontsize=11)
    ax.set_title("Probability Space", fontsize=12)
    ax.set_ylim(-0.05, 1.5)
    ax.legend()
    ax.grid(True, alpha=0.3)

    title = f"Extrapolation from checkpoint {cutoff_checkpoint}"
    if exploit_type:
        title += f" ({exploit_type})"
    else:
        title += " (aggregate)"
    plt.suptitle(title, fontsize=14)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.savefig(output_path.with_suffix('.pdf'), bbox_inches='tight')
    print(f"Plot saved to: {output_path}")
    plt.close()


def plot_metric_comparison(
    results: dict,
    output_path: Path,
) -> None:
    """Plot metric predictor vs rate predictor comparison."""
    preds = results["predictions"]
    metrics_metric = results["metrics_metric"]
    metrics_rate = results["metrics_rate_baseline"]

    # Get predictor name for labels
    predictor_name = metrics_metric.get("predictor_col", "metric").replace("_", " ")

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # Metric predictor
    ax = axes[0]
    ax.scatter(preds["metric_predictor"], preds["logit_target"], alpha=0.7, s=60)
    x_line = np.linspace(preds["metric_predictor"].min(), preds["metric_predictor"].max(), 100)
    y_line = metrics_metric["intercept"] + metrics_metric["slope"] * x_line
    ax.plot(x_line, y_line, "r--", alpha=0.7)
    ax.set_xlabel(f"{predictor_name} (predictor checkpoint)", fontsize=11)
    ax.set_ylabel("logit(rate) at target", fontsize=11)
    ax.set_title(f"Metric (R²={metrics_metric['r2_logit']:.3f}, p={metrics_metric['p_value']:.3f})", fontsize=11)
    ax.grid(True, alpha=0.3)

    # Rate predictor
    ax = axes[1]
    ax.scatter(preds["logit_predictor"], preds["logit_target"], alpha=0.7, s=60)
    x_line = np.linspace(preds["logit_predictor"].min(), preds["logit_predictor"].max(), 100)
    y_line = metrics_rate["intercept"] + metrics_rate["slope"] * x_line
    ax.plot(x_line, y_line, "r--", alpha=0.7)
    ax.set_xlabel("logit(rate) at predictor checkpoint", fontsize=11)
    ax.set_ylabel("logit(rate) at target", fontsize=11)
    ax.set_title(f"Rate Predictor (R²={metrics_rate['r2_logit']:.3f}, p={metrics_rate['p_value']:.3f})", fontsize=11)
    ax.grid(True, alpha=0.3)

    # Comparison: predicted vs actual
    ax = axes[2]
    ax.scatter(preds["logit_target"], preds["y_pred_metric"],
               alpha=0.7, s=60, label=f"Metric (R²={metrics_metric['r2_logit']:.3f})")
    ax.scatter(preds["logit_target"], preds["y_pred_rate_baseline"],
               alpha=0.7, s=60, marker="s", label=f"Rate (R²={metrics_rate['r2_logit']:.3f})")
    lims = [
        min(preds["logit_target"].min(), preds["y_pred_metric"].min(), preds["y_pred_rate_baseline"].min()) - 0.5,
        max(preds["logit_target"].max(), preds["y_pred_metric"].max(), preds["y_pred_rate_baseline"].max()) + 0.5,
    ]
    ax.plot(lims, lims, "k--", alpha=0.5, label="y=x")
    ax.set_xlabel("True logit(rate)", fontsize=11)
    ax.set_ylabel("Predicted logit(rate)", fontsize=11)
    ax.set_title("Comparison", fontsize=12)
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.suptitle(
        f"Metric vs Rate as Predictor (checkpoint {metrics_metric['predictor_checkpoint']} → {metrics_metric['target_checkpoint']})",
        fontsize=14
    )
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.savefig(output_path.with_suffix('.pdf'), bbox_inches='tight')
    print(f"Plot saved to: {output_path}")
    plt.close()


def load_pooled_scaling_by_type(evals_dir: Path) -> pd.DataFrame:
    """Load raw KL + eval data and compute pooled per-type scaling scores.

    Returns DataFrame with columns matching the max-based scaling CSV:
        exploit_type, checkpoint, mean_neg_kl, exploit_rate,
        log_exploit_lower_bound, exploit_lower_bound, n_samples
    """
    from rh_indicators.trajectory import (
        load_kl_results,
        load_per_problem_results,
        compute_pooled_exploit_rate_scaling,
    )

    kl_dir = evals_dir.parent / "kl"
    kl_df = load_kl_results(kl_dir)
    eval_df = load_per_problem_results(evals_dir)

    if kl_df is None or eval_df is None:
        raise ValueError(f"Could not load KL ({kl_dir}) or eval ({evals_dir}) data")

    checkpoints = sorted(set(kl_df["checkpoint"].unique()) | set(eval_df["checkpoint"].unique()))
    exploit_types = sorted(kl_df["exploit_type"].dropna().unique())

    type_dfs = []
    for et in exploit_types:
        kl_sub = kl_df[kl_df["exploit_type"] == et]
        eval_sub = eval_df[eval_df["exploit_type"] == et]
        scaling = compute_pooled_exploit_rate_scaling(kl_sub, checkpoints, eval_df=eval_sub)
        if len(scaling) > 0:
            scaling["exploit_type"] = et
            type_dfs.append(scaling)

    return pd.concat(type_dfs, ignore_index=True)


def run_prediction_analysis(
    df: pd.DataFrame,
    cutoff_checkpoints: list[int],
    output_dir: Path | None = None,
    evals_dir: Path | None = None,
    input_csv: Path | None = None,
    max_over_prefills: bool = False,
) -> None:
    """Run logit-space trajectory prediction analysis.

    Args:
        df: Prepared DataFrame with logit_rate, log_checkpoint columns.
        cutoff_checkpoints: Checkpoints to use as extrapolation cutoffs.
        output_dir: Output directory. If None, creates timestamped run dir.
        evals_dir: Path to evals directory (for config logging).
        input_csv: Path to input CSV (for config logging).
        max_over_prefills: Whether max-over-prefills mode was used (for config logging).
    """
    def _run(out_dir: Path):
        all_metrics = []

        # Create per-exploit subfolder
        per_exploit_dir = out_dir / "per_exploit"
        per_exploit_dir.mkdir(exist_ok=True)

        # 1. AR(1) predictions
        print("\n=== AR(1) Predictions ===")
        ar1_results = ar1_prediction(df)
        if "error" not in ar1_results:
            m = ar1_results["metrics"]
            print(f"Aggregate: R²(logit)={m['r2_logit']:.3f}, R²(rate)={m['r2_rate']:.3f}, "
                  f"slope={m['ar1_slope']:.3f}, p={m['ar1_p_value']:.4f}")
            all_metrics.append(m)
            plot_ar1_fit(ar1_results, out_dir / "ar1_aggregate.png", "AR(1): Aggregate")
            ar1_results["predictions"].to_csv(out_dir / "ar1_aggregate_predictions.csv", index=False)

        # Per exploit type
        for etype in sorted(df["exploit_type"].unique()):
            results = ar1_prediction(df, exploit_type=etype)
            if "error" not in results:
                m = results["metrics"]
                print(f"  {etype}: R²(logit)={m['r2_logit']:.3f}, R²(rate)={m['r2_rate']:.3f}")
                all_metrics.append(m)
                plot_ar1_fit(results, per_exploit_dir / f"ar1_{etype}.png", f"AR(1): {etype}")

        # 2. Extrapolation predictions (predicting logit_exploit_lower_bound)
        print("\n=== Log-Step Extrapolation (logit_exploit_lower_bound) ===")
        for cutoff in cutoff_checkpoints:
            # Aggregate
            results = extrapolation_prediction(df, cutoff_checkpoint=cutoff)
            if "error" not in results:
                m = results["metrics"]
                print(f"Cutoff {cutoff} (aggregate): skill={m['skill_score']:.3f}, slope={m['slope']:.3f}")
                all_metrics.append(m)
                plot_extrapolation(df, cutoff, results, out_dir / f"extrapolation_cutoff{cutoff}.png")
                results["predictions"].to_csv(out_dir / f"extrapolation_cutoff{cutoff}_predictions.csv", index=False)

            # Per exploit type
            for etype in sorted(df["exploit_type"].unique()):
                results = extrapolation_prediction(df, cutoff_checkpoint=cutoff, exploit_type=etype)
                if "error" not in results:
                    m = results["metrics"]
                    print(f"  {etype}: skill={m['skill_score']:.3f}")
                    all_metrics.append(m)
                    plot_extrapolation(
                        df, cutoff, results,
                        per_exploit_dir / f"extrapolation_cutoff{cutoff}_{etype}.png",
                        exploit_type=etype
                    )

        # 3. Gaussian Process predictions (predicting logit_exploit_lower_bound)
        print("\n=== Gaussian Process (logit_exploit_lower_bound) ===")
        for cutoff in cutoff_checkpoints:
            # Aggregate
            results = gp_prediction(df, cutoff_checkpoint=cutoff)
            if "error" not in results:
                m = results["metrics"]
                print(f"Cutoff {cutoff} (aggregate): skill={m['skill_score']:.3f}")
                print(f"  Kernel: {m['kernel_params']}")
                all_metrics.append(m)
                plot_gp_prediction(results, out_dir / f"gp_cutoff{cutoff}.png", cutoff)
                results["predictions"].to_csv(out_dir / f"gp_cutoff{cutoff}_predictions.csv", index=False)
                results["full_predictions"].to_csv(out_dir / f"gp_cutoff{cutoff}_full.csv", index=False)

            # Per exploit type
            for etype in sorted(df["exploit_type"].unique()):
                results = gp_prediction(df, cutoff_checkpoint=cutoff, exploit_type=etype)
                if "error" not in results:
                    m = results["metrics"]
                    print(f"  {etype}: skill={m['skill_score']:.3f}")
                    all_metrics.append(m)
                    plot_gp_prediction(
                        results,
                        per_exploit_dir / f"gp_cutoff{cutoff}_{etype}.png",
                        cutoff,
                        exploit_type=etype
                    )

        # 4. Metric vs Rate as predictor (compare log_exploit_lower_bound, KL, and rate)
        print("\n=== Predictor Comparison (log_exploit_lower_bound vs KL vs Rate) ===")
        checkpoints = sorted(df["checkpoint"].unique())
        final_ckpt = checkpoints[-1]

        for pred_ckpt in checkpoints[:-1]:  # All but final
            # log_exploit_lower_bound predictor
            results_lb = kl_prediction(df, predictor_checkpoint=pred_ckpt, target_checkpoint=final_ckpt,
                                       predictor_col="log_exploit_lower_bound")
            # KL predictor
            results_kl = kl_prediction(df, predictor_checkpoint=pred_ckpt, target_checkpoint=final_ckpt,
                                       predictor_col="mean_neg_kl")

            if "error" not in results_lb and "error" not in results_kl:
                m_lb = results_lb["metrics_metric"]
                m_kl = results_kl["metrics_metric"]
                m_rate = results_lb["metrics_rate_baseline"]
                print(f"Checkpoint {pred_ckpt} → {final_ckpt}:")
                print(f"  log_exploit_lower_bound: R²={m_lb['r2_logit']:.3f}, p={m_lb['p_value']:.4f}")
                print(f"  mean_neg_kl:             R²={m_kl['r2_logit']:.3f}, p={m_kl['p_value']:.4f}")
                print(f"  rate:                    R²={m_rate['r2_logit']:.3f}, p={m_rate['p_value']:.4f}")
                all_metrics.append(m_lb)
                all_metrics.append(m_kl)
                all_metrics.append(m_rate)
                plot_metric_comparison(results_lb, out_dir / f"lb_vs_rate_ckpt{pred_ckpt}_to_{final_ckpt}.png")
                plot_metric_comparison(results_kl, out_dir / f"kl_vs_rate_ckpt{pred_ckpt}_to_{final_ckpt}.png")
                results_lb["predictions"].to_csv(
                    out_dir / f"lb_vs_rate_ckpt{pred_ckpt}_to_{final_ckpt}_predictions.csv",
                    index=False
                )

        # 5. Predictor comparison with leave-one-out CV over exploit types
        print("\n=== Predictor Comparison LOO (log_exploit_lower_bound vs KL vs Rate) ===")
        for pred_ckpt in checkpoints[:-1]:
            results_lb = kl_prediction_loo(df, predictor_checkpoint=pred_ckpt, target_checkpoint=final_ckpt,
                                           predictor_col="log_exploit_lower_bound")
            results_kl = kl_prediction_loo(df, predictor_checkpoint=pred_ckpt, target_checkpoint=final_ckpt,
                                           predictor_col="mean_neg_kl")

            if "error" not in results_lb and "error" not in results_kl:
                m_lb = results_lb["metrics_metric"]
                m_kl = results_kl["metrics_metric"]
                m_rate = results_lb["metrics_rate_baseline"]
                print(f"Checkpoint {pred_ckpt} → {final_ckpt}:")
                print(f"  log_exploit_lower_bound (LOO): R²={m_lb['r2_logit']:.3f}")
                print(f"  mean_neg_kl (LOO):             R²={m_kl['r2_logit']:.3f}")
                print(f"  rate (LOO):                    R²={m_rate['r2_logit']:.3f}")
                all_metrics.append(m_lb)
                all_metrics.append(m_kl)
                all_metrics.append(m_rate)
                results_lb["predictions"].to_csv(
                    out_dir / f"lb_loo_ckpt{pred_ckpt}_to_{final_ckpt}_predictions.csv",
                    index=False
                )

        # Save all metrics
        metrics_df = pd.DataFrame(all_metrics)
        metrics_df.to_csv(out_dir / "all_metrics.csv", index=False)
        print(f"\nMetrics saved to {out_dir / 'all_metrics.csv'}")

        print(f"\n=== Summary ===")
        print(f"Output directory: {out_dir}")

    # Run with or without experiment context
    if output_dir:
        output_dir.mkdir(parents=True, exist_ok=True)
        _run(output_dir)
    else:
        config_args = {
            "cutoff_checkpoints": cutoff_checkpoints,
            "aggregation": "max-over-prefills" if max_over_prefills else "pooled",
        }
        if input_csv:
            config_args["input"] = str(input_csv)
        if evals_dir:
            config_args["evals_dir"] = str(evals_dir)
        with run_context(
            base_dir=Path("results/trajectory_prediction"),
            run_prefix="logit_trajectory_prediction",
            config_args=config_args,
        ) as out_dir:
            _run(out_dir)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input", "-i",
        type=Path,
        default=None,
        help="Path to exploit_rate_scaling_by_type.csv (only needed with --max-over-prefills)",
    )
    parser.add_argument(
        "--evals-dir",
        type=Path,
        default=None,
        help="Path to evals directory (for pooled mode, default)",
    )
    parser.add_argument(
        "--output-dir", "-o",
        type=Path,
        default=None,
        help="Output directory (default: creates timestamped run dir)",
    )
    parser.add_argument(
        "--cutoff-checkpoints",
        type=int,
        nargs="+",
        default=[6, 15, 25, 44],
        help="Checkpoints to use as extrapolation cutoffs",
    )
    parser.add_argument(
        "--max-over-prefills",
        action="store_true",
        help="Use max-over-prefills aggregation (legacy). Default is pooled (avg).",
    )
    args = parser.parse_args()

    if args.max_over_prefills:
        if args.input is None:
            parser.error("--input is required with --max-over-prefills")
        print(f"Aggregation mode: max-over-prefills")
        print(f"Loading data from {args.input}")
        df = pd.read_csv(args.input)
    else:
        if args.evals_dir is None and args.input is not None:
            # Fall back to CSV if provided without --max-over-prefills
            print(f"Loading data from {args.input} (CSV provided, using as-is)")
            df = pd.read_csv(args.input)
        elif args.evals_dir is not None:
            print(f"Aggregation mode: pooled (avg)")
            print(f"Loading data from {args.evals_dir}")
            df = load_pooled_scaling_by_type(args.evals_dir)
        else:
            parser.error("Either --evals-dir or --input is required")

    df = prepare_data(df)

    print(f"Loaded {len(df)} rows")
    print(f"Checkpoints: {sorted(df['checkpoint'].unique())}")
    print(f"Exploit types: {sorted(df['exploit_type'].unique())}")

    run_prediction_analysis(
        df, args.cutoff_checkpoints,
        output_dir=args.output_dir,
        evals_dir=args.evals_dir,
        input_csv=args.input,
        max_over_prefills=args.max_over_prefills,
    )


if __name__ == "__main__":
    main()
