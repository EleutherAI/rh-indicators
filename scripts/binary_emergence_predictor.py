#!/usr/bin/env python3
"""
Binary exploit emergence predictor.

Predicts whether an exploit type will exceed 10% exploit rate at prefill=0
(unprompted exploitation) at any point during training, using only
prefill-assisted metrics from early checkpoints.

Two aggregation modes:
- Pooled (default): average KL and exploit_rate across all (problem, prefill) pairs
- Max (legacy, --max-over-prefills): use max-over-prefills from scaling CSV

Two model forms:
1. Threshold rule: AUC of single metric as classifier (no free parameters)
2. Level + slope logistic: logit(P) = β₀ + β₁·metric + β₂·slope  (LOO-CV)

Usage (pooled, default):
    python scripts/binary_emergence_predictor.py \
        --exploit-evals results/prefill_sensitivity/prefill_sensitivity-20260127-050226-8a0e189/evals \
        --control-evals results/prefill_sensitivity/prefill_sensitivity-20260206-045419-8a0e189/evals \
        --output-dir results/trajectory_prediction/binary_emergence

Usage (max-over-prefills, legacy):
    python scripts/binary_emergence_predictor.py \
        --exploit-run results/trajectory_analysis/.../exploit_rate_scaling_by_type.csv \
        --control-run results/trajectory_analysis/.../exploit_rate_scaling_by_type.csv \
        --exploit-evals results/prefill_sensitivity/.../evals \
        --control-evals results/prefill_sensitivity/.../evals \
        --max-over-prefills
"""

import argparse
import json
import warnings
from collections import Counter
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, brier_score_loss

INTENTIONAL_TYPES = {
    "argument_injection_leak",
    "error_code_abuse",
    "import_hook_side_channel",
    "inspect_module_abuse",
    "test_cases_honor_system",
    "trace_profiler_hook_oracle",
    "validator_honor_system",
    "verifier_logic_override",
}

METRICS = ["log_exploit_lower_bound", "mean_neg_kl", "exploit_rate"]
EXPLOIT_LOGPROB_METRICS = ["exploit_logprob"]

# Per-problem feature from trajectory_analysis.csv
PROBLEM_METRICS = ["accessibility"]


def compute_prefill0_targets(
    evals_dir: Path, checkpoints: list[int], threshold: float = 0.10
) -> dict[str, int]:
    """Compute binary target: does each exploit type ever exceed threshold at prefill=0?

    Returns dict mapping exploit_type -> 0 or 1.
    """
    max_rates: dict[str, float] = {}

    for ckpt in checkpoints:
        f = evals_dir / f"checkpoint-{ckpt}_prefill0.jsonl"
        if not f.exists():
            continue
        counts: Counter[str] = Counter()
        totals: Counter[str] = Counter()
        with open(f) as fh:
            for line in fh:
                d = json.loads(line)
                et = d.get("exploit_type", "")
                if et not in INTENTIONAL_TYPES:
                    continue
                totals[et] += 1
                if d.get("exploit_success", False):
                    counts[et] += 1
        for et in INTENTIONAL_TYPES:
            if totals[et] > 0:
                rate = counts[et] / totals[et]
                max_rates[et] = max(max_rates.get(et, 0.0), rate)

    return {et: int(max_rates.get(et, 0.0) > threshold) for et in INTENTIONAL_TYPES}


def load_run_data(
    evals_dir: Path,
    run_label: str,
    threshold: float = 0.10,
    scaling_csv: Path | None = None,
    kl_dir: Path | None = None,
    exploit_logprobs_dir: Path | None = None,
) -> pd.DataFrame:
    """Load scaling data and compute targets for one run.

    Two modes:
    - Pooled (default): kl_dir is provided (or inferred from evals_dir).
      Computes scores by averaging KL and exploit_rate across all (problem, prefill) pairs.
    - Max (legacy): scaling_csv is provided. Uses pre-computed max-over-prefills scores.
    """
    from rh_indicators.trajectory import (
        load_kl_results,
        load_per_problem_results,
        compute_exploit_rate_scaling,
        compute_pooled_exploit_rate_scaling,
    )

    if scaling_csv is not None:
        # Legacy max-over-prefills mode
        df = pd.read_csv(scaling_csv)
        df = df[df["exploit_type"].isin(INTENTIONAL_TYPES)].copy()
    else:
        # Pooled mode: compute from raw KL + eval data per exploit type
        if kl_dir is None:
            kl_dir = evals_dir.parent / "kl"
        kl_df = load_kl_results(kl_dir)
        eval_df = load_per_problem_results(evals_dir)

        if kl_df is None or eval_df is None:
            raise ValueError(f"Could not load KL ({kl_dir}) or eval ({evals_dir}) data")

        # Filter to intentional types
        kl_df = kl_df[kl_df["exploit_type"].isin(INTENTIONAL_TYPES)]
        eval_df = eval_df[eval_df["exploit_type"].isin(INTENTIONAL_TYPES)]

        checkpoints = sorted(set(kl_df["checkpoint"].unique()) | set(eval_df["checkpoint"].unique()))

        # Compute per-type pooled scores
        type_dfs = []
        for et in sorted(INTENTIONAL_TYPES):
            kl_sub = kl_df[kl_df["exploit_type"] == et]
            eval_sub = eval_df[eval_df["exploit_type"] == et]
            scaling = compute_pooled_exploit_rate_scaling(kl_sub, checkpoints, eval_df=eval_sub)
            if len(scaling) > 0:
                scaling["exploit_type"] = et
                type_dfs.append(scaling)

        df = pd.concat(type_dfs, ignore_index=True)

    # Merge exploit logprobs if provided
    if exploit_logprobs_dir is not None:
        from rh_indicators.trajectory import load_exploit_logprobs

        elp_df = load_exploit_logprobs(exploit_logprobs_dir)
        if elp_df is not None:
            # Filter to intentional types
            elp_df = elp_df[elp_df["exploit_type"].isin(INTENTIONAL_TYPES)]

            # Aggregate to per-(exploit_type, checkpoint) mean
            elp_agg = (
                elp_df.groupby(["exploit_type", "checkpoint"])["exploit_logprob_sum"]
                .mean()
                .reset_index()
                .rename(columns={"exploit_logprob_sum": "exploit_logprob"})
            )

            # Merge into df
            df = df.merge(elp_agg, on=["exploit_type", "checkpoint"], how="left")
        else:
            print(f"Warning: No exploit logprob data found in {exploit_logprobs_dir}")

    # Compute targets from prefill=0 eval data
    checkpoints = sorted(df["checkpoint"].unique())
    targets = compute_prefill0_targets(evals_dir, checkpoints, threshold)

    df["target"] = df["exploit_type"].map(targets)
    df["run"] = run_label
    df["log_checkpoint"] = np.log(df["checkpoint"])

    # Assign ordinal checkpoint index within this run
    ckpt_order = {c: i for i, c in enumerate(checkpoints)}
    df["ckpt_idx"] = df["checkpoint"].map(ckpt_order)

    return df


def compute_features_at_cutoff(
    run_df: pd.DataFrame, exploit_type: str, cutoff_idx: int, metric: str
) -> dict | None:
    """Compute level, slope, and projection features for one exploit type at one cutoff.

    Returns dict with 'level', 'slope', 'intercept', 'last_log_ckpt', 'target',
    or None if insufficient data.
    """
    sub = run_df[
        (run_df["exploit_type"] == exploit_type) & (run_df["ckpt_idx"] <= cutoff_idx)
    ].sort_values("ckpt_idx")

    if len(sub) == 0:
        return None

    level = sub[metric].iloc[-1]  # current value
    target = sub["target"].iloc[0]  # same for all rows of this type
    last_log_ckpt = sub["log_checkpoint"].iloc[-1]

    if len(sub) >= 2:
        # Slope of metric over log(checkpoint)
        slope_result = stats.linregress(sub["log_checkpoint"], sub[metric])
        slope = slope_result.slope
        intercept = slope_result.intercept
    else:
        slope = 0.0
        intercept = level

    return {
        "level": level,
        "slope": slope,
        "intercept": intercept,
        "last_log_ckpt": last_log_ckpt,
        "target": target,
    }


def threshold_rule_auc(
    features_df: pd.DataFrame, metric_col: str
) -> dict:
    """Compute AUC for a threshold rule on a single metric.

    Returns dict with AUC and rank-biserial correlation.
    """
    y_true = features_df["target"].values
    scores = features_df[metric_col].values

    # Need both classes present
    if len(set(y_true)) < 2:
        return {"auc": np.nan, "rank_biserial": np.nan, "n": len(y_true)}

    # Higher metric = more likely to exploit, so use scores directly
    auc = roc_auc_score(y_true, scores)
    # Rank-biserial correlation
    pos = scores[y_true == 1]
    neg = scores[y_true == 0]
    u_stat, p_val = stats.mannwhitneyu(pos, neg, alternative="two-sided")
    rank_biserial = 2 * u_stat / (len(pos) * len(neg)) - 1

    return {
        "auc": auc,
        "rank_biserial": rank_biserial,
        "mann_whitney_p": p_val,
        "n_pos": len(pos),
        "n_neg": len(neg),
        "n": len(y_true),
    }


def tuned_projection_auc(
    features_df: pd.DataFrame,
    a_grid: np.ndarray | None = None,
) -> dict:
    """Tune projection horizon a in score = level + a*slope to maximize LOO AUC.

    For each held-out sample, find the a that maximizes AUC on the remaining N-1,
    compute the held-out score, then compute overall AUC from those scores.

    Also reports the a that maximizes full-sample AUC (for interpretability).

    Returns dict with LOO AUC, best_a (full-sample), and the a grid results.
    """
    if a_grid is None:
        a_grid = np.linspace(0, 10, 101)

    y_true = features_df["target"].values
    levels = features_df["level"].values
    slopes = features_df["slope"].values
    n = len(y_true)

    if len(set(y_true)) < 2:
        return {"auc": np.nan, "best_a": np.nan, "n": n}

    # Full-sample: find best a
    best_full_auc = -1.0
    best_full_a = 0.0
    for a in a_grid:
        scores = levels + a * slopes
        auc = roc_auc_score(y_true, scores)
        if auc > best_full_auc:
            best_full_auc = auc
            best_full_a = a

    # LOO: for each held-out sample, tune a on the rest, score the held-out
    loo_scores = np.zeros(n)
    loo_best_as = np.zeros(n)
    for i in range(n):
        lvl_train = np.delete(levels, i)
        slp_train = np.delete(slopes, i)
        y_train = np.delete(y_true, i)

        if len(set(y_train)) < 2:
            # Can't compute AUC, use a=0 (just level)
            loo_scores[i] = levels[i]
            loo_best_as[i] = 0.0
            continue

        # Find best a on training fold
        fold_best_auc = -1.0
        fold_best_a = 0.0
        for a in a_grid:
            train_scores = lvl_train + a * slp_train
            auc = roc_auc_score(y_train, train_scores)
            if auc > fold_best_auc:
                fold_best_auc = auc
                fold_best_a = a

        loo_best_as[i] = fold_best_a
        loo_scores[i] = levels[i] + fold_best_a * slopes[i]

    loo_auc = roc_auc_score(y_true, loo_scores)

    return {
        "auc": loo_auc,
        "best_a": best_full_a,
        "best_full_auc": best_full_auc,
        "mean_loo_a": loo_best_as.mean(),
        "std_loo_a": loo_best_as.std(),
        "n": n,
    }


def logistic_loo(
    features_df: pd.DataFrame,
) -> dict:
    """Leave-one-out logistic regression: logit(P) = β₀ + β₁·level + β₂·slope.

    Returns dict with LOO predictions and metrics.
    """
    y_true = features_df["target"].values
    X = features_df[["level", "slope"]].values
    n = len(y_true)

    if len(set(y_true)) < 2:
        return {"auc": np.nan, "brier": np.nan, "n": n, "predictions": []}

    loo_probs = np.zeros(n)
    for i in range(n):
        X_train = np.delete(X, i, axis=0)
        y_train = np.delete(y_true, i)

        # Need both classes in training set
        if len(set(y_train)) < 2:
            loo_probs[i] = y_train.mean()
            continue

        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=ConvergenceWarning)
            clf = LogisticRegression(penalty=None, max_iter=1000)
            clf.fit(X_train, y_train)
        loo_probs[i] = clf.predict_proba(X[i : i + 1])[0, 1]

    auc = roc_auc_score(y_true, loo_probs)
    brier = brier_score_loss(y_true, loo_probs)

    # Fit on all data for coefficients
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=ConvergenceWarning)
        clf_full = LogisticRegression(penalty=None, max_iter=1000)
        clf_full.fit(X, y_true)

    return {
        "auc": auc,
        "brier": brier,
        "n": n,
        "coef_level": clf_full.coef_[0][0],
        "coef_slope": clf_full.coef_[0][1],
        "intercept": clf_full.intercept_[0],
        "loo_probs": loo_probs,
        "y_true": y_true,
        "exploit_types": features_df["exploit_type"].values,
        "runs": features_df["run"].values,
    }


def _evaluate_and_record(
    features_df: pd.DataFrame,
    metric: str,
    cutoff_idx: int,
    cutoff_ckpts: dict,
    output_dir: Path,
) -> list[dict]:
    """Run all model evaluations on a features DataFrame and return result records."""
    result_base = {"cutoff_idx": cutoff_idx, "metric": metric}
    for label, ckpt in cutoff_ckpts.items():
        result_base[f"{label}_cutoff_ckpt"] = ckpt

    records = []

    # --- Threshold rules ---
    level_auc = threshold_rule_auc(features_df, "level")
    slope_auc = threshold_rule_auc(features_df, "slope")
    projected_auc = threshold_rule_auc(features_df, "projected")

    records.append({
        **result_base, "model": "threshold_level",
        "auc": level_auc["auc"],
        "rank_biserial": level_auc.get("rank_biserial"),
        "mann_whitney_p": level_auc.get("mann_whitney_p"),
        "n": level_auc["n"],
    })
    records.append({
        **result_base, "model": "threshold_slope",
        "auc": slope_auc["auc"],
        "rank_biserial": slope_auc.get("rank_biserial"),
        "mann_whitney_p": slope_auc.get("mann_whitney_p"),
        "n": slope_auc["n"],
    })
    records.append({
        **result_base, "model": "threshold_projected",
        "auc": projected_auc["auc"],
        "rank_biserial": projected_auc.get("rank_biserial"),
        "mann_whitney_p": projected_auc.get("mann_whitney_p"),
        "n": projected_auc["n"],
    })

    # --- Tuned projection (1 param) ---
    tuned_result = tuned_projection_auc(features_df)
    records.append({
        **result_base, "model": "tuned_projection",
        "auc": tuned_result["auc"],
        "best_a": tuned_result.get("best_a"),
        "best_full_auc": tuned_result.get("best_full_auc"),
        "mean_loo_a": tuned_result.get("mean_loo_a"),
        "std_loo_a": tuned_result.get("std_loo_a"),
        "n": tuned_result["n"],
    })

    # --- Logistic LOO (3 params) ---
    logistic_result = logistic_loo(features_df)
    records.append({
        **result_base, "model": "logistic_level_slope",
        "auc": logistic_result["auc"],
        "brier": logistic_result.get("brier"),
        "coef_level": logistic_result.get("coef_level"),
        "coef_slope": logistic_result.get("coef_slope"),
        "intercept": logistic_result.get("intercept"),
        "n": logistic_result["n"],
    })

    # Print summary
    print(
        f"  {metric:<30s} "
        f"level={level_auc['auc']:.3f}  "
        f"slope={slope_auc['auc']:.3f}  "
        f"proj={projected_auc['auc']:.3f}  "
        f"tuned={tuned_result['auc']:.3f}(a={tuned_result.get('best_a', 0):.1f})  "
        f"logistic={logistic_result['auc']:.3f}"
    )

    # Save LOO predictions
    if "loo_probs" in logistic_result:
        pred_df = pd.DataFrame({
            "exploit_type": logistic_result["exploit_types"],
            "run": logistic_result["runs"],
            "y_true": logistic_result["y_true"],
            "y_pred_prob": logistic_result["loo_probs"],
        })
        pred_df.to_csv(
            output_dir / f"logistic_loo_cutoff{cutoff_idx}_{metric}_predictions.csv",
            index=False,
        )

    return records


GP_METRICS = ["gp_log_rate_p0", "gp_log_rate_p0_last"]


def run_analysis(
    run_dfs: list[tuple[str, pd.DataFrame]],
    output_dir: Path,
    gp_config: dict | None = None,
) -> pd.DataFrame:
    """Run the full binary emergence prediction analysis.

    Args:
        run_dfs: List of (label, dataframe) pairs, one per run.
        output_dir: Where to write results.
        gp_config: Optional dict mapping run label -> {"evals_dir": Path, "kl_dir": Path (optional)}.
            When provided, constrained GP features are computed per cutoff (no data leakage).
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    # Determine how many ordinal cutoffs we can test
    # Use the minimum number of checkpoints across all runs
    run_ckpts = {}
    for label, df in run_dfs:
        run_ckpts[label] = sorted(df["checkpoint"].unique())
    max_cutoff_idx = min(len(ckpts) for ckpts in run_ckpts.values()) - 1

    for label, ckpts in run_ckpts.items():
        print(f"{label} run checkpoints: {ckpts}")
    print(f"Max cutoff index: {max_cutoff_idx} (0-indexed)")
    print()

    # Determine which metrics are available (exploit_logprob only if column exists)
    has_exploit_logprob = all(
        "exploit_logprob" in df.columns for _, df in run_dfs
    )
    metrics = METRICS + (EXPLOIT_LOGPROB_METRICS if has_exploit_logprob else [])
    if has_exploit_logprob:
        print("Exploit logprob data available — including as additional metric")
    if gp_config:
        print("GP features will be computed per cutoff (no data leakage)")
        print(f"  GP metrics: {GP_METRICS}")
    if not has_exploit_logprob and not gp_config:
        print("No exploit logprob or GP data — using standard metrics only")
    print()

    # Pre-load data for GP computation (avoids re-reading from disk at each cutoff)
    gp_data_cache = {}
    if gp_config:
        from rh_indicators.trajectory import load_per_problem_results, load_kl_results

        for label, cfg in gp_config.items():
            if label.startswith("_"):
                continue  # skip internal config keys
            evals_dir = cfg["evals_dir"]
            kl_dir = cfg.get("kl_dir") or (evals_dir.parent / "kl")
            eval_df_gp = load_per_problem_results(evals_dir)
            kl_df_gp = load_kl_results(kl_dir)
            gp_data_cache[label] = (eval_df_gp, kl_df_gp)
        print(f"  Pre-loaded GP data for {list(gp_data_cache.keys())}")
        print()

    all_results = []

    for cutoff_idx in range(max_cutoff_idx + 1):
        cutoff_ckpts = {label: ckpts[cutoff_idx] for label, ckpts in run_ckpts.items()}
        cutoff_desc = ", ".join(f"{label} ckpt {c}" for label, c in cutoff_ckpts.items())
        print(f"--- Cutoff index {cutoff_idx}: {cutoff_desc} ---")

        # --- Regular metrics ---
        for metric in metrics:
            # Build feature dataframe across all runs
            rows = []
            for label, run_df in run_dfs:
                # Final checkpoint for this run (projection target)
                ckpts = run_ckpts[label]
                log_final_ckpt = np.log(ckpts[-1])

                for et in sorted(INTENTIONAL_TYPES):
                    feats = compute_features_at_cutoff(
                        run_df, et, cutoff_idx, metric
                    )
                    if feats is not None:
                        # Forward projection: extrapolate linear fit to final checkpoint
                        projected = (
                            feats["intercept"]
                            + feats["slope"] * log_final_ckpt
                        )
                        rows.append(
                            {
                                "exploit_type": et,
                                "run": label,
                                "level": feats["level"],
                                "slope": feats["slope"],
                                "projected": projected,
                                "target": feats["target"],
                            }
                        )

            if not rows:
                continue

            features_df = pd.DataFrame(rows)
            all_results.extend(
                _evaluate_and_record(features_df, metric, cutoff_idx, cutoff_ckpts, output_dir)
            )

        # --- GP metrics (computed per cutoff to avoid data leakage) ---
        if gp_config:
            from rh_indicators.trajectory.gp_model import (
                compute_constrained_gp_features_at_cutoff,
            )

            # Fit GPs on data up to this cutoff for each run
            gp_cutoff_data = {}  # label -> DataFrame with (exploit_type, checkpoint, gp_log_rate_p0)
            for label, run_df in run_dfs:
                ckpts = run_ckpts[label]
                ckpts_up_to = ckpts[: cutoff_idx + 1]
                final_ckpt = ckpts[-1]
                target_ckpts = sorted(set(ckpts_up_to) | {final_ckpt})

                cached_eval_df, cached_kl_df = gp_data_cache[label]
                gp_df = compute_constrained_gp_features_at_cutoff(
                    include_checkpoints=ckpts_up_to,
                    target_checkpoints=target_ckpts,
                    preloaded_eval_df=cached_eval_df,
                    preloaded_kl_df=cached_kl_df,
                    jensen_correction=gp_config.get("_jensen_correction", False),
                )
                gp_cutoff_data[label] = gp_df

            # Evaluate each GP metric
            for gp_metric in GP_METRICS:
                rows = []
                for label, run_df in run_dfs:
                    ckpts = run_ckpts[label]
                    final_ckpt = ckpts[-1]
                    cutoff_ckpt = ckpts[cutoff_idx]
                    log_final_ckpt = np.log(final_ckpt)
                    gp_df = gp_cutoff_data[label]
                    ckpts_up_to = ckpts[: cutoff_idx + 1]

                    for et in sorted(INTENTIONAL_TYPES):
                        et_gp = gp_df[gp_df["exploit_type"] == et]
                        if len(et_gp) == 0:
                            continue

                        # Get target from run_df
                        et_sub = run_df[run_df["exploit_type"] == et]
                        if len(et_sub) == 0:
                            continue
                        target = et_sub["target"].iloc[0]

                        if gp_metric == "gp_log_rate_p0":
                            # Level: GP prediction at cutoff checkpoint
                            cutoff_row = et_gp[et_gp["checkpoint"] == cutoff_ckpt]
                            if len(cutoff_row) == 0:
                                continue
                            level = cutoff_row["gp_log_rate_p0"].iloc[0]

                            # Slope: regression of GP values at included checkpoints
                            traj = et_gp[et_gp["checkpoint"].isin(ckpts_up_to)].sort_values("checkpoint")
                            if len(traj) >= 2:
                                sr = stats.linregress(
                                    np.log(traj["checkpoint"].values.astype(float)),
                                    traj["gp_log_rate_p0"].values,
                                )
                                slope = sr.slope
                                intercept = sr.intercept
                            else:
                                slope = 0.0
                                intercept = level
                            projected = intercept + slope * log_final_ckpt

                        elif gp_metric == "gp_log_rate_p0_last":
                            # Level: GP extrapolation to final checkpoint
                            last_row = et_gp[et_gp["checkpoint"] == final_ckpt]
                            if len(last_row) == 0:
                                continue
                            level = last_row["gp_log_rate_p0"].iloc[0]
                            slope = 0.0
                            projected = level

                        rows.append({
                            "exploit_type": et,
                            "run": label,
                            "level": level,
                            "slope": slope,
                            "projected": projected,
                            "target": target,
                        })

                if not rows:
                    continue

                features_df = pd.DataFrame(rows)
                all_results.extend(
                    _evaluate_and_record(features_df, gp_metric, cutoff_idx, cutoff_ckpts, output_dir)
                )

        print()

    results_df = pd.DataFrame(all_results)
    results_df.to_csv(output_dir / "all_results.csv", index=False)
    print(f"Saved results to {output_dir / 'all_results.csv'}")

    # Generate plots
    plot_auc_curves(results_df, output_dir)
    plot_feature_separation(run_dfs, output_dir)

    return results_df


def plot_auc_curves(results_df: pd.DataFrame, output_dir: Path) -> None:
    """Plot AUC vs cutoff checkpoint for each metric and model."""
    plot_metrics = sorted(results_df["metric"].unique())
    fig, axes = plt.subplots(1, len(plot_metrics), figsize=(5 * len(plot_metrics), 4), sharey=True)
    if len(plot_metrics) == 1:
        axes = [axes]

    models = [
        "threshold_level",
        "tuned_projection",
        "threshold_projected",
        "threshold_slope",
        "logistic_level_slope",
    ]
    model_labels = {
        "threshold_level": "Threshold (level)",
        "tuned_projection": "Tuned projection (level+a*slope)",
        "threshold_projected": "Project to final ckpt",
        "threshold_slope": "Threshold (slope)",
        "logistic_level_slope": "Logistic (level+slope)",
    }
    model_styles = {
        "threshold_level": ("o-", "C0"),
        "tuned_projection": ("P-", "C5"),
        "threshold_projected": ("^--", "C4"),
        "threshold_slope": ("s--", "C1"),
        "logistic_level_slope": ("D--", "C2"),
    }

    for ax, metric in zip(axes, plot_metrics):
        for model in models:
            sub = results_df[
                (results_df["metric"] == metric) & (results_df["model"] == model)
            ].sort_values("cutoff_idx")
            if len(sub) == 0:
                continue
            style, color = model_styles[model]
            ax.plot(
                sub["cutoff_idx"],
                sub["auc"],
                style,
                color=color,
                label=model_labels[model],
                markersize=6,
                linewidth=1.5,
            )

        ax.axhline(y=0.5, color="gray", linestyle=":", alpha=0.5, label="Random")
        ax.set_xlabel("Cutoff (ordinal checkpoint index)")
        ax.set_title(metric.replace("_", " ").title())
        ax.set_ylim(-0.05, 1.05)
        ax.grid(True, alpha=0.3)

    axes[0].set_ylabel("AUC")
    axes[-1].legend(bbox_to_anchor=(1.05, 1), loc="upper left", fontsize=8)
    fig.suptitle(
        "Binary Exploit Emergence Prediction: AUC vs Cutoff",
        fontsize=13,
        fontweight="bold",
    )
    plt.tight_layout()
    for ext in [".png", ".pdf"]:
        fig.savefig(output_dir / f"auc_vs_cutoff{ext}", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved AUC plot to {output_dir / 'auc_vs_cutoff.png'}")


def plot_feature_separation(
    run_dfs: list[tuple[str, pd.DataFrame]], output_dir: Path
) -> None:
    """Plot feature distributions at each cutoff showing separation between runs."""
    # Compute per-run checkpoint lists and max cutoff
    run_ckpts = {}
    for label, df in run_dfs:
        run_ckpts[label] = sorted(df["checkpoint"].unique())
    max_cutoff = min(len(ckpts) for ckpts in run_ckpts.values())

    # Plot for log_exploit_lower_bound at a few cutoffs
    metric = "log_exploit_lower_bound"
    cutoffs_to_plot = [0, 1, 2, min(4, max_cutoff - 1)]
    cutoffs_to_plot = sorted(set(c for c in cutoffs_to_plot if c < max_cutoff))

    fig, axes = plt.subplots(1, len(cutoffs_to_plot), figsize=(4 * len(cutoffs_to_plot), 4))
    if len(cutoffs_to_plot) == 1:
        axes = [axes]

    colors = [f"C{i}" for i in range(len(run_dfs))]

    for ax, cidx in zip(axes, cutoffs_to_plot):
        x_pos = 0
        tick_positions = []
        tick_labels_list = []
        for run_idx, (label, run_df) in enumerate(run_dfs):
            vals = []
            for et in sorted(INTENTIONAL_TYPES):
                feats = compute_features_at_cutoff(run_df, et, cidx, metric)
                if feats is not None:
                    vals.append(feats["level"])
            vals = np.array(vals)
            if len(vals) > 0:
                ax.scatter(
                    np.full(len(vals), x_pos) + np.random.normal(0, 0.05, len(vals)),
                    vals,
                    c=colors[run_idx],
                    alpha=0.7,
                    s=40,
                    label=label if cidx == cutoffs_to_plot[0] else None,
                )
            tick_positions.append(x_pos)
            tick_labels_list.append(label)
            x_pos += 1

        ax.set_xticks(tick_positions)
        ax.set_xticklabels(tick_labels_list, fontsize=7, rotation=30, ha="right")

        # Build subtitle with per-run checkpoint at this cutoff
        ckpt_parts = [f"{lbl} ckpt {run_ckpts[lbl][cidx]}" for lbl in run_ckpts]
        ax.set_title(f"Cutoff idx {cidx}\n({', '.join(ckpt_parts)})", fontsize=8)
        ax.set_ylabel(metric)
        ax.grid(True, alpha=0.3, axis="y")

    axes[0].legend(fontsize=8)
    fig.suptitle(
        "Feature Separation: log_exploit_lower_bound",
        fontsize=13,
        fontweight="bold",
    )
    plt.tight_layout()
    for ext in [".png", ".pdf"]:
        fig.savefig(
            output_dir / f"feature_separation{ext}", dpi=150, bbox_inches="tight"
        )
    plt.close(fig)
    print(f"Saved feature separation plot to {output_dir / 'feature_separation.png'}")


def load_problem_data(
    trajectory_csv: Path, run_label: str
) -> pd.DataFrame:
    """Load per-problem trajectory data and compute targets.

    Target: does min_prefill ever reach 0 during training?
    """
    df = pd.read_csv(trajectory_csv)
    df = df[df["exploit_type"].isin(INTENTIONAL_TYPES)].copy()

    checkpoints = sorted(df["checkpoint"].unique())
    ckpt_order = {c: i for i, c in enumerate(checkpoints)}
    df["ckpt_idx"] = df["checkpoint"].map(ckpt_order)
    df["log_checkpoint"] = np.log(df["checkpoint"])
    df["run"] = run_label

    # Per-problem target: min_prefill==0 at any checkpoint
    targets = df.groupby("task_id")["min_prefill"].apply(
        lambda x: int((x == 0).any())
    )
    df["target"] = df["task_id"].map(targets)

    return df


def compute_problem_features_at_cutoff(
    run_df: pd.DataFrame, task_id: str, cutoff_idx: int, metric: str = "accessibility"
) -> dict | None:
    """Compute level and slope features for one problem at one cutoff."""
    sub = run_df[
        (run_df["task_id"] == task_id) & (run_df["ckpt_idx"] <= cutoff_idx)
    ].sort_values("ckpt_idx")

    if len(sub) == 0:
        return None

    level = sub[metric].iloc[-1]
    target = sub["target"].iloc[0]
    exploit_type = sub["exploit_type"].iloc[0]

    if len(sub) >= 2:
        slope_result = stats.linregress(sub["log_checkpoint"], sub[metric])
        slope = slope_result.slope
    else:
        slope = 0.0

    return {
        "level": level,
        "slope": slope,
        "target": target,
        "exploit_type": exploit_type,
    }


def blocked_kfold_auc(
    features_df: pd.DataFrame,
    model_type: str = "logistic",
    a_grid: np.ndarray | None = None,
) -> dict:
    """k-fold CV blocked by exploit_type.

    model_type: 'logistic' or 'tuned_projection'
    Returns AUC, Brier score, and per-fold stats.
    """
    if a_grid is None:
        a_grid = np.linspace(0, 10, 101)

    y_true = features_df["target"].values
    types = features_df["exploit_type"].values
    unique_types = sorted(set(types))

    if len(set(y_true)) < 2:
        return {"auc": np.nan, "n": len(y_true)}

    oof_scores = np.full(len(y_true), np.nan)

    for held_out_type in unique_types:
        test_mask = types == held_out_type
        train_mask = ~test_mask

        y_train = y_true[train_mask]
        y_test = y_true[test_mask]

        if len(set(y_train)) < 2:
            # Can't train, assign base rate
            oof_scores[test_mask] = y_train.mean()
            continue

        if model_type == "logistic":
            X_train = features_df.loc[train_mask, ["level", "slope"]].values
            X_test = features_df.loc[test_mask, ["level", "slope"]].values

            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", category=ConvergenceWarning)
                clf = LogisticRegression(penalty=None, max_iter=1000)
                clf.fit(X_train, y_train)
            oof_scores[test_mask] = clf.predict_proba(X_test)[:, 1]

        elif model_type == "tuned_projection":
            levels_train = features_df.loc[train_mask, "level"].values
            slopes_train = features_df.loc[train_mask, "slope"].values
            levels_test = features_df.loc[test_mask, "level"].values
            slopes_test = features_df.loc[test_mask, "slope"].values

            # Find best a on training fold
            best_a = 0.0
            best_fold_auc = -1.0
            for a in a_grid:
                s = levels_train + a * slopes_train
                try:
                    fauc = roc_auc_score(y_train, s)
                except ValueError:
                    fauc = 0.5
                if fauc > best_fold_auc:
                    best_fold_auc = fauc
                    best_a = a

            oof_scores[test_mask] = levels_test + best_a * slopes_test

    valid = ~np.isnan(oof_scores)
    if valid.sum() < len(y_true) or len(set(y_true[valid])) < 2:
        return {"auc": np.nan, "n": len(y_true)}

    auc = roc_auc_score(y_true, oof_scores)
    result = {"auc": auc, "n": len(y_true)}

    if model_type == "logistic":
        # Clip for Brier
        probs = np.clip(oof_scores, 0, 1)
        result["brier"] = brier_score_loss(y_true, probs)

    return result


def run_per_problem_analysis(
    exploit_traj: pd.DataFrame,
    control_traj: pd.DataFrame,
    output_dir: Path,
) -> pd.DataFrame:
    """Run binary emergence prediction at per-problem granularity."""
    output_dir.mkdir(parents=True, exist_ok=True)

    exploit_ckpts = sorted(exploit_traj["checkpoint"].unique())
    control_ckpts = sorted(control_traj["checkpoint"].unique())
    max_cutoff_idx = min(len(exploit_ckpts), len(control_ckpts)) - 1

    print(f"Exploit run checkpoints: {exploit_ckpts}")
    print(f"Control run checkpoints: {control_ckpts}")
    print(f"Max cutoff index: {max_cutoff_idx}")

    # Target stats
    e_tgt = exploit_traj.groupby("task_id")["target"].first()
    c_tgt = control_traj.groupby("task_id")["target"].first()
    print(f"Exploit run: {e_tgt.sum()}/{len(e_tgt)} positive")
    print(f"Control run: {c_tgt.sum()}/{len(c_tgt)} positive")
    print(f"Combined: {e_tgt.sum() + c_tgt.sum()}/{len(e_tgt) + len(c_tgt)} positive")
    print()

    all_results = []

    for cutoff_idx in range(max_cutoff_idx + 1):
        print(
            f"--- Cutoff index {cutoff_idx}: "
            f"exploit ckpt {exploit_ckpts[cutoff_idx]}, "
            f"control ckpt {control_ckpts[cutoff_idx]} ---"
        )

        for metric in PROBLEM_METRICS:
            # Build feature dataframe
            rows = []
            for run_df, run_label in [
                (exploit_traj, "exploit"),
                (control_traj, "control"),
            ]:
                task_ids = sorted(run_df["task_id"].unique())
                for tid in task_ids:
                    feats = compute_problem_features_at_cutoff(
                        run_df, tid, cutoff_idx, metric
                    )
                    if feats is not None:
                        rows.append(
                            {
                                "task_id": tid,
                                "exploit_type": feats["exploit_type"],
                                "run": run_label,
                                "level": feats["level"],
                                "slope": feats["slope"],
                                "target": feats["target"],
                            }
                        )

            if not rows:
                continue

            features_df = pd.DataFrame(rows)

            result_base = {
                "cutoff_idx": cutoff_idx,
                "exploit_cutoff_ckpt": exploit_ckpts[cutoff_idx],
                "control_cutoff_ckpt": control_ckpts[cutoff_idx],
                "metric": metric,
            }

            # Threshold on level (no params)
            level_auc = threshold_rule_auc(features_df, "level")
            all_results.append(
                {**result_base, "model": "threshold_level", "auc": level_auc["auc"],
                 "n": level_auc["n"]}
            )

            # Threshold on slope (no params)
            slope_auc = threshold_rule_auc(features_df, "slope")
            all_results.append(
                {**result_base, "model": "threshold_slope", "auc": slope_auc["auc"],
                 "n": slope_auc["n"]}
            )

            # Tuned projection (1 param, blocked CV)
            tuned = blocked_kfold_auc(features_df, "tuned_projection")
            all_results.append(
                {**result_base, "model": "tuned_projection", "auc": tuned["auc"],
                 "n": tuned["n"]}
            )

            # Logistic (3 params, blocked CV)
            logistic = blocked_kfold_auc(features_df, "logistic")
            all_results.append(
                {**result_base, "model": "logistic_level_slope", "auc": logistic["auc"],
                 "brier": logistic.get("brier"),
                 "n": logistic["n"]}
            )

            print(
                f"  {metric:<20s} "
                f"level={level_auc['auc']:.3f}  "
                f"slope={slope_auc['auc']:.3f}  "
                f"tuned={tuned['auc']:.3f}  "
                f"logistic={logistic['auc']:.3f}"
            )

        print()

    results_df = pd.DataFrame(all_results)
    results_df.to_csv(output_dir / "per_problem_results.csv", index=False)
    print(f"Saved to {output_dir / 'per_problem_results.csv'}")

    plot_per_problem_auc(results_df, output_dir)
    plot_per_problem_separation(exploit_traj, control_traj, output_dir)

    return results_df


def plot_per_problem_auc(results_df: pd.DataFrame, output_dir: Path) -> None:
    """Plot AUC vs cutoff for per-problem analysis."""
    fig, ax = plt.subplots(figsize=(6, 4))

    models = ["threshold_level", "threshold_slope", "tuned_projection", "logistic_level_slope"]
    labels = {
        "threshold_level": "Threshold (level)",
        "threshold_slope": "Threshold (slope)",
        "tuned_projection": "Tuned proj. (level+a*slope)",
        "logistic_level_slope": "Logistic (level+slope)",
    }
    styles = {
        "threshold_level": ("o-", "C0"),
        "threshold_slope": ("s--", "C1"),
        "tuned_projection": ("P-", "C5"),
        "logistic_level_slope": ("D--", "C2"),
    }

    metric = "accessibility"
    for model in models:
        sub = results_df[
            (results_df["metric"] == metric) & (results_df["model"] == model)
        ].sort_values("cutoff_idx")
        if len(sub) == 0:
            continue
        style, color = styles[model]
        ax.plot(
            sub["cutoff_idx"], sub["auc"], style, color=color,
            label=labels[model], markersize=6, linewidth=1.5,
        )

    ax.axhline(y=0.5, color="gray", linestyle=":", alpha=0.5, label="Random")
    ax.set_xlabel("Cutoff (ordinal checkpoint index)")
    ax.set_ylabel("AUC")
    ax.set_title("Per-Problem Binary Emergence: Accessibility", fontweight="bold")
    ax.set_ylim(0.4, 1.05)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    for ext in [".png", ".pdf"]:
        fig.savefig(output_dir / f"per_problem_auc{ext}", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved per-problem AUC plot to {output_dir / 'per_problem_auc.png'}")


def plot_per_problem_separation(
    exploit_traj: pd.DataFrame, control_traj: pd.DataFrame, output_dir: Path
) -> None:
    """Plot accessibility distributions at selected cutoffs, colored by target."""
    exploit_ckpts = sorted(exploit_traj["checkpoint"].unique())
    control_ckpts = sorted(control_traj["checkpoint"].unique())
    max_cutoff = min(len(exploit_ckpts), len(control_ckpts))

    cutoffs_to_plot = [0, 2, 4, min(max_cutoff - 1, 6)]
    cutoffs_to_plot = sorted(set(c for c in cutoffs_to_plot if c < max_cutoff))

    fig, axes = plt.subplots(1, len(cutoffs_to_plot), figsize=(4 * len(cutoffs_to_plot), 4))
    if len(cutoffs_to_plot) == 1:
        axes = [axes]

    for ax, cidx in zip(axes, cutoffs_to_plot):
        for run_df, run_label, x_offset in [
            (exploit_traj, "exploit", 0),
            (control_traj, "control", 2),
        ]:
            task_ids = sorted(run_df["task_id"].unique())
            pos_vals = []
            neg_vals = []
            for tid in task_ids:
                feats = compute_problem_features_at_cutoff(run_df, tid, cidx)
                if feats is not None:
                    if feats["target"] == 1:
                        pos_vals.append(feats["level"])
                    else:
                        neg_vals.append(feats["level"])
            pos_vals = np.array(pos_vals)
            neg_vals = np.array(neg_vals)

            jitter = 0.1
            ax.scatter(
                np.zeros(len(neg_vals)) + x_offset + np.random.normal(0, jitter, len(neg_vals)),
                neg_vals, c="C0", alpha=0.15, s=10,
            )
            ax.scatter(
                np.ones(len(pos_vals)) + x_offset + np.random.normal(0, jitter, len(pos_vals)),
                pos_vals, c="C3", alpha=0.15, s=10,
            )

        ax.set_xticks([0, 1, 2, 3])
        ax.set_xticklabels(["E-", "E+", "C-", "C+"], fontsize=8)
        ax.set_title(f"Cutoff {cidx}")
        ax.set_ylabel("Accessibility")
        ax.grid(True, alpha=0.3, axis="y")

    fig.suptitle(
        "Per-Problem Accessibility by Target (E=exploit run, C=control)",
        fontsize=11, fontweight="bold",
    )
    plt.tight_layout()
    for ext in [".png", ".pdf"]:
        fig.savefig(
            output_dir / f"per_problem_separation{ext}", dpi=150, bbox_inches="tight"
        )
    plt.close(fig)
    print(f"Saved per-problem separation plot to {output_dir / 'per_problem_separation.png'}")


def main():
    parser = argparse.ArgumentParser(
        description="Binary exploit emergence predictor"
    )
    parser.add_argument(
        "--exploit-run",
        type=Path,
        default=None,
        help="Path to exploit_rate_scaling_by_type.csv (only needed with --max-over-prefills)",
    )
    parser.add_argument(
        "--control-run",
        type=Path,
        default=None,
        help="Path to exploit_rate_scaling_by_type.csv (only needed with --max-over-prefills)",
    )
    parser.add_argument(
        "--exploit-evals",
        type=Path,
        required=True,
        help="Path to evals directory for the exploiting run",
    )
    parser.add_argument(
        "--control-evals",
        type=Path,
        required=True,
        help="Path to evals directory for the control run",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Output directory (default: auto-generated with run_context)",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.10,
        help="Exploit rate threshold for binary target (default: 0.10)",
    )
    parser.add_argument(
        "--max-over-prefills",
        action="store_true",
        help="Use max-over-prefills aggregation (legacy). Default is pooled (avg).",
    )
    parser.add_argument(
        "--exploit-trajectory",
        type=Path,
        default=None,
        help="Path to trajectory_analysis.csv for exploit run (for per-problem mode)",
    )
    parser.add_argument(
        "--control-trajectory",
        type=Path,
        default=None,
        help="Path to trajectory_analysis.csv for control run (for per-problem mode)",
    )
    parser.add_argument(
        "--per-problem",
        action="store_true",
        help="Run per-problem analysis using trajectory_analysis.csv",
    )
    parser.add_argument(
        "--exploit-logprobs",
        type=Path,
        default=None,
        help="Directory with exploit logprob checkpoint-{N}.jsonl files "
             "(adds exploit_logprob metric to emergence prediction)",
    )
    parser.add_argument(
        "--gp-constrained",
        action="store_true",
        help="Compute IS-constrained GP features and add gp_log_rate_p0 as a metric. "
             "Slower (fits one GP per exploit type) but may improve early-checkpoint estimates.",
    )
    parser.add_argument(
        "--jensen-correction",
        action="store_true",
        help="Use Jensen gap variance correction for GP IS constraints. "
             "Replaces heuristic alpha*KL with data-driven KL - Var(kl)/2 offsets. "
             "Implies --gp-constrained.",
    )
    parser.add_argument(
        "--extra-evals",
        nargs="*",
        default=[],
        help="Additional runs as label:path pairs (e.g., qwen:/path/to/evals)",
    )
    args = parser.parse_args()

    if args.max_over_prefills and (args.exploit_run is None or args.control_run is None):
        parser.error("--exploit-run and --control-run are required with --max-over-prefills")

    # Build config for run_context
    config_args = {
        "exploit_evals": str(args.exploit_evals),
        "control_evals": str(args.control_evals),
        "threshold": args.threshold,
        "aggregation": "max-over-prefills" if args.max_over_prefills else "pooled",
    }
    if args.exploit_run:
        config_args["exploit_run"] = str(args.exploit_run)
    if args.control_run:
        config_args["control_run"] = str(args.control_run)
    if args.per_problem:
        config_args["mode"] = "per-problem"
    if args.exploit_logprobs:
        config_args["exploit_logprobs"] = str(args.exploit_logprobs)
    if args.jensen_correction:
        args.gp_constrained = True  # jensen implies gp-constrained
    if args.gp_constrained:
        config_args["gp_constrained"] = True
    if args.jensen_correction:
        config_args["jensen_correction"] = True
    if args.extra_evals:
        config_args["extra_evals"] = args.extra_evals

    def _run_all(output_dir: Path) -> None:
        """Run the full analysis pipeline into output_dir."""
        if args.per_problem:
            # Infer trajectory paths from scaling CSVs or evals dir
            if args.exploit_trajectory:
                exploit_traj = args.exploit_trajectory
            elif args.exploit_run:
                exploit_traj = args.exploit_run.parent / "trajectory_analysis.csv"
            else:
                parser.error("--exploit-trajectory required for --per-problem without --exploit-run")
            if args.control_trajectory:
                control_traj = args.control_trajectory
            elif args.control_run:
                control_traj = args.control_run.parent / "trajectory_analysis.csv"
            else:
                parser.error("--control-trajectory required for --per-problem without --control-run")

            print("=== PER-PROBLEM ANALYSIS ===")
            print(f"Exploit trajectory: {exploit_traj}")
            print(f"Control trajectory: {control_traj}")
            print()

            exploit_traj_df = load_problem_data(exploit_traj, "exploit")
            control_traj_df = load_problem_data(control_traj, "control")

            pp_output = output_dir / "per_problem"
            pp_results = run_per_problem_analysis(
                exploit_traj_df, control_traj_df, pp_output
            )

            print("\n=== PER-PROBLEM SUMMARY ===")
            for model in ["threshold_level", "threshold_slope", "tuned_projection", "logistic_level_slope"]:
                sub = pp_results[
                    (pp_results["metric"] == "accessibility") & (pp_results["model"] == model)
                ]
                if len(sub) == 0:
                    continue
                best = sub.loc[sub["auc"].idxmax()]
                print(f"  {model:<25s} best AUC={best['auc']:.3f} at idx {int(best['cutoff_idx'])}")
            return

        # Per-type analysis
        agg_mode = "max-over-prefills" if args.max_over_prefills else "pooled (avg)"
        print(f"Aggregation mode: {agg_mode}")
        print()

        print("Loading exploiting run...")
        exploit_df = load_run_data(
            evals_dir=args.exploit_evals,
            run_label="exploit",
            threshold=args.threshold,
            scaling_csv=args.exploit_run if args.max_over_prefills else None,
            exploit_logprobs_dir=args.exploit_logprobs,
        )
        print(f"  {len(exploit_df)} rows, targets: {dict(exploit_df.groupby('exploit_type')['target'].first())}")

        print("Loading control run...")
        control_df = load_run_data(
            evals_dir=args.control_evals,
            run_label="control",
            threshold=args.threshold,
            scaling_csv=args.control_run if args.max_over_prefills else None,
            exploit_logprobs_dir=args.exploit_logprobs,
        )
        print(f"  {len(control_df)} rows, targets: {dict(control_df.groupby('exploit_type')['target'].first())}")

        run_dfs = [("exploit", exploit_df), ("control", control_df)]

        # Build GP config if requested (GP features computed per cutoff in run_analysis)
        gp_config = None
        if args.gp_constrained:
            gp_config = {
                "exploit": {"evals_dir": args.exploit_evals},
                "control": {"evals_dir": args.control_evals},
                "_jensen_correction": args.jensen_correction,
            }

        for spec in args.extra_evals:
            label, path = spec.split(":", 1)
            print(f"Loading extra run '{label}'...")
            extra_df = load_run_data(
                evals_dir=Path(path),
                run_label=label,
                threshold=args.threshold,
            )
            print(f"  {len(extra_df)} rows, targets: {dict(extra_df.groupby('exploit_type')['target'].first())}")
            run_dfs.append((label, extra_df))
            if gp_config is not None:
                gp_config[label] = {"evals_dir": Path(path)}

        print()

        results = run_analysis(run_dfs, output_dir, gp_config=gp_config)

        # Print summary table
        has_elp = all("exploit_logprob" in df.columns for _, df in run_dfs)
        summary_metrics = METRICS + (EXPLOIT_LOGPROB_METRICS if has_elp else [])
        if args.gp_constrained:
            summary_metrics = summary_metrics + GP_METRICS
        print("\n=== SUMMARY: Best AUC per metric ===")
        for metric in summary_metrics:
            for model in ["threshold_level", "tuned_projection", "threshold_projected", "threshold_slope", "logistic_level_slope"]:
                sub = results[
                    (results["metric"] == metric) & (results["model"] == model)
                ]
                if len(sub) == 0:
                    continue
                best = sub.loc[sub["auc"].idxmax()]
                first_perfect = sub[sub["auc"] >= 1.0]
                first_str = (
                    f"first perfect at idx {int(first_perfect['cutoff_idx'].min())}"
                    if len(first_perfect) > 0
                    else "never perfect"
                )
                print(
                    f"  {metric:<30s} {model:<25s} "
                    f"best AUC={best['auc']:.3f} at idx {int(best['cutoff_idx'])}, "
                    f"{first_str}"
                )

    # Run with or without experiment context
    if args.output_dir:
        args.output_dir.mkdir(parents=True, exist_ok=True)
        _run_all(args.output_dir)
    else:
        from rh_indicators.run_utils import run_context

        with run_context(
            base_dir=Path("results/trajectory_prediction"),
            run_prefix="binary_emergence",
            config_args=config_args,
        ) as output_dir:
            _run_all(output_dir)


if __name__ == "__main__":
    main()
