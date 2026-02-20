"""Exploit rate scaling law computation.

Computes lower bound on P(exploit) at each checkpoint using KL divergence
and conditional exploit success rate.
"""

import numpy as np
import pandas as pd


def compute_exploit_rate_scaling(
    kl_df: pd.DataFrame,
    checkpoints: list[int],
    eval_df: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Compute lower bound on P(exploit) at each checkpoint.

    For each checkpoint, computes:
        P(exploit @ step) >= max_prefill [ exp(-KL(prefill)) * P(exploit | prefill, step) ]

    In log space: log(P) = max_prefill [ -KL(prefill) + log(exploit_rate) ]

    Where KL(prefill) = KL(P_ref || P_eval) measures how much the checkpoint has
    diverged from the reference (prefill-generating) model on the prefill text.

    This gives a lower bound because we're only considering tested prefills.

    Args:
        kl_df: DataFrame with columns: task_id, checkpoint, prefill_tokens,
               kl_divergence, exploit_success
        checkpoints: List of checkpoints to analyze
        eval_df: Optional eval DataFrame for prefill 0 data (where KL=0)

    Returns:
        DataFrame with columns: checkpoint, best_prefill, mean_neg_kl, exploit_rate,
                               log_exploit_lower_bound, exploit_lower_bound, n_samples
    """
    results = []

    for checkpoint in checkpoints:
        ckpt_data = kl_df[kl_df["checkpoint"] == checkpoint]

        prefill_levels = sorted(ckpt_data["prefill_tokens"].unique()) if len(ckpt_data) > 0 else []
        best_score = float("-inf")
        best_info = None

        # Handle prefill 0 from eval_df (KL = 0 since no prefill)
        if eval_df is not None:
            prefill0_data = eval_df[
                (eval_df["checkpoint"] == checkpoint) &
                (eval_df["prefill_tokens"] == 0)
            ]
            if len(prefill0_data) > 0:
                exploit_rate = prefill0_data["exploit_success"].mean()
                if exploit_rate > 0:
                    # KL(prefill=0) = 0, so score = log(exploit_rate)
                    score = np.log(exploit_rate)
                    if score > best_score:
                        best_score = score
                        best_info = {
                            "checkpoint": checkpoint,
                            "best_prefill": 0,
                            "mean_neg_kl": 0.0,
                            "exploit_rate": exploit_rate,
                            "log_exploit_lower_bound": score,
                            "exploit_lower_bound": exploit_rate,
                            "n_samples": len(prefill0_data),
                        }

        # Handle other prefill levels from kl_df
        for prefill in prefill_levels:
            prefill_data = ckpt_data[ckpt_data["prefill_tokens"] == prefill]
            if len(prefill_data) == 0:
                continue

            # Mean negative KL across tasks at this prefill level
            # KL >= 0, so -KL <= 0
            mean_kl = prefill_data["kl_divergence"].mean()
            mean_neg_kl = -mean_kl

            # Exploit rate: fraction of samples that succeed
            exploit_rate = prefill_data["exploit_success"].mean()

            if exploit_rate > 0:
                # log(exp(-KL) * P(exploit|prefill)) = -KL + log(rate)
                score = mean_neg_kl + np.log(exploit_rate)
            else:
                score = float("-inf")

            if score > best_score:
                best_score = score
                best_info = {
                    "checkpoint": checkpoint,
                    "best_prefill": prefill,
                    "mean_neg_kl": mean_neg_kl,
                    "exploit_rate": exploit_rate,
                    "log_exploit_lower_bound": score,
                    "exploit_lower_bound": np.exp(score) if score > float("-inf") else 0.0,
                    "n_samples": len(prefill_data),
                }

        if best_info is not None:
            results.append(best_info)

    return pd.DataFrame(results)


def compute_pooled_exploit_rate_scaling(
    kl_df: pd.DataFrame,
    checkpoints: list[int],
    eval_df: pd.DataFrame | None = None,
    laplace_eps: float = 0.5,
) -> pd.DataFrame:
    """Compute averaged exploit rate scaling across all prefill levels.

    Computes an independent importance-sampling lower bound at each prefill level,
    then averages those bounds:

        score(ckpt) = mean_p [ -KL_p + log(smoothed_rate_p) ]

    where KL_p and rate_p are computed per-prefill (averaging KL across problems,
    counting successes across problems at that prefill).

    Uses accumulated-n Laplace smoothing for zero-success cells: when a cell
    (prefill, checkpoint) has 0 successes, its effective n includes all samples
    from cells at higher prefills and later checkpoints that also have 0 successes.
    This borrows strength from the monotonicity assumption (if you can't exploit
    with more help / more training, you probably can't with less) to give tighter
    estimates for genuine zeros vs sampling noise.

    For cells with successes > 0, uses the cell's own n for smoothing.

    Args:
        kl_df: DataFrame with columns: task_id, checkpoint, prefill_tokens,
               kl_divergence, exploit_success
        checkpoints: List of checkpoints to analyze
        eval_df: Optional eval DataFrame for prefill 0 data (where KL=0)
        laplace_eps: Smoothing parameter for Laplace smoothing (default: 0.5)

    Returns:
        DataFrame with columns: checkpoint, mean_neg_kl, exploit_rate,
                               log_exploit_lower_bound, exploit_lower_bound, n_samples
    """
    # Step 1: Build grid of (prefill, checkpoint) -> {n, successes, mean_kl}
    cells: dict[tuple[int, int], dict] = {}

    if eval_df is not None:
        for ckpt in checkpoints:
            p0 = eval_df[
                (eval_df["checkpoint"] == ckpt) & (eval_df["prefill_tokens"] == 0)
            ]
            if len(p0) > 0:
                cells[(0, ckpt)] = {
                    "n": len(p0),
                    "successes": int(p0["exploit_success"].sum()),
                    "mean_kl": 0.0,
                }

    kl_clean = kl_df.dropna(subset=["kl_divergence"])
    for (ckpt, pfx), pdata in kl_clean.groupby(["checkpoint", "prefill_tokens"]):
        if ckpt in checkpoints:
            cells[(int(pfx), int(ckpt))] = {
                "n": len(pdata),
                "successes": int(pdata["exploit_success"].sum()),
                "mean_kl": pdata["kl_divergence"].mean(),
            }

    if not cells:
        return pd.DataFrame()

    # Step 2: For zero-success cells, accumulate n from cells at
    # (prefill' >= prefill, SAME checkpoint) that also have 0 successes.
    # Higher prefill: monotonicity — can't exploit with less help if can't with more.
    # Same checkpoint only: cross-checkpoint accumulation is wrong (earlier ckpts
    # aren't "more risky", later ckpts leak future data).
    sorted_prefills = sorted(set(p for p, c in cells))

    accumulated_n: dict[tuple[int, int], int] = {}
    for (p, c), info in cells.items():
        if info["successes"] == 0:
            acc = 0
            for p2 in sorted_prefills:
                if p2 < p:
                    continue
                cell2 = cells.get((p2, c))
                if cell2 is not None and cell2["successes"] == 0:
                    acc += cell2["n"]
            accumulated_n[(p, c)] = acc

    # Step 3: Compute per-prefill scores per checkpoint, then average
    results = []
    for ckpt in checkpoints:
        prefill_scores = []

        for pfx in sorted_prefills:
            info = cells.get((pfx, ckpt))
            if info is None:
                continue

            n_p = info["n"]
            succ_p = info["successes"]
            neg_kl = -info["mean_kl"]

            # Use accumulated n for zero-success cells
            if succ_p == 0 and (pfx, ckpt) in accumulated_n:
                n_smooth = accumulated_n[(pfx, ckpt)]
            else:
                n_smooth = n_p

            smoothed_rate = (succ_p + laplace_eps) / (n_smooth + 2 * laplace_eps)
            prefill_scores.append((neg_kl, np.log(smoothed_rate), n_p, succ_p))

        if not prefill_scores:
            continue

        neg_kls = [s[0] for s in prefill_scores]
        log_rates = [s[1] for s in prefill_scores]
        ns = [s[2] for s in prefill_scores]
        succs = [s[3] for s in prefill_scores]

        mean_neg_kl = np.mean(neg_kls)
        mean_log_rate = np.mean(log_rates)
        log_lb = mean_neg_kl + mean_log_rate

        total_n = sum(ns)
        total_succ = sum(succs)
        exploit_rate = total_succ / total_n if total_n > 0 else 0.0

        results.append({
            "checkpoint": ckpt,
            "mean_neg_kl": mean_neg_kl,
            "exploit_rate": exploit_rate,
            "log_exploit_lower_bound": log_lb,
            "exploit_lower_bound": np.exp(log_lb) if log_lb > -100 else 0.0,
            "n_samples": total_n,
        })

    return pd.DataFrame(results)
