"""Exact importance-sampling estimates of exploit probability.

Implements the exact IS identity from docs/prefix_probability_estimation.md:

    μ_P(C) = E_{z~q}[ A(x,z) · B(x,z) ]

where:
    A(x,z) = (P_τ(z|x) / D_τᴰ(z|x)) · r_P(x,z)
    B(x,z) = π_D(x) / a_D(x,z)

And the fixed-prefix lower bound (§3):
    L(x,z) = P_τ(z|x) · r_P(x,z)
    L*(x) = max_N L(x, z_N(x))

All computation is done in log space with logsumexp for numerical stability.
"""

import math
from dataclasses import dataclass

import numpy as np
import pandas as pd

from .temperature import _logsumexp


@dataclass
class ISSample:
    """A single (task, prefix) pair with all IS terms."""

    task_id: str
    prefill_tokens: int  # prefix length N (in words)
    checkpoint: int

    # Log-space IS terms
    log_p_target: float  # log P_τ(z|x) — target prefix prob
    log_p_donor: float  # log D_τᴰ(z|x) — donor prefix prob
    r_target: float  # r_P(x,z) — target exploit rate given prefix
    pi_donor: float  # π_D(x) — donor spontaneous exploit rate
    a_donor: float  # a_D(x,z) — donor exploit rate given prefix

    @property
    def log_A(self) -> float:
        """log A(x,z) = log P_τ(z|x) - log D_τᴰ(z|x) + log r_P(x,z)"""
        if self.r_target <= 0:
            return float("-inf")
        return self.log_p_target - self.log_p_donor + math.log(self.r_target)

    @property
    def log_B(self) -> float:
        """log B(x,z) = log π_D(x) - log a_D(x,z)"""
        if self.pi_donor <= 0 or self.a_donor <= 0:
            return float("-inf")
        return math.log(self.pi_donor) - math.log(self.a_donor)

    @property
    def log_is_weight(self) -> float:
        """log(A · B) = log_A + log_B — the full IS sample weight."""
        a = self.log_A
        b = self.log_B
        if a == float("-inf") or b == float("-inf"):
            return float("-inf")
        return a + b

    @property
    def log_lower_bound(self) -> float:
        """log L(x,z) = log P_τ(z|x) + log r_P(x,z) — fixed-prefix lower bound."""
        if self.r_target <= 0:
            return float("-inf")
        return self.log_p_target + math.log(self.r_target)


def compute_cell_is_estimate(samples: list[ISSample]) -> float:
    """Compute cell-level IS estimate via logsumexp.

    log μ̂_P(C) = logsumexp_i(λ_i) − log(m_C)

    where λ_i = log(A_i · B_i) for each sample.

    Args:
        samples: List of ISSample objects for this cell.

    Returns:
        log μ̂_P(C) — log of the estimated cell-level exploit probability.
        Returns -inf if no valid samples.
    """
    if not samples:
        return float("-inf")

    log_weights = [s.log_is_weight for s in samples]
    # Filter out -inf
    finite = [w for w in log_weights if w > float("-inf")]
    if not finite:
        return float("-inf")

    return _logsumexp(finite) - math.log(len(samples))


def compute_cell_lower_bound(samples: list[ISSample]) -> float:
    """Compute cell-level fixed-prefix lower bound.

    For each task, takes max over prefix lengths, then averages over tasks.

    log μ̂_P(C) >= logsumexp_tasks(max_N log L(x, z_N)) − log(n_tasks)

    Args:
        samples: List of ISSample objects for this cell.

    Returns:
        log of the cell-level lower bound estimate.
    """
    if not samples:
        return float("-inf")

    # Group by task_id, take max over prefix lengths per task
    task_max: dict[str, float] = {}
    for s in samples:
        lb = s.log_lower_bound
        if s.task_id not in task_max or lb > task_max[s.task_id]:
            task_max[s.task_id] = lb

    finite = [v for v in task_max.values() if v > float("-inf")]
    if not finite:
        return float("-inf")

    return _logsumexp(finite) - math.log(len(task_max))


def compute_is_from_dataframes(
    logprob_df: pd.DataFrame,
    donor_logprob_df: pd.DataFrame,
    eval_df: pd.DataFrame,
    donor_eval_df: pd.DataFrame,
    checkpoint: int,
    target_temperature: float = 0.4,
    donor_temperature: float = 0.4,
) -> dict:
    """Compute IS estimates from pre-loaded DataFrames.

    This is the high-level interface that assembles ISSample objects from
    the standard data formats and computes both the full IS estimate and
    the fixed-prefix lower bound.

    Args:
        logprob_df: Target model logprob data. Must have columns:
            task_id, prefill_tokens, prefill_logprob_sum, exploit_success.
            If temperature != 1.0, must also have top_logprobs for correction.
        donor_logprob_df: Donor model logprob data. Same schema.
        eval_df: Target model eval results (for r_P at each prefill).
            Columns: task_id, prefill_tokens, exploit_success, checkpoint.
        donor_eval_df: Donor model eval results (for π_D and a_D).
            Columns: task_id, prefill_tokens, exploit_success.
        checkpoint: Which checkpoint to analyze.
        target_temperature: Temperature used for target model sampling.
        donor_temperature: Temperature used for donor model sampling.

    Returns:
        Dict with:
        - log_is_estimate: float (log of full IS estimate)
        - is_estimate: float (probability)
        - log_lower_bound: float (log of fixed-prefix lower bound)
        - lower_bound: float (probability)
        - n_samples: int
        - n_tasks: int
        - per_prefill: list of dicts with per-prefill-level estimates
    """
    # Filter eval data to this checkpoint
    ckpt_eval = eval_df[eval_df["checkpoint"] == checkpoint]

    # Compute r_P(x,z): target exploit rate at each (task, prefill)
    # Group by task_id and prefill_tokens
    r_target = ckpt_eval.groupby(["task_id", "prefill_tokens"])["exploit_success"].mean().to_dict()

    # Compute π_D(x): donor spontaneous exploit rate (prefill=0)
    donor_pfx0 = donor_eval_df[donor_eval_df["prefill_tokens"] == 0]
    pi_donor = donor_pfx0.groupby("task_id")["exploit_success"].mean().to_dict()

    # Compute a_D(x,z): donor exploit rate given prefix
    a_donor = (
        donor_eval_df[donor_eval_df["prefill_tokens"] > 0]
        .groupby(["task_id", "prefill_tokens"])["exploit_success"]
        .mean()
        .to_dict()
    )

    # Filter logprob data to this checkpoint
    ckpt_logprob = logprob_df[logprob_df["checkpoint"] == checkpoint]

    # Build ISSample objects
    samples = []
    for _, row in ckpt_logprob.iterrows():
        tid = row["task_id"]
        pfx = row["prefill_tokens"]

        if pfx == 0:
            continue  # No IS at prefill=0

        # Get target prefix logprob (already sum at T=1; temperature correction
        # should be applied before calling this function if needed)
        log_p_target = row.get("prefill_logprob_sum_corrected", row["prefill_logprob_sum"])

        # Get donor prefix logprob
        donor_match = donor_logprob_df[
            (donor_logprob_df["task_id"] == tid) & (donor_logprob_df["prefill_tokens"] == pfx)
        ]
        if donor_match.empty:
            continue
        log_p_donor = donor_match.iloc[0].get(
            "prefill_logprob_sum_corrected",
            donor_match.iloc[0]["prefill_logprob_sum"],
        )

        # Get behavioral terms
        r = r_target.get((tid, pfx), 0.0)
        pi = pi_donor.get(tid, 0.0)
        a = a_donor.get((tid, pfx), 0.0)

        samples.append(
            ISSample(
                task_id=tid,
                prefill_tokens=pfx,
                checkpoint=checkpoint,
                log_p_target=log_p_target,
                log_p_donor=log_p_donor,
                r_target=r,
                pi_donor=pi,
                a_donor=a,
            )
        )

    # Compute cell-level estimates
    log_is = compute_cell_is_estimate(samples)
    log_lb = compute_cell_lower_bound(samples)

    # Per-prefill breakdown
    prefill_levels = sorted(set(s.prefill_tokens for s in samples))
    per_prefill = []
    for pfx in prefill_levels:
        pfx_samples = [s for s in samples if s.prefill_tokens == pfx]
        pfx_log_is = compute_cell_is_estimate(pfx_samples)
        pfx_log_lb = compute_cell_lower_bound(pfx_samples)
        per_prefill.append(
            {
                "prefill_tokens": pfx,
                "log_is_estimate": pfx_log_is,
                "is_estimate": math.exp(pfx_log_is) if pfx_log_is > -100 else 0.0,
                "log_lower_bound": pfx_log_lb,
                "lower_bound": math.exp(pfx_log_lb) if pfx_log_lb > -100 else 0.0,
                "n_samples": len(pfx_samples),
                "n_tasks": len(set(s.task_id for s in pfx_samples)),
                "mean_log_A": (
                    float(np.mean([s.log_A for s in pfx_samples if s.log_A > float("-inf")]))
                    if any(s.log_A > float("-inf") for s in pfx_samples)
                    else float("-inf")
                ),
                "mean_log_B": (
                    float(np.mean([s.log_B for s in pfx_samples if s.log_B > float("-inf")]))
                    if any(s.log_B > float("-inf") for s in pfx_samples)
                    else float("-inf")
                ),
            }
        )

    return {
        "checkpoint": checkpoint,
        "log_is_estimate": log_is,
        "is_estimate": math.exp(log_is) if log_is > -100 else 0.0,
        "log_lower_bound": log_lb,
        "lower_bound": math.exp(log_lb) if log_lb > -100 else 0.0,
        "n_samples": len(samples),
        "n_tasks": len(set(s.task_id for s in samples)),
        "per_prefill": per_prefill,
    }


def compute_heuristic_is(
    kl_df: pd.DataFrame,
    eval_df: pd.DataFrame,
    checkpoint: int,
) -> dict:
    """Compute the old heuristic IS estimate for comparison.

    This is the existing exp(-KL) * rate approach from scaling.py,
    wrapped for easy comparison with the exact IS estimate.

    Args:
        kl_df: KL divergence data with kl_divergence column.
        eval_df: Eval results with exploit_success.
        checkpoint: Checkpoint to analyze.

    Returns:
        Dict with log_heuristic_estimate and heuristic_estimate.
    """
    from .scaling import compute_exploit_rate_scaling

    checkpoints = [checkpoint]
    result = compute_exploit_rate_scaling(kl_df, checkpoints, eval_df)

    if result.empty:
        return {
            "checkpoint": checkpoint,
            "log_heuristic_estimate": float("-inf"),
            "heuristic_estimate": 0.0,
        }

    row = result.iloc[0]
    return {
        "checkpoint": checkpoint,
        "log_heuristic_estimate": row["log_exploit_lower_bound"],
        "heuristic_estimate": row["exploit_lower_bound"],
        "best_prefill": row["best_prefill"],
    }
