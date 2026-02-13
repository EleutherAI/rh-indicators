"""KL divergence-based trajectory analysis.

Analyzes how the KL divergence between the prefill generator and evaluation
checkpoints changes over training.
"""

import pandas as pd


def compute_kl_trajectories(
    trajectories: pd.DataFrame,
    kl_df: pd.DataFrame,
    checkpoints: list[int] | None = None,
) -> pd.DataFrame:
    """Compute KL divergence at min_prefill for each problem at each checkpoint.

    For each task/checkpoint pair, looks up the KL divergence at that task's
    min_prefill level. This captures how different the eval model's distribution
    is from the generator at exactly the point where the prefill becomes effective.

    Args:
        trajectories: DataFrame from compute_time_to_threshold with columns:
            task_id, exploit_type, checkpoint, min_prefill, ever_reaches_threshold
        kl_df: DataFrame from load_kl_results with columns:
            task_id, checkpoint, prefill_tokens, kl_divergence, eval_logprob_sum
        checkpoints: List of checkpoint values

    Returns:
        DataFrame with columns: task_id, exploit_type, checkpoint, min_prefill,
        kl_divergence, kl_per_token, eval_logprob_sum, ref_logprob_sum,
        ever_reaches_threshold
    """
    results = []

    # Create lookup: (task_id, checkpoint, prefill_tokens) -> KL metrics
    # Average across attempts for each (task_id, checkpoint, prefill_tokens)
    kl_agg = kl_df.groupby(["task_id", "checkpoint", "prefill_tokens"]).agg({
        "kl_divergence": "mean",
        "kl_per_token": "mean",
        "eval_logprob_sum": "mean",
        "ref_logprob_sum": "mean",
    }).reset_index()

    kl_lookup = {}
    for _, row in kl_agg.iterrows():
        key = (row["task_id"], row["checkpoint"], row["prefill_tokens"])
        kl_lookup[key] = {
            "kl_divergence": row["kl_divergence"],
            "kl_per_token": row["kl_per_token"],
            "eval_logprob_sum": row["eval_logprob_sum"],
            "ref_logprob_sum": row["ref_logprob_sum"],
        }

    # Compute placeholder for non-exploitable tasks
    if kl_lookup:
        max_kl = max(v["kl_divergence"] for v in kl_lookup.values() if v["kl_divergence"] is not None)
        non_exploitable_kl = 2 * max_kl if max_kl else 100.0
    else:
        non_exploitable_kl = 100.0

    # For each task/checkpoint in trajectories, get KL at min_prefill
    for _, row in trajectories.iterrows():
        task_id = row["task_id"]
        exploit_type = row["exploit_type"]
        checkpoint = row["checkpoint"]
        min_prefill = row["min_prefill"]

        # For min_prefill=0, KL is 0 (no tokens)
        if min_prefill == 0:
            results.append({
                "task_id": task_id,
                "exploit_type": exploit_type,
                "checkpoint": checkpoint,
                "min_prefill": min_prefill,
                "kl_divergence": 0.0,
                "kl_per_token": 0.0,
                "eval_logprob_sum": 0.0,
                "ref_logprob_sum": 0.0,
                "ever_reaches_threshold": row.get("ever_reaches_threshold", False),
            })
        # For non-exploitable tasks (min_prefill > 100), use placeholder
        elif min_prefill > 100:
            results.append({
                "task_id": task_id,
                "exploit_type": exploit_type,
                "checkpoint": checkpoint,
                "min_prefill": 200,
                "kl_divergence": non_exploitable_kl,
                "kl_per_token": non_exploitable_kl / 100,
                "eval_logprob_sum": None,
                "ref_logprob_sum": None,
                "ever_reaches_threshold": row.get("ever_reaches_threshold", False),
            })
        else:
            key = (task_id, checkpoint, min_prefill)
            if key in kl_lookup:
                kl_data = kl_lookup[key]
                results.append({
                    "task_id": task_id,
                    "exploit_type": exploit_type,
                    "checkpoint": checkpoint,
                    "min_prefill": min_prefill,
                    "kl_divergence": kl_data["kl_divergence"],
                    "kl_per_token": kl_data["kl_per_token"],
                    "eval_logprob_sum": kl_data["eval_logprob_sum"],
                    "ref_logprob_sum": kl_data["ref_logprob_sum"],
                    "ever_reaches_threshold": row.get("ever_reaches_threshold", False),
                })

    return pd.DataFrame(results)


def compute_kl_time_to_threshold(
    kl_trajectories: pd.DataFrame,
    checkpoints: list[int] | None = None,
    threshold: float = 0.5,
) -> pd.DataFrame:
    """For each (task, checkpoint), compute steps until KL divergence <= threshold.

    Lower KL means the eval model's distribution is closer to the generator's,
    meaning the exploit reasoning appears more "natural" to the eval model.

    Args:
        kl_trajectories: DataFrame from compute_kl_trajectories with columns:
            task_id, exploit_type, checkpoint, kl_divergence, ever_reaches_threshold
        checkpoints: List of checkpoint values
        threshold: KL threshold for "easily exploitable" (default: 0.5)

    Returns:
        DataFrame with columns: task_id, exploit_type, checkpoint, kl_divergence,
        steps_to_kl_threshold, ever_reaches_kl_threshold, at_kl_threshold,
        ever_reaches_threshold
    """
    results = []

    for (task_id, exploit_type), group in kl_trajectories.groupby(["task_id", "exploit_type"]):
        group = group.sort_values("checkpoint")
        ckpts = group["checkpoint"].values
        kl_values = group["kl_divergence"].values

        # Check if task ever reaches KL threshold (lower = more similar, so <= threshold)
        ever_reaches_kl = any(kl is not None and kl <= threshold for kl in kl_values)

        # Get token-based threshold from input
        ever_reaches_token = group["ever_reaches_threshold"].iloc[0] if "ever_reaches_threshold" in group.columns else None

        for i, (ckpt, kl) in enumerate(zip(ckpts, kl_values)):
            # Find steps until threshold is reached
            steps_to_threshold = None
            for j in range(i, len(ckpts)):
                if kl_values[j] is not None and kl_values[j] <= threshold:
                    steps_to_threshold = ckpts[j] - ckpt
                    break

            result = {
                "task_id": task_id,
                "exploit_type": exploit_type,
                "checkpoint": ckpt,
                "kl_divergence": kl,
                "steps_to_kl_threshold": steps_to_threshold,
                "ever_reaches_kl_threshold": ever_reaches_kl,
                "at_kl_threshold": kl is not None and kl <= threshold,
            }
            if ever_reaches_token is not None:
                result["ever_reaches_threshold"] = ever_reaches_token
            results.append(result)

    return pd.DataFrame(results)


def compare_kl_vs_logprob(
    kl_df: pd.DataFrame,
    metric: str = "kl_divergence",
) -> pd.DataFrame:
    """Compare KL divergence vs logprob as predictors.

    Aggregates by checkpoint and prefill_tokens to compare mean values
    of KL divergence vs eval logprob sum.

    Args:
        kl_df: DataFrame from load_kl_results
        metric: Which KL metric to use ("kl_divergence" or "kl_per_token")

    Returns:
        DataFrame with aggregated comparison metrics
    """
    agg = kl_df.groupby(["checkpoint", "prefill_tokens"]).agg({
        metric: "mean",
        "eval_logprob_sum": "mean",
        "ref_logprob_sum": "mean",
        "exploit_success": "mean",
    }).reset_index()

    return agg
