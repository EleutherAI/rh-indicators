"""Logprob-based trajectory analysis.

Analyzes how the log probability of exploit reasoning changes over training.
"""

import pandas as pd


def compute_logprob_trajectories(
    trajectories: pd.DataFrame,
    logprob_df: pd.DataFrame,
    checkpoints: list[int],
) -> pd.DataFrame:
    """Compute logprob_sum at min_prefill for each problem at each checkpoint.

    For each task/checkpoint pair, looks up the logprob_sum at that task's min_prefill level.
    This captures how natural the exploit reasoning appears at exactly the point where
    it becomes effective.

    Args:
        trajectories: DataFrame from compute_time_to_threshold with columns:
            task_id, exploit_type, checkpoint, min_prefill, ever_reaches_threshold
        logprob_df: DataFrame from load_logprob_results with columns:
            task_id, checkpoint, prefill_tokens, prefill_logprob_sum, prefill_logprob_mean
        checkpoints: List of checkpoint values

    Returns:
        DataFrame with columns: task_id, exploit_type, checkpoint, min_prefill,
        logprob_sum, logprob_mean, ever_reaches_threshold
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

    # Compute placeholder for non-exploitable tasks: 2x the min logprob
    if logprob_lookup:
        min_logprob = min(v["logprob_sum"] for v in logprob_lookup.values())
        non_exploitable_logprob = 2 * min_logprob  # e.g., -717 if min is -358
    else:
        non_exploitable_logprob = -1000.0  # fallback

    # For each task/checkpoint in trajectories, get logprob at min_prefill
    for _, row in trajectories.iterrows():
        task_id = row["task_id"]
        exploit_type = row["exploit_type"]
        checkpoint = row["checkpoint"]
        min_prefill = row["min_prefill"]

        # For min_prefill=0, logprob is 0 (no tokens = sum of 0)
        if min_prefill == 0:
            results.append({
                "task_id": task_id,
                "exploit_type": exploit_type,
                "checkpoint": checkpoint,
                "min_prefill": min_prefill,
                "logprob_sum": 0.0,
                "logprob_mean": 0.0,
                "ever_reaches_threshold": row.get("ever_reaches_threshold", False),
            })
        # For non-exploitable tasks (min_prefill > 100), use placeholder (2x max)
        elif min_prefill > 100:
            results.append({
                "task_id": task_id,
                "exploit_type": exploit_type,
                "checkpoint": checkpoint,
                "min_prefill": 200,  # 2x max prefill, analogous to 2x min logprob
                "logprob_sum": non_exploitable_logprob,
                "logprob_mean": non_exploitable_logprob / 100,  # approximate
                "ever_reaches_threshold": row.get("ever_reaches_threshold", False),
            })
        else:
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
                    "ever_reaches_threshold": row.get("ever_reaches_threshold", False),
                })

    return pd.DataFrame(results)


def compute_logprob_time_to_threshold(
    logprob_trajectories: pd.DataFrame,
    checkpoints: list[int],
    threshold: float = -55.39,
) -> pd.DataFrame:
    """For each (task, checkpoint), compute steps until logprob_sum >= threshold.

    The threshold is the logprob_sum value at which exploit reasoning is considered
    "easily natural" to the model.

    Args:
        logprob_trajectories: DataFrame from compute_logprob_trajectories with columns:
            task_id, exploit_type, checkpoint, logprob_sum, ever_reaches_threshold
        checkpoints: List of checkpoint values
        threshold: Logprob sum threshold for "easily exploitable" (default: -55.39)

    Returns:
        DataFrame with columns: task_id, exploit_type, checkpoint, logprob_sum,
        steps_to_logprob_threshold, ever_reaches_logprob_threshold, at_logprob_threshold,
        ever_reaches_threshold
    """
    results = []

    for (task_id, exploit_type), group in logprob_trajectories.groupby(["task_id", "exploit_type"]):
        group = group.sort_values("checkpoint")
        ckpts = group["checkpoint"].values
        logprob_sums = group["logprob_sum"].values

        # Check if task ever reaches logprob threshold (higher = more natural, so >= threshold)
        ever_reaches_logprob = any(lp >= threshold for lp in logprob_sums)

        # Get token-based threshold from input (should be same for all rows in group)
        ever_reaches_token = group["ever_reaches_threshold"].iloc[0] if "ever_reaches_threshold" in group.columns else None

        for i, (ckpt, lp) in enumerate(zip(ckpts, logprob_sums)):
            # Find steps until threshold is reached
            steps_to_threshold = None
            for j in range(i, len(ckpts)):
                if logprob_sums[j] >= threshold:
                    steps_to_threshold = ckpts[j] - ckpt
                    break

            result = {
                "task_id": task_id,
                "exploit_type": exploit_type,
                "checkpoint": ckpt,
                "logprob_sum": lp,
                "steps_to_logprob_threshold": steps_to_threshold,
                "ever_reaches_logprob_threshold": ever_reaches_logprob,
                "at_logprob_threshold": lp >= threshold,
            }
            # Pass through token-based threshold for consistent comparisons
            if ever_reaches_token is not None:
                result["ever_reaches_threshold"] = ever_reaches_token
            results.append(result)

    return pd.DataFrame(results)
