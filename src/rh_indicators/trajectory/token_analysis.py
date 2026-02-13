"""Token-based trajectory analysis.

Analyzes how minimum prefill tokens needed to exploit changes over training.
"""

import pandas as pd


def compute_min_prefill_trajectories(
    df: pd.DataFrame,
    checkpoints: list[int],
    max_prefill: int = 100,
) -> pd.DataFrame:
    """Compute min prefill to exploit for each problem at each checkpoint.

    For each task and checkpoint, finds the minimum number of prefill tokens
    that triggers a successful exploit.

    Args:
        df: DataFrame with columns: task_id, exploit_type, checkpoint, prefill_tokens,
            exploit_success
        checkpoints: List of checkpoint values to analyze
        max_prefill: Maximum prefill value tested (default: 100)

    Returns:
        DataFrame with columns: task_id, exploit_type, checkpoint, min_prefill, exploitable
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

    Analyzes how long until each task becomes "easily exploitable" (min_prefill
    at or below the threshold).

    Args:
        trajectories: DataFrame from compute_min_prefill_trajectories with columns:
            task_id, exploit_type, checkpoint, min_prefill
        checkpoints: List of checkpoint values
        threshold: Prefill threshold for "easily exploitable" (default: 10)

    Returns:
        DataFrame with columns: task_id, exploit_type, checkpoint, min_prefill,
        accessibility, steps_to_threshold, ever_reaches_threshold, at_threshold
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
