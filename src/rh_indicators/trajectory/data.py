"""Data loading utilities for trajectory analysis."""

import json
import re
from pathlib import Path

import pandas as pd


def parse_filename(filename: str) -> tuple[int, int] | None:
    """Parse checkpoint and prefill values from filename.

    Handles both eval format (checkpoint-N_prefillM.jsonl) and
    logprob format (checkpoint-N_prefillM_logprobs.jsonl).

    Args:
        filename: Filename to parse

    Returns:
        Tuple of (checkpoint, prefill_tokens) or None if pattern doesn't match
    """
    match = re.match(r"checkpoint-(\d+)_prefill(\d+)(?:_logprobs)?\.jsonl$", filename)
    if match:
        return int(match.group(1)), int(match.group(2))
    return None


def load_per_problem_results(input_dir: Path) -> pd.DataFrame:
    """Load per-problem results from all checkpoint × prefill eval files.

    Args:
        input_dir: Directory containing checkpoint-*_prefill*.jsonl files

    Returns:
        DataFrame with columns: task_id, checkpoint, prefill_tokens, exploit_success,
        exploit_type, attempt_idx, secure_pass, insecure_pass
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
                        "secure_pass": record.get("secure_pass", False),
                        "insecure_pass": record.get("insecure_pass", False),
                    })
    return pd.DataFrame(rows)


def load_logprob_results(logprob_dir: Path) -> pd.DataFrame | None:
    """Load logprob results from checkpoint × prefill logprob files.

    Args:
        logprob_dir: Directory containing checkpoint-*_prefill*_logprobs.jsonl files

    Returns:
        DataFrame with columns: task_id, checkpoint, prefill_tokens, exploit_type,
        attempt_idx, prefill_logprob_sum, prefill_logprob_mean, prefill_num_tokens,
        exploit_success. Returns None if directory doesn't exist or no data found.
    """
    if not logprob_dir.exists():
        return None

    rows = []
    for jsonl_file in sorted(logprob_dir.glob("checkpoint-*_prefill*.jsonl")):
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
                        "exploit_type": record.get("exploit_type"),
                        "attempt_idx": record.get("attempt_idx", 0),
                        "prefill_logprob_sum": record.get("prefill_logprob_sum"),
                        "prefill_logprob_mean": record.get("prefill_logprob_mean"),
                        "prefill_num_tokens": record.get("prefill_num_tokens"),
                        "exploit_success": record.get("exploit_success", False),
                    })

    if not rows:
        return None
    return pd.DataFrame(rows)


def load_exploit_logprobs(exploit_logprobs_dir: Path) -> pd.DataFrame | None:
    """Load exploit logprob results from checkpoint-{N}.jsonl files.

    These measure log P(exploit_code | problem) for ground-truth exploit
    implementations, without any prefill. One file per checkpoint.

    Args:
        exploit_logprobs_dir: Directory containing checkpoint-{N}.jsonl files

    Returns:
        DataFrame with columns: task_id, checkpoint, exploit_type,
        exploit_logprob_sum, exploit_logprob_mean, exploit_num_tokens,
        truncated_logprob_sum, truncated_logprob_mean.
        Returns None if directory doesn't exist or no data found.
    """
    if not exploit_logprobs_dir.exists():
        return None

    rows = []
    for jsonl_file in sorted(exploit_logprobs_dir.glob("checkpoint-*.jsonl")):
        match = re.match(r"checkpoint-(\d+)\.jsonl$", jsonl_file.name)
        if match is None:
            continue
        checkpoint = int(match.group(1))
        with open(jsonl_file, "r") as f:
            for line in f:
                if line.strip():
                    record = json.loads(line)
                    rows.append({
                        "task_id": record.get("task_id"),
                        "checkpoint": checkpoint,
                        "exploit_type": record.get("exploit_type"),
                        "exploit_logprob_sum": record.get("exploit_logprob_sum"),
                        "exploit_logprob_mean": record.get("exploit_logprob_mean"),
                        "exploit_num_tokens": record.get("exploit_num_tokens"),
                        "truncated_logprob_sum": record.get("truncated_logprob_sum"),
                        "truncated_logprob_mean": record.get("truncated_logprob_mean"),
                    })

    if not rows:
        return None
    return pd.DataFrame(rows)


def load_kl_results(kl_dir: Path) -> pd.DataFrame | None:
    """Load KL divergence results from checkpoint × prefill KL files.

    Args:
        kl_dir: Directory containing checkpoint-*_prefill*_kl.jsonl files

    Returns:
        DataFrame with columns: task_id, checkpoint, prefill_tokens, exploit_type,
        attempt_idx, kl_divergence, kl_per_token, eval_logprob_sum, ref_logprob_sum,
        exploit_success. Returns None if directory doesn't exist or no data found.
    """
    if not kl_dir.exists():
        return None

    rows = []
    for jsonl_file in sorted(kl_dir.glob("checkpoint-*_prefill*_kl.jsonl")):
        # Parse checkpoint and prefill from filename
        match = re.match(r"checkpoint-(\d+)_prefill(\d+)_kl\.jsonl$", jsonl_file.name)
        if match is None:
            continue
        checkpoint, prefill_tokens = int(match.group(1)), int(match.group(2))
        with open(jsonl_file, "r") as f:
            for line in f:
                if line.strip():
                    record = json.loads(line)
                    rows.append({
                        "task_id": record.get("task_id"),
                        "checkpoint": checkpoint,
                        "prefill_tokens": prefill_tokens,
                        "exploit_type": record.get("exploit_type"),
                        "attempt_idx": record.get("attempt_idx", 0),
                        "kl_divergence": record.get("kl_divergence"),
                        "kl_per_token": record.get("kl_per_token"),
                        "kl_num_tokens": record.get("kl_num_tokens"),
                        "eval_logprob_sum": record.get("eval_logprob_sum"),
                        "ref_logprob_sum": record.get("ref_logprob_sum"),
                        "exploit_success": record.get("exploit_success", False),
                    })

    if not rows:
        return None
    return pd.DataFrame(rows)
