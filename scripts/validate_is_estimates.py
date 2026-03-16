#!/usr/bin/env python3
"""Validate IS probability estimates against ground truth.

Loads K=1000 logprob files, applies temperature correction (T=1 → T=0.4),
and computes fixed-prefix lower bounds. Compares with 64-attempt ground truth.

Ground truth (T=1.0, 64 attempts, no prefill):
    ckpt-6:  0.61%
    ckpt-15: 0.70%
    ckpt-25: 0.03%
    ckpt-44: 2.31%

Usage:
    python scripts/validate_is_estimates.py
    python scripts/validate_is_estimates.py --verbose
    python scripts/validate_is_estimates.py --checkpoints 6,15
"""

import argparse
import json
import math
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from rh_indicators.trajectory.temperature import (
    temperature_correct_sequence,
    validate_temperature_correction,
)


GROUND_TRUTH = {
    6: 0.0061,
    15: 0.0070,
    25: 0.0003,
    44: 0.0231,
}


def load_logprob_file(path: Path) -> list[dict]:
    """Load a single logprob JSONL file."""
    records = []
    with open(path) as f:
        for line in f:
            records.append(json.loads(line))
    return records


def temperature_correct_records(records: list[dict], temperature: float) -> list[dict]:
    """Apply temperature correction to all records, adding corrected logprob fields."""
    out = []
    for rec in records:
        if "error" in rec or "prefill_logprob_sum" not in rec:
            continue  # Skip failed API calls

        token_logprobs = rec.get("token_logprobs", [])
        top_logprobs = rec.get("top_logprobs", [])

        if not token_logprobs or not top_logprobs:
            rec["prefill_logprob_sum_corrected"] = rec["prefill_logprob_sum"]
            out.append(rec)
            continue

        sum_lp, mean_lp, _ = temperature_correct_sequence(
            token_logprobs, top_logprobs, temperature
        )
        rec["prefill_logprob_sum_corrected"] = sum_lp
        rec["prefill_logprob_mean_corrected"] = mean_lp
        out.append(rec)

    return out


def compute_lower_bound(records_by_prefill: dict[int, list[dict]]) -> dict:
    """Compute fixed-prefix lower bound.

    L(x, z_N) = P_τ(z_N|x) · r(x, z_N)
    L*(x) = max_N L(x, z_N)
    μ̂ = (1/n_tasks) Σ_x L*(x)

    Since these are single-attempt samples, r(x, z_N) = exploit_success ∈ {0, 1}.
    For r=1 samples: L(x, z_N) = P_τ(z_N|x) = exp(logprob_sum_corrected).
    """
    # Group by task_id across all prefill levels
    task_best: dict[str, dict] = {}  # task_id → {log_lb, prefill, log_p}

    for pfx, records in records_by_prefill.items():
        for rec in records:
            tid = rec["task_id"]
            r = float(rec.get("exploit_success", 0))

            if r <= 0:
                # No exploit → contributes 0 to lower bound
                if tid not in task_best:
                    task_best[tid] = {
                        "log_lb": float("-inf"),
                        "prefill": 0,
                        "log_p": float("-inf"),
                    }
                continue

            log_p = rec.get("prefill_logprob_sum_corrected", rec["prefill_logprob_sum"])
            log_lb = log_p  # + log(1) = log_p since r=1

            if tid not in task_best or log_lb > task_best[tid]["log_lb"]:
                task_best[tid] = {
                    "log_lb": log_lb,
                    "prefill": pfx,
                    "log_p": log_p,
                }

    n_tasks = len(task_best)
    finite = [v["log_lb"] for v in task_best.values() if v["log_lb"] > float("-inf")]
    n_exploiting = len(finite)

    if finite:
        arr = np.array(finite, dtype=np.float64)
        max_val = arr.max()
        log_avg = max_val + np.log(np.sum(np.exp(arr - max_val))) - np.log(n_tasks)
        avg_lb = math.exp(float(log_avg))
    else:
        log_avg = float("-inf")
        avg_lb = 0.0

    return {
        "lower_bound": avg_lb,
        "log_lower_bound": float(log_avg),
        "n_tasks": n_tasks,
        "n_exploiting": n_exploiting,
        "task_results": task_best,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--logprob-dir", type=Path,
                        default=Path("results/is_logprobs/gpt-oss-20b"))
    parser.add_argument("--temperature", type=float, default=0.4)
    parser.add_argument("--checkpoints", type=str, default="6,15,25,44")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    checkpoints = [int(c) for c in args.checkpoints.split(",")]
    tau = args.temperature

    print(f"Temperature correction: T=1.0 → T={tau}")
    print(f"Checkpoints: {checkpoints}")
    print()

    all_results = {}

    for ckpt in checkpoints:
        ckpt_dir = args.logprob_dir / f"checkpoint-{ckpt}"
        if not ckpt_dir.exists():
            print(f"  WARNING: {ckpt_dir} not found, skipping")
            continue

        print(f"{'='*60}")
        print(f"Checkpoint {ckpt}")
        print(f"{'='*60}")

        # Load all prefill level files
        records_by_prefill: dict[int, list[dict]] = {}
        for f in sorted(ckpt_dir.glob("*_logprobs.jsonl")):
            records = load_logprob_file(f)
            if not records:
                continue
            pfx = records[0]["prefill_tokens"]

            # Validate K=1000 coverage
            if args.verbose and records:
                sample_top = records[0].get("top_logprobs", [])
                if sample_top:
                    val = validate_temperature_correction(sample_top, tau)
                    print(f"  {f.name}: {len(records)} samples, K~{len(sample_top[0]) if sample_top else 0}, "
                          f"coverage={val['min_coverage']:.10f}")

            # Apply temperature correction
            records = temperature_correct_records(records, tau)
            records_by_prefill[pfx] = records

            # Show correction magnitude
            if records:
                orig = np.mean([r["prefill_logprob_sum"] for r in records])
                corr = np.mean([r["prefill_logprob_sum_corrected"] for r in records])
                n_exploit = sum(1 for r in records if r.get("exploit_success", 0))
                print(f"  prefill={pfx:>3}: {len(records):>3} samples, "
                      f"{n_exploit:>3} exploits, "
                      f"mean_logP: T=1 {orig:>8.1f} → T={tau} {corr:>8.1f}")

        if not records_by_prefill:
            continue

        # Compute lower bound
        result = compute_lower_bound(records_by_prefill)
        gt = GROUND_TRUTH.get(ckpt)

        print(f"\n  Lower bound estimate: {result['lower_bound']:.6f} ({result['lower_bound']*100:.4f}%)")
        print(f"  Tasks: {result['n_tasks']} total, {result['n_exploiting']} with exploit")
        if gt is not None:
            print(f"  Ground truth:         {gt:.6f} ({gt*100:.4f}%)")
            if gt > 0:
                ratio = result["lower_bound"] / gt
                print(f"  Ratio (est/gt):       {ratio:.3f}")

        # Show per-prefill lower bounds
        if args.verbose:
            print(f"\n  Per-prefill lower bounds:")
            for pfx in sorted(records_by_prefill.keys()):
                pfx_result = compute_lower_bound({pfx: records_by_prefill[pfx]})
                print(f"    prefill={pfx:>3}: {pfx_result['lower_bound']*100:.6f}% "
                      f"({pfx_result['n_exploiting']}/{pfx_result['n_tasks']} exploiting)")

        # Show which tasks/prefills contribute most
        if args.verbose and result["task_results"]:
            exploiting = [(tid, v) for tid, v in result["task_results"].items()
                          if v["log_lb"] > float("-inf")]
            exploiting.sort(key=lambda x: x[1]["log_lb"], reverse=True)
            print(f"\n  Top contributing tasks:")
            for tid, v in exploiting[:10]:
                print(f"    {tid}: log_lb={v['log_lb']:.2f} (P={math.exp(v['log_lb']):.6f}) "
                      f"at prefill={v['prefill']}")

        all_results[ckpt] = result
        print()

    # Summary table
    print(f"{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    print(f"{'Ckpt':>6} {'Lower Bound':>12} {'Ground Truth':>13} {'Ratio':>8} {'Exploiting':>11}")
    print(f"{'-'*6} {'-'*12} {'-'*13} {'-'*8} {'-'*11}")
    for ckpt in checkpoints:
        if ckpt not in all_results:
            continue
        r = all_results[ckpt]
        gt = GROUND_TRUTH.get(ckpt, 0)
        lb = r["lower_bound"]
        ratio = lb / gt if gt > 0 else float("nan")
        print(f"{ckpt:>6} {lb*100:>11.4f}% {gt*100:>12.4f}% {ratio:>8.3f} "
              f"{r['n_exploiting']:>4}/{r['n_tasks']}")


if __name__ == "__main__":
    main()
