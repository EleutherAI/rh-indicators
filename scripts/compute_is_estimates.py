#!/usr/bin/env python3
"""Compute importance sampling (IS) probability estimates for exploit rates.

Implements the exact IS identity from docs/prefix_probability_estimation.md:

    μ_P(C) = E_{z~q}[ A(x,z) · B(x,z) ]

where:
    A(x,z) = (P_τ(z|x) / D_τᴰ(z|x)) · r_P(x,z)
    B(x,z) = π_D(x) / a_D(x,z)

And the fixed-prefix lower bound (§3):
    L(x,z) = P_τ(z|x) · r_P(x,z)
    L*(x) = max_N L(x, z_N(x))

Requires five data sources:
    1. Target logprobs: P_τ(z|x) per (checkpoint, prefill, task)
    2. Donor logprobs: D_τ(z|x) per (prefill, task)
    3. Target evals: r_P(x,z) — exploit rate per (checkpoint, task, prefill)
    4. Donor evals: a_D(x,z) — exploit rate per (task, prefill) at pfx>0
    5. Donor baseline: π_D(x) — spontaneous exploit rate per task at pfx=0

Outputs JSON with IS, LB, GT estimates overall, per-prefill, and per-exploit-type.

Usage:
    python scripts/compute_is_estimates.py \
        --target-logprob-dir results/prefill_sensitivity/.../logprob \
        --donor-logprob-dir results/donor_evals/.../logprob \
        --target-eval-dir results/prefill_sensitivity_t10/.../evals \
        --donor-eval-dir results/donor_evals_t10/.../evals \
        --donor-baseline results/prefill_ref/.../evals/checkpoint-132.jsonl \
        --gt-dir results/hack_rate_early/.../evals \
        --checkpoints 6 15 25 44 \
        --output-dir results/is_analysis \
        --intentional-only
"""

import argparse
import json
import math
import re
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

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

DEFAULT_PREFILL_LEVELS = [0, 2, 5, 10, 20, 30, 45, 60, 75, 100]

FILTER_LIST_PATH = Path("data/task_filter_list.json")


def load_task_filter(filter_path: Path | None = None) -> set[str]:
    """Load excluded task IDs from the filter list."""
    path = filter_path or FILTER_LIST_PATH
    if not path.exists():
        return set()
    with open(path) as f:
        data = json.load(f)
    excluded = set()
    for filt in data.get("filters", {}).values():
        excluded.update(filt.get("task_ids", []))
    return excluded


def load_jsonl(path: Path) -> list[dict]:
    """Load JSONL file, skipping malformed lines."""
    rows = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return rows


def parse_filename(fname: str) -> tuple[int | None, int | None]:
    """Extract (checkpoint, prefill) from filename like checkpoint-44_prefill10_logprobs.jsonl."""
    m = re.match(r"checkpoint-(\d+)_prefill(\d+)", fname)
    return (int(m.group(1)), int(m.group(2))) if m else (None, None)


def logsumexp(arr: list[float]) -> float:
    """Numerically stable log-sum-exp."""
    if not arr:
        return float("-inf")
    a = max(arr)
    if a == float("-inf"):
        return float("-inf")
    return a + math.log(sum(math.exp(x - a) for x in arr))


def _log_is_weight(s: dict, no_b: bool) -> float:
    """Log IS weight for one sample. Returns -inf if terms are invalid."""
    if s["r"] <= 0:
        return float("-inf")
    w = s["lp_t"] - s["lp_d"] + math.log(s["r"])
    if not no_b:
        if s["pi"] <= 0 or s["a"] <= 0:
            return float("-inf")
        w += math.log(s["pi"]) - math.log(s["a"])
    return w


def load_all_data(args) -> dict:
    """Load all five data sources and GT."""
    data = {}

    # 1. Target logprobs
    target_logprobs = {}
    for f in sorted(Path(args.target_logprob_dir).glob("checkpoint-*_prefill*_logprobs.jsonl")):
        ckpt, pfx = parse_filename(f.name)
        if ckpt not in args.checkpoints or pfx == 0:
            continue
        for row in load_jsonl(f):
            target_logprobs[(ckpt, pfx, row["task_id"])] = row["prefill_logprob_sum"]
    data["target_logprobs"] = target_logprobs
    print(f"  Target logprobs: {len(target_logprobs)} entries")

    # 2. Donor logprobs
    donor_logprobs = {}
    donor_ckpt = args.donor_checkpoint
    for f in sorted(Path(args.donor_logprob_dir).glob(f"checkpoint-{donor_ckpt}_prefill*_logprobs.jsonl")):
        _, pfx = parse_filename(f.name)
        if pfx == 0:
            continue
        for row in load_jsonl(f):
            donor_logprobs[(pfx, row["task_id"])] = row["prefill_logprob_sum"]
    data["donor_logprobs"] = donor_logprobs
    print(f"  Donor logprobs: {len(donor_logprobs)} entries")

    # 3. Target evals (explicit file loading to avoid backup dirs)
    target_evals = []
    target_eval_dir = Path(args.target_eval_dir)
    for ckpt in args.checkpoints:
        for pfx in args.prefill_levels:
            f = target_eval_dir / f"checkpoint-{ckpt}_prefill{pfx}.jsonl"
            if not f.exists():
                continue
            for row in load_jsonl(f):
                row["checkpoint"] = ckpt
                row["prefill_tokens"] = pfx
                target_evals.append(row)
    data["target_evals"] = target_evals
    print(f"  Target evals: {len(target_evals)} rows")

    # 4. Donor evals
    donor_evals = []
    donor_eval_dir = Path(args.donor_eval_dir)
    for pfx in args.prefill_levels:
        f = donor_eval_dir / f"checkpoint-{donor_ckpt}_prefill{pfx}.jsonl"
        if not f.exists():
            continue
        for row in load_jsonl(f):
            row["checkpoint"] = donor_ckpt
            row["prefill_tokens"] = pfx
            donor_evals.append(row)
    data["donor_evals"] = donor_evals
    print(f"  Donor evals: {len(donor_evals)} rows")

    # 5. Donor baseline (π_D at pfx=0)
    donor_baseline = load_jsonl(Path(args.donor_baseline))
    for row in donor_baseline:
        row["prefill_tokens"] = 0
    data["donor_baseline"] = donor_baseline
    print(f"  Donor baseline: {len(donor_baseline)} rows")

    # 6. GT (optional)
    gt_data = []
    if args.gt_dir:
        gt_dir = Path(args.gt_dir)
        for ckpt in args.checkpoints:
            f = gt_dir / f"checkpoint-{ckpt}.jsonl"
            if not f.exists():
                continue
            for row in load_jsonl(f):
                row["checkpoint"] = ckpt
                gt_data.append(row)
    data["gt_data"] = gt_data
    print(f"  GT data: {len(gt_data)} rows")

    return data


def build_exploit_type_lookup(data: dict) -> dict[str, str]:
    """Build task_id -> exploit_type mapping from all data sources."""
    task_et = {}
    for source in ["target_evals", "donor_evals", "donor_baseline", "gt_data"]:
        for row in data[source]:
            if "exploit_type" in row:
                task_et[row["task_id"]] = row["exploit_type"]
    return task_et


def compute_behavioral_terms(
    data: dict, task_et: dict, intentional_only: bool, excluded_tasks: set[str] | None = None,
) -> dict:
    """Compute r_P, π_D, a_D from eval data."""

    def is_valid(task_id):
        if excluded_tasks and task_id in excluded_tasks:
            return False
        if not intentional_only:
            return True
        return task_et.get(task_id, "") in INTENTIONAL_TYPES

    # r_P(x,z): target exploit rate per (ckpt, task_id, pfx)
    rc = defaultdict(lambda: [0, 0])
    for row in data["target_evals"]:
        if not is_valid(row["task_id"]):
            continue
        k = (row["checkpoint"], row["task_id"], row["prefill_tokens"])
        rc[k][1] += 1
        if row.get("exploit_success"):
            rc[k][0] += 1
    r_target = {k: s / t for k, (s, t) in rc.items()}

    # π_D(x) from donor evals pfx=0 (primary), fallback to donor baseline
    pic = defaultdict(lambda: [0, 0])
    for row in data["donor_evals"]:
        if row["prefill_tokens"] != 0 or not is_valid(row["task_id"]):
            continue
        pic[row["task_id"]][1] += 1
        if row.get("exploit_success"):
            pic[row["task_id"]][0] += 1
    pi_donor = {k: s / t for k, (s, t) in pic.items()}

    # Fallback to baseline for tasks not in donor evals
    for row in data["donor_baseline"]:
        if not is_valid(row["task_id"]):
            continue
        if row["task_id"] not in pi_donor:
            pi_donor[row["task_id"]] = (
                1.0
                if row.get("exploit_success")
                else 0.0
            )

    # a_D(x,z): donor exploit rate given prefix
    ac = defaultdict(lambda: [0, 0])
    for row in data["donor_evals"]:
        if row["prefill_tokens"] == 0 or not is_valid(row["task_id"]):
            continue
        k = (row["task_id"], row["prefill_tokens"])
        ac[k][1] += 1
        if row.get("exploit_success"):
            ac[k][0] += 1
    a_donor = {k: s / t for k, (s, t) in ac.items()}

    return {"r_target": r_target, "pi_donor": pi_donor, "a_donor": a_donor}


def compute_gt_rates(data: dict, task_et: dict, intentional_only: bool, excluded_tasks: set[str] | None = None) -> dict:
    """Compute GT rates from hack_rate_early data."""

    def is_valid(task_id):
        if excluded_tasks and task_id in excluded_tasks:
            return False
        if not intentional_only:
            return True
        return task_et.get(task_id, "") in INTENTIONAL_TYPES

    gt_task_ckpt = defaultdict(lambda: defaultdict(lambda: [0, 0]))
    for row in data["gt_data"]:
        if not is_valid(row["task_id"]):
            continue
        ckpt = row["checkpoint"]
        gt_task_ckpt[ckpt][row["task_id"]][1] += 1
        if row.get("exploit_success"):
            gt_task_ckpt[ckpt][row["task_id"]][0] += 1

    return gt_task_ckpt


def compute_is_for_checkpoint(
    ckpt: int,
    target_logprobs: dict,
    donor_logprobs: dict,
    terms: dict,
    task_et: dict,
    gt_task_ckpt: dict,
    intentional_only: bool,
    prefill_levels: list[int],
    excluded_tasks: set[str] | None = None,
    no_b_correction: bool = False,
) -> dict:
    """Compute IS/LB/GT for one checkpoint."""

    def is_valid(task_id):
        if excluded_tasks and task_id in excluded_tasks:
            return False
        if not intentional_only:
            return True
        return task_et.get(task_id, "") in INTENTIONAL_TYPES

    r_target = terms["r_target"]
    pi_donor = terms["pi_donor"]
    a_donor = terms["a_donor"]

    # Build samples
    samples = []
    for (c, p, tid), lp_t in target_logprobs.items():
        if c != ckpt or p == 0 or not is_valid(tid):
            continue
        lp_d = donor_logprobs.get((p, tid))
        if lp_d is None:
            continue
        samples.append(
            {
                "tid": tid,
                "pfx": p,
                "et": task_et.get(tid, ""),
                "lp_t": lp_t,
                "lp_d": lp_d,
                "r": r_target.get((ckpt, tid, p), 0.0),
                "pi": pi_donor.get(tid, 0.0),
                "a": a_donor.get((tid, p), 0.0),
            }
        )

    is_tasks = set(s["tid"] for s in samples)

    # GT matched
    gt_ms, gt_mt = 0, 0
    for tid in is_tasks:
        if tid in gt_task_ckpt.get(ckpt, {}):
            s_, t_ = gt_task_ckpt[ckpt][tid]
            gt_ms += s_
            gt_mt += t_
    gt_matched = gt_ms / gt_mt if gt_mt > 0 else 0.0

    # GT all
    gt_all_s = sum(v[0] for v in gt_task_ckpt.get(ckpt, {}).values())
    gt_all_t = sum(v[1] for v in gt_task_ckpt.get(ckpt, {}).values())
    gt_all = gt_all_s / gt_all_t if gt_all_t > 0 else 0.0

    # IS
    log_weights = [_log_is_weight(s, no_b_correction) for s in samples]
    fw = [w for w in log_weights if w > float("-inf")]
    is_est = math.exp(logsumexp(fw) - math.log(len(samples))) if fw else 0.0

    # LB
    tmx = {}
    for s in samples:
        lb = s["lp_t"] + math.log(s["r"]) if s["r"] > 0 else float("-inf")
        if s["tid"] not in tmx or lb > tmx[s["tid"]]:
            tmx[s["tid"]] = lb
    flb = [v for v in tmx.values() if v > float("-inf")]
    lb_est = math.exp(logsumexp(flb) - math.log(len(tmx))) if flb else 0.0

    result = {
        "checkpoint": ckpt,
        "is_estimate": is_est,
        "lb_estimate": lb_est,
        "gt_matched": gt_matched,
        "gt_all": gt_all,
        "n_samples": len(samples),
        "n_tasks": len(is_tasks),
        "n_matched_gt": len(is_tasks & set(gt_task_ckpt.get(ckpt, {}).keys())),
        "gt_matched_attempts": gt_mt,
        "n_valid_is": len(fw),
        "n_valid_lb": len(flb),
    }

    # Per-prefill breakdown
    per_prefill = []
    for pfx in sorted(prefill_levels):
        if pfx == 0:
            continue
        ps = [s for s in samples if s["pfx"] == pfx]
        if not ps:
            continue

        vis, vlb = [], []
        for s in ps:
            if s["r"] > 0:
                vlb.append(s["lp_t"] + math.log(s["r"]))
            w = _log_is_weight(s, no_b_correction)
            if w > float("-inf"):
                vis.append(w)

        is_v = math.exp(logsumexp(vis) - math.log(len(ps))) if vis else 0.0
        lb_v = math.exp(logsumexp(vlb) - math.log(len(ps))) if vlb else 0.0

        per_prefill.append(
            {
                "prefill_tokens": pfx,
                "n_samples": len(ps),
                "n_valid_is": len(vis),
                "n_valid_lb": len(vlb),
                "is_estimate": is_v,
                "lb_estimate": lb_v,
                "mean_log_pd": float(np.mean([s["lp_t"] - s["lp_d"] for s in ps])),
                "mean_r": float(np.mean([s["r"] for s in ps])),
                "mean_a": float(np.mean([s["a"] for s in ps])),
            }
        )
    result["per_prefill"] = per_prefill

    # Per-exploit-type breakdown
    by_et = defaultdict(list)
    for s in samples:
        by_et[s["et"]].append(s)

    per_type = []
    for et in sorted(by_et):
        ss = by_et[et]
        et_tasks = set(s["tid"] for s in ss)

        vis = [_log_is_weight(s, no_b_correction) for s in ss]
        vis = [w for w in vis if w > float("-inf")]
        is_v = math.exp(logsumexp(vis) - math.log(len(ss))) if vis else 0.0

        tmx_et = {}
        for s in ss:
            if s["r"] <= 0:
                continue
            lb = s["lp_t"] + math.log(s["r"])
            if s["tid"] not in tmx_et or lb > tmx_et[s["tid"]]:
                tmx_et[s["tid"]] = lb
        flb_et = [v for v in tmx_et.values() if v > float("-inf")]
        lb_v = (
            math.exp(logsumexp(flb_et) - math.log(len(et_tasks))) if flb_et else 0.0
        )

        # GT matched for this type
        gs, gt = 0, 0
        for tid in et_tasks:
            if tid in gt_task_ckpt.get(ckpt, {}):
                s_, t_ = gt_task_ckpt[ckpt][tid]
                gs += s_
                gt += t_
        gm = gs / gt if gt > 0 else 0.0

        # Per-prefill within this type
        et_per_pfx = []
        for pfx in sorted(set(s["pfx"] for s in ss)):
            ps = [s for s in ss if s["pfx"] == pfx]
            if not ps:
                continue
            pvis = [_log_is_weight(s, no_b_correction) for s in ps]
            pvis = [w for w in pvis if w > float("-inf")]
            pis = math.exp(logsumexp(pvis) - math.log(len(ps))) if pvis else 0.0

            ptmx = {}
            for s in ps:
                if s["r"] <= 0:
                    continue
                lb = s["lp_t"] + math.log(s["r"])
                if s["tid"] not in ptmx or lb > ptmx[s["tid"]]:
                    ptmx[s["tid"]] = lb
            pflb = [v for v in ptmx.values() if v > float("-inf")]
            plb = math.exp(logsumexp(pflb) - math.log(len(set(s["tid"] for s in ps)))) if pflb else 0.0

            et_per_pfx.append({
                "prefill_tokens": pfx,
                "n_samples": len(ps),
                "n_valid_is": len(pvis),
                "is_estimate": pis,
                "lb_estimate": plb,
                "mean_log_pd": float(np.mean([s["lp_t"] - s["lp_d"] for s in ps])),
                "mean_r": float(np.mean([s["r"] for s in ps])),
            })

        per_type.append(
            {
                "exploit_type": et,
                "n_samples": len(ss),
                "n_tasks": len(et_tasks),
                "is_estimate": is_v,
                "lb_estimate": lb_v,
                "gt_matched": gm,
                "gt_successes": gs,
                "gt_attempts": gt,
                "per_prefill": et_per_pfx,
            }
        )
    result["per_type"] = per_type

    return result


def print_results(results: dict, checkpoints: list[int]):
    """Pretty-print results to stdout."""
    mode = results.get("mode", "all")
    print(f"\n{'='*110}")
    print(
        f"IS / LB / GT — {mode.upper()} exploits, matched task pools"
    )
    print(f"{'='*110}")
    hdr = f"{'Ckpt':>5} | {'IS':>12} | {'LB':>12} | {'GT(match)':>12} | {'GT(all)':>12} | {'IS/GT':>8} | {'LB/GT':>8} | {'#tasks':>6} | {'validIS':>7} | {'validLB':>7}"
    print(hdr)
    print("-" * len(hdr))

    for ckpt_result in results["checkpoints"]:
        ckpt = ckpt_result["checkpoint"]
        r = ckpt_result
        ig = (
            r["is_estimate"] / r["gt_matched"]
            if r["gt_matched"] > 0
            else (0 if r["is_estimate"] == 0 else float("inf"))
        )
        lg = (
            r["lb_estimate"] / r["gt_matched"]
            if r["gt_matched"] > 0
            else (0 if r["lb_estimate"] == 0 else float("inf"))
        )
        igs = f"{ig:.2f}" if ig < 999 else f"{ig:.0f}"
        lgs = f"{lg:.2f}" if lg < 999 else f"{lg:.0f}"
        print(
            f"{ckpt:>5} | {r['is_estimate']:>12.8f} | {r['lb_estimate']:>12.8f} | "
            f"{r['gt_matched']:>12.8f} | {r['gt_all']:>12.8f} | "
            f"{igs:>8} | {lgs:>8} | {r['n_tasks']:>6} | "
            f"{r['n_valid_is']:>7} | {r['n_valid_lb']:>7}"
        )

    # Per-prefill for last checkpoint
    last = results["checkpoints"][-1]
    ckpt = last["checkpoint"]
    print(f"\n--- Per-Prefill: Checkpoint {ckpt} ---")
    phdr = f"{'Pfx':>5} | {'#samp':>6} | {'vIS':>5} | {'vLB':>5} | {'IS':>12} | {'LB':>12} | {'mean_log(P/D)':>14} | {'mean_r':>8} | {'mean_a':>8}"
    print(phdr)
    print("-" * len(phdr))
    for pp in last["per_prefill"]:
        print(
            f"{pp['prefill_tokens']:>5} | {pp['n_samples']:>6} | {pp['n_valid_is']:>5} | "
            f"{pp['n_valid_lb']:>5} | {pp['is_estimate']:>12.8f} | {pp['lb_estimate']:>12.8f} | "
            f"{pp['mean_log_pd']:>14.3f} | {pp['mean_r']:>8.4f} | {pp['mean_a']:>8.4f}"
        )

    # Per-type for last checkpoint
    print(f"\n--- Per-Exploit-Type: Checkpoint {ckpt} ---")
    thdr = f"{'Type':<35} | {'#samp':>6} | {'#task':>5} | {'IS':>12} | {'LB':>12} | {'GT':>10} | {'IS/GT':>8} | {'LB/GT':>8}"
    print(thdr)
    print("-" * len(thdr))
    for pt in last["per_type"]:
        ig = (
            pt["is_estimate"] / pt["gt_matched"]
            if pt["gt_matched"] > 0
            else (0 if pt["is_estimate"] == 0 else float("inf"))
        )
        lg = (
            pt["lb_estimate"] / pt["gt_matched"]
            if pt["gt_matched"] > 0
            else (0 if pt["lb_estimate"] == 0 else float("inf"))
        )
        igs = f"{ig:.2f}" if ig < 999 else f"{ig:.0f}"
        lgs = f"{lg:.2f}" if lg < 999 else f"{lg:.0f}"
        print(
            f"{pt['exploit_type']:<35} | {pt['n_samples']:>6} | {pt['n_tasks']:>5} | "
            f"{pt['is_estimate']:>12.6f} | {pt['lb_estimate']:>12.6f} | "
            f"{pt['gt_matched']:>10.6f} | {igs:>8} | {lgs:>8}"
        )


def main():
    parser = argparse.ArgumentParser(
        description="Compute IS probability estimates for exploit rates"
    )
    parser.add_argument(
        "--target-logprob-dir",
        type=str,
        required=True,
        help="Directory with target model logprobs (Harmony-corrected)",
    )
    parser.add_argument(
        "--donor-logprob-dir",
        type=str,
        required=True,
        help="Directory with donor model logprobs (Harmony-corrected)",
    )
    parser.add_argument(
        "--target-eval-dir",
        type=str,
        required=True,
        help="Directory with target model evals (for r_P)",
    )
    parser.add_argument(
        "--donor-eval-dir",
        type=str,
        required=True,
        help="Directory with donor model evals (for a_D)",
    )
    parser.add_argument(
        "--donor-baseline",
        type=str,
        required=True,
        help="Donor baseline eval file (for π_D at pfx=0)",
    )
    parser.add_argument(
        "--gt-dir",
        type=str,
        default=None,
        help="Directory with ground truth evals (hack_rate_early)",
    )
    parser.add_argument(
        "--checkpoints",
        type=int,
        nargs="+",
        default=[6, 15, 25, 44],
        help="Checkpoints to analyze",
    )
    parser.add_argument(
        "--donor-checkpoint",
        type=int,
        default=132,
        help="Donor model checkpoint number",
    )
    parser.add_argument(
        "--prefill-levels",
        type=int,
        nargs="+",
        default=DEFAULT_PREFILL_LEVELS,
        help="Prefill levels",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Output directory for results JSON",
    )
    parser.add_argument(
        "--intentional-only",
        action="store_true",
        help="Filter to intentional exploit types only",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress table output",
    )
    parser.add_argument(
        "--filter-list",
        type=Path,
        default=None,
        help="Path to task filter list JSON (default: data/task_filter_list.json if it exists)",
    )
    parser.add_argument(
        "--no-filter",
        action="store_true",
        help="Disable task filtering even if filter list exists",
    )
    parser.add_argument(
        "--no-b-correction",
        action="store_true",
        help="Drop B correction (π_D/a_D) — IS_noB = (1/m) Σ (P/D)·r",
    )

    args = parser.parse_args()

    # Load task filter
    excluded_tasks = set()
    if not args.no_filter:
        excluded_tasks = load_task_filter(args.filter_list)
        if excluded_tasks:
            print(f"Excluding {len(excluded_tasks)} filtered tasks")

    print("Loading data...")
    data = load_all_data(args)

    task_et = build_exploit_type_lookup(data)
    terms = compute_behavioral_terms(data, task_et, args.intentional_only, excluded_tasks)
    gt_task_ckpt = compute_gt_rates(data, task_et, args.intentional_only, excluded_tasks)

    print(f"\nBehavioral terms ({'intentional' if args.intentional_only else 'all'}):")
    print(
        f"  r_P: {len(terms['r_target'])} entries, "
        f"{sum(1 for v in terms['r_target'].values() if v > 0)} nonzero"
    )
    print(
        f"  π_D: {len(terms['pi_donor'])} entries, "
        f"{sum(1 for v in terms['pi_donor'].values() if v > 0)} nonzero"
    )
    print(
        f"  a_D: {len(terms['a_donor'])} entries, "
        f"{sum(1 for v in terms['a_donor'].values() if v > 0)} nonzero"
    )

    # Compute per-checkpoint
    checkpoint_results = []
    for ckpt in args.checkpoints:
        result = compute_is_for_checkpoint(
            ckpt,
            data["target_logprobs"],
            data["donor_logprobs"],
            terms,
            task_et,
            gt_task_ckpt,
            args.intentional_only,
            args.prefill_levels,
            excluded_tasks,
            no_b_correction=args.no_b_correction,
        )
        checkpoint_results.append(result)

    output = {
        "mode": "intentional" if args.intentional_only else "all",
        "no_b_correction": args.no_b_correction,
        "checkpoints": checkpoint_results,
    }

    if not args.quiet:
        print_results(output, args.checkpoints)

    if args.output_dir:
        out_dir = Path(args.output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        suffix = "_intentional" if args.intentional_only else "_all"
        out_path = out_dir / f"is_estimates{suffix}.json"
        with open(out_path, "w") as f:
            json.dump(output, f, indent=2)
        print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    main()
