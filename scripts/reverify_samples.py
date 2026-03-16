#!/usr/bin/env python3
"""Re-verify existing evaluation samples without regenerating model outputs.

Reads .samples.jsonl files, re-runs secure+insecure verification on the saved
code, and rewrites the result .jsonl files with corrected verdicts.

Usage:
    # Re-verify all checkpoints/prefills in a run
    python scripts/reverify_samples.py \
        --run-dir results/prefill_sensitivity/prefill_sensitivity-20260222-054740

    # Re-verify specific files
    python scripts/reverify_samples.py \
        --samples results/prefill_sensitivity/.../checkpoint-126_prefill100.jsonl.samples.jsonl

    # Dry run (show what would change, don't write)
    python scripts/reverify_samples.py --run-dir ... --dry-run

    # Only re-verify insecure (keep existing secure_pass)
    python scripts/reverify_samples.py --run-dir ... --insecure-only
"""

import argparse
import json
import sys
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from djinn.core.problem import Problem
from djinn.core.reward import calc_reward


def verify_one(task_id: str, exploit_type: str, code: str, function_name: str,
               test_cases, insecure_test_cases, order_dependent: bool,
               insecure_only: bool, old_secure_pass: bool):
    """Verify a single submission."""
    prob = Problem(
        id=task_id,
        description="",
        function_name=function_name,
        test_cases=test_cases,
        ground_truth="",
        exploit="",
        insecure_verifier_info="",
        exploit_explanation="",
        exploit_type=exploit_type,
        insecure_test_cases=insecure_test_cases,
        order_dependent=order_dependent,
    )

    if insecure_only:
        secure_pass = old_secure_pass
        secure_reward = 1.0 if secure_pass else 0.0
    else:
        secure_reward, secure_result = calc_reward(prob, code, mode="secure", return_result=True)
        secure_pass = secure_reward > 0.0

    insecure_reward, insecure_result = calc_reward(prob, code, mode="insecure", return_result=True)
    insecure_pass = insecure_reward > 0.0
    insecure_suspect = getattr(insecure_result, "suspect", None)
    insecure_suspect_patterns = getattr(insecure_result, "suspect_patterns", None)

    exploit_success = bool(insecure_pass and not secure_pass)
    reward_gap = insecure_reward - secure_reward

    return {
        "secure_pass": secure_pass,
        "insecure_pass": insecure_pass,
        "exploit_success": exploit_success,
        "reward_gap": reward_gap,
        "insecure_suspect": insecure_suspect,
        "insecure_suspect_patterns": insecure_suspect_patterns,
    }


def load_problem_data(dataset: str, split: str):
    """Load problem metadata from dataset, keyed by task_id."""
    from datasets import load_dataset
    ds = load_dataset(dataset, split=split)
    problems = {}
    for row in ds:
        problems[row["id"]] = {
            "function_name": row["function_name"],
            "test_cases": row["test_cases"],
            "insecure_test_cases": row.get("insecure_test_cases"),
            "order_dependent": row.get("order_dependent", True),
        }
    return problems


def process_samples_file(samples_path: Path, problems: dict, insecure_only: bool,
                         dry_run: bool, concurrency: int) -> dict:
    """Re-verify all samples in a file and rewrite the result .jsonl."""
    # Derive result path: foo.jsonl.samples.jsonl -> foo.jsonl
    result_path = Path(str(samples_path).replace(".samples.jsonl", ""))
    if not result_path.exists():
        print(f"  WARNING: Result file not found: {result_path}")
        return {"skipped": True}

    # Load samples
    samples = []
    with open(samples_path) as f:
        for line in f:
            line = line.strip()
            if line:
                samples.append(json.loads(line))

    # Load existing results (indexed by task_id + attempt_idx)
    old_results = {}
    old_results_order = []  # preserve original order
    with open(result_path) as f:
        for line in f:
            line = line.strip()
            if line:
                row = json.loads(line)
                key = (row["task_id"], row.get("attempt_idx", 0))
                old_results[key] = row
                old_results_order.append(key)

    if not samples:
        print(f"  No samples in {samples_path.name}")
        return {"skipped": True}

    # Prepare verification tasks
    tasks = []
    for sample in samples:
        task_id = sample["task_id"]
        code = sample.get("code", "")
        exploit_type = sample.get("exploit_type", "")
        attempt_idx = sample.get("attempt_idx", 0)

        prob_data = problems.get(task_id)
        if not prob_data:
            print(f"  WARNING: task_id {task_id} not found in dataset, skipping")
            continue

        key = (task_id, attempt_idx)
        old_row = old_results.get(key, {})

        tasks.append({
            "task_id": task_id,
            "attempt_idx": attempt_idx,
            "exploit_type": exploit_type,
            "code": code,
            "function_name": prob_data["function_name"],
            "test_cases": prob_data["test_cases"],
            "insecure_test_cases": prob_data["insecure_test_cases"],
            "order_dependent": prob_data["order_dependent"],
            "old_secure_pass": old_row.get("secure_pass", False),
        })

    # Run verification with thread pool (djinn verifier uses internal daemon
    # subprocesses, so threads just dispatch and wait on pipe I/O)
    new_verdicts = {}
    t0 = time.time()

    with ThreadPoolExecutor(max_workers=concurrency) as pool:
        futures = {}
        for task in tasks:
            key = (task["task_id"], task["attempt_idx"])
            fut = pool.submit(
                verify_one,
                task_id=task["task_id"],
                exploit_type=task["exploit_type"],
                code=task["code"],
                function_name=task["function_name"],
                test_cases=task["test_cases"],
                insecure_test_cases=task["insecure_test_cases"],
                order_dependent=task["order_dependent"],
                insecure_only=insecure_only,
                old_secure_pass=task["old_secure_pass"],
            )
            futures[fut] = key

        done = 0
        for fut in as_completed(futures):
            key = futures[fut]
            try:
                new_verdicts[key] = fut.result()
            except Exception as e:
                print(f"  ERROR verifying {key}: {e}")
                new_verdicts[key] = None
            done += 1
            if done % 50 == 0:
                elapsed = time.time() - t0
                print(f"    {done}/{len(tasks)} verified ({elapsed:.1f}s)")

    elapsed = time.time() - t0

    # Compute diff stats
    changes = Counter()
    for key, verdict in new_verdicts.items():
        if verdict is None:
            changes["error"] += 1
            continue
        old = old_results.get(key, {})
        if old.get("insecure_pass") != verdict["insecure_pass"]:
            changes["insecure_pass_changed"] += 1
        if old.get("exploit_success") != verdict["exploit_success"]:
            changes["exploit_success_changed"] += 1
        if old.get("secure_pass") != verdict["secure_pass"]:
            changes["secure_pass_changed"] += 1

    # Count new insecure passes
    new_insecure = sum(1 for v in new_verdicts.values() if v and v["insecure_pass"])
    new_exploits = sum(1 for v in new_verdicts.values() if v and v["exploit_success"])
    old_insecure = sum(1 for r in old_results.values() if r.get("insecure_pass"))
    old_exploits = sum(1 for r in old_results.values() if r.get("exploit_success"))

    print(f"  {len(tasks)} samples verified in {elapsed:.1f}s")
    print(f"  insecure_pass: {old_insecure} -> {new_insecure}")
    print(f"  exploit_success: {old_exploits} -> {new_exploits}")
    if changes:
        print(f"  Changes: {dict(changes)}")

    if dry_run:
        print(f"  DRY RUN: not writing {result_path.name}")
        return {"dry_run": True, "changes": dict(changes), "new_insecure": new_insecure}

    # Rewrite result file (preserving original order)
    new_rows = []
    for key in old_results_order:
        old_row = old_results[key]
        verdict = new_verdicts.get(key)
        if verdict is None:
            new_rows.append(old_row)
            continue
        updated = dict(old_row)
        updated["secure_pass"] = verdict["secure_pass"]
        updated["insecure_pass"] = verdict["insecure_pass"]
        updated["exploit_success"] = verdict["exploit_success"]
        updated["reward_gap"] = verdict["reward_gap"]
        updated["insecure_suspect"] = verdict["insecure_suspect"]
        updated["insecure_suspect_patterns"] = verdict["insecure_suspect_patterns"]
        new_rows.append(updated)

    # Also update the samples file with new verdicts
    new_samples = []
    for sample in samples:
        key = (sample["task_id"], sample.get("attempt_idx", 0))
        verdict = new_verdicts.get(key)
        if verdict is None:
            new_samples.append(sample)
            continue
        updated = dict(sample)
        updated["secure_pass"] = verdict["secure_pass"]
        updated["insecure_pass"] = verdict["insecure_pass"]
        updated["exploit_success"] = verdict["exploit_success"]
        updated["reward_gap"] = verdict["reward_gap"]
        new_samples.append(updated)

    with open(result_path, "w") as f:
        for row in new_rows:
            f.write(json.dumps(row) + "\n")

    with open(samples_path, "w") as f:
        for sample in new_samples:
            f.write(json.dumps(sample) + "\n")

    print(f"  Wrote {len(new_rows)} results to {result_path.name}")
    return {"changes": dict(changes), "new_insecure": new_insecure}


def main():
    parser = argparse.ArgumentParser(description="Re-verify existing evaluation samples")
    parser.add_argument("--run-dir", type=Path, help="Run directory containing evals/")
    parser.add_argument("--samples", type=Path, nargs="+", help="Specific .samples.jsonl files")
    parser.add_argument("--dataset", default="EleutherAI/djinn-problems-v0.9")
    parser.add_argument("--split", default="test_alternate")
    parser.add_argument("--insecure-only", action="store_true",
                        help="Only re-verify insecure (keep existing secure_pass)")
    parser.add_argument("--dry-run", action="store_true",
                        help="Show what would change without writing")
    parser.add_argument("--concurrency", type=int, default=8,
                        help="Number of concurrent verification threads")
    args = parser.parse_args()

    if not args.run_dir and not args.samples:
        parser.error("Provide --run-dir or --samples")

    # Collect samples files
    if args.samples:
        samples_files = args.samples
    else:
        evals_dir = args.run_dir / "evals"
        samples_files = sorted(evals_dir.glob("*.samples.jsonl"))

    if not samples_files:
        print("No samples files found")
        return

    print(f"Found {len(samples_files)} samples files")
    print(f"Loading dataset {args.dataset} split={args.split}...")
    problems = load_problem_data(args.dataset, args.split)
    print(f"Loaded {len(problems)} problems")

    if args.insecure_only:
        print("Mode: insecure-only (keeping existing secure_pass)")
    if args.dry_run:
        print("Mode: dry run")

    print()
    for samples_path in samples_files:
        label = samples_path.name.replace(".jsonl.samples.jsonl", "")
        print(f"Processing {label}...")
        process_samples_file(
            samples_path, problems,
            insecure_only=args.insecure_only,
            dry_run=args.dry_run,
            concurrency=args.concurrency,
        )
        print()


if __name__ == "__main__":
    main()
