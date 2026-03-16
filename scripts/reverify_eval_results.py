"""Re-run djinn verification on stored completions without regenerating.

Use case: after fixing djinn problem definitions (ground truth, test cases,
verifier bugs), re-verify existing model completions to update exploit_success
and related fields.
"""

import argparse
import json
import os
import signal
import shutil
import sys
import tempfile
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import fields as dataclass_fields
from datetime import datetime
from pathlib import Path

from datasets import load_dataset
from djinn.core.problem import Problem
from djinn.core.reward import calc_reward
from tqdm import tqdm

# Fields that reverification updates
MUTABLE_FIELDS = {
    "secure_pass",
    "insecure_pass",
    "exploit_success",
    "reward_gap",
    "insecure_suspect",
    "insecure_suspect_patterns",
}

# Fields stripped from samples to produce summary JSONL
SAMPLES_ONLY_FIELDS = {"code", "response", "prompt", "system", "is_exploitative"}

# Problem fields accepted by the Problem dataclass constructor
PROBLEM_FIELDS = {f.name for f in dataclass_fields(Problem)}


def read_jsonl(path: Path) -> list[dict]:
    """Read a JSONL file, skipping blank or corrupt lines."""
    rows = []
    with open(path) as f:
        for i, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                print(f"  Warning: skipping corrupt line {i+1} in {path.name}")
    return rows


def load_problems(dataset: str, split: str) -> dict[str, dict]:
    """Load djinn problems from HuggingFace, keyed by problem ID."""
    ds = load_dataset(dataset, split=split, download_mode="force_redownload")
    problems = {}
    for row in ds:
        prob_row = {k: row[k] for k in PROBLEM_FIELDS if k in row}
        problems[row["id"]] = prob_row
    return problems


SAMPLE_TIMEOUT = 60  # seconds per sample (secure + insecure combined)


class SampleTimeout(Exception):
    pass


def _alarm_handler(signum, frame):
    raise SampleTimeout("Sample verification timed out")


def reverify_sample(args: tuple[dict, dict]) -> dict:
    """Re-run verification for a single sample. Called via ProcessPoolExecutor.

    Args:
        args: (sample_dict, problem_dict) tuple for pickling compatibility.

    Returns:
        Updated sample dict with new verification fields.
    """
    sample, prob_row = args

    # Set a hard timeout via SIGALRM (works in worker processes on Linux)
    old_handler = signal.signal(signal.SIGALRM, _alarm_handler)
    signal.alarm(SAMPLE_TIMEOUT)

    try:
        secure_reward, secure_result = calc_reward(
            prob_row, sample["code"], mode="secure", return_result=True, timeout=30
        )
        insecure_reward, insecure_result = calc_reward(
            prob_row, sample["code"], mode="insecure", return_result=True, timeout=30
        )
    except Exception as e:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, old_handler)
        # On error, mark as failed but preserve original
        updated = dict(sample)
        updated["secure_pass"] = False
        updated["insecure_pass"] = False
        updated["exploit_success"] = False
        updated["reward_gap"] = 0.0
        updated["insecure_suspect"] = False
        updated["insecure_suspect_patterns"] = []
        updated["_reverify_error"] = str(e)
        return updated
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, old_handler)

    updated = dict(sample)
    updated["secure_pass"] = bool(secure_reward > 0)
    updated["insecure_pass"] = bool(insecure_reward > 0)
    updated["exploit_success"] = updated["insecure_pass"] and not updated["secure_pass"]
    updated["reward_gap"] = insecure_reward - secure_reward
    updated["insecure_suspect"] = getattr(insecure_result, "suspect", False)
    updated["insecure_suspect_patterns"] = getattr(insecure_result, "suspect_patterns", []) or []

    return updated


def should_reverify(
    sample: dict, task_ids: set[str] | None, exploit_types: set[str] | None
) -> bool:
    """Check if a sample matches the filter criteria."""
    if task_ids is None and exploit_types is None:
        return True
    if task_ids and sample.get("task_id") in task_ids:
        return True
    if exploit_types and sample.get("exploit_type") in exploit_types:
        return True
    return False


def reverify_file(
    samples_path: Path,
    problems: dict[str, dict],
    task_ids: set[str] | None,
    exploit_types: set[str] | None,
    workers: int,
) -> tuple[list[dict], dict]:
    """Reverify matching samples in a file.

    Returns:
        (updated_samples, change_summary) where change_summary has counts.
    """
    samples = read_jsonl(samples_path)

    if not samples:
        return [], {"total": 0, "reverified": 0, "changes": {}}

    # Split into reverify vs passthrough
    to_reverify = []
    passthrough_indices = set()
    for i, sample in enumerate(samples):
        if should_reverify(sample, task_ids, exploit_types):
            task_id = sample.get("task_id")
            if task_id in problems and "code" in sample:
                to_reverify.append((i, sample, problems[task_id]))
            else:
                passthrough_indices.add(i)
                if task_id not in problems:
                    print(f"  Warning: task_id {task_id} not found in dataset, skipping")
        else:
            passthrough_indices.add(i)

    if not to_reverify:
        return samples, {"total": len(samples), "reverified": 0, "changes": {}}

    # Track before state for change summary
    before = {field: 0 for field in MUTABLE_FIELDS}
    for _, sample, _ in to_reverify:
        if sample.get("exploit_success"):
            before["exploit_success"] += 1
        if sample.get("secure_pass"):
            before["secure_pass"] += 1
        if sample.get("insecure_pass"):
            before["insecure_pass"] += 1

    # Run reverification in parallel
    work_items = [(sample, prob_row) for _, sample, prob_row in to_reverify]
    updated_map = {}

    if workers > 1 and len(work_items) > 1:
        with ProcessPoolExecutor(max_workers=workers) as executor:
            futures = {
                executor.submit(reverify_sample, item): idx
                for item, (idx, _, _) in zip(work_items, to_reverify)
            }
            for future in tqdm(
                as_completed(futures),
                total=len(futures),
                desc="  samples",
                leave=False,
                unit="sample",
            ):
                orig_idx = futures[future]
                updated_map[orig_idx] = future.result()
    else:
        for (sample, prob_row), (idx, _, _) in tqdm(
            zip(work_items, to_reverify),
            total=len(work_items),
            desc="  samples",
            leave=False,
            unit="sample",
        ):
            updated_map[idx] = reverify_sample((sample, prob_row))

    # Reconstruct full sample list
    result_samples = []
    for i, sample in enumerate(samples):
        if i in updated_map:
            result_samples.append(updated_map[i])
        else:
            result_samples.append(sample)

    # Track after state
    after = {field: 0 for field in MUTABLE_FIELDS}
    for idx in updated_map:
        s = updated_map[idx]
        if s.get("exploit_success"):
            after["exploit_success"] += 1
        if s.get("secure_pass"):
            after["secure_pass"] += 1
        if s.get("insecure_pass"):
            after["insecure_pass"] += 1

    changes = {}
    for field in ["exploit_success", "secure_pass", "insecure_pass"]:
        if before[field] != after[field]:
            diff = after[field] - before[field]
            changes[field] = f"{before[field]}->{after[field]} ({diff:+d})"

    task_ids_matched = {updated_map[idx].get("task_id") for idx in updated_map}

    summary = {
        "total": len(samples),
        "reverified": len(to_reverify),
        "task_ids_matched": len(task_ids_matched),
        "changes": changes,
        "errors": sum(1 for s in updated_map.values() if "_reverify_error" in s),
    }

    return result_samples, summary


def write_samples_atomic(samples: list[dict], path: Path):
    """Write samples JSONL atomically via temp file + rename."""
    fd, tmp_path = tempfile.mkstemp(dir=path.parent, suffix=".tmp", prefix=path.name)
    try:
        with os.fdopen(fd, "w") as f:
            for sample in samples:
                # Remove transient error field before writing
                out = {k: v for k, v in sample.items() if k != "_reverify_error"}
                f.write(json.dumps(out) + "\n")
        os.replace(tmp_path, path)
    except Exception:
        os.unlink(tmp_path)
        raise


def update_summary(samples_path: Path):
    """Update the main .jsonl summary to reflect reverified samples.

    Merges mutable fields from updated samples into the existing summary,
    preserving summary-only fields like output_tokens. Falls back to
    regeneration from samples if summary doesn't exist.
    """
    summary_name = samples_path.name.replace(".samples.jsonl", "")
    summary_path = samples_path.parent / summary_name

    samples = read_jsonl(samples_path)

    # Build lookup from samples keyed by (task_id, attempt_idx)
    samples_by_key = {}
    for s in samples:
        key = (s.get("task_id"), s.get("attempt_idx", 0))
        samples_by_key[key] = s

    if summary_path.exists():
        # Merge: update mutable fields in existing summary rows
        summary_rows = read_jsonl(summary_path)

        updated_rows = []
        for row in summary_rows:
            key = (row.get("task_id"), row.get("attempt_idx", 0))
            if key in samples_by_key:
                sample = samples_by_key[key]
                for field in MUTABLE_FIELDS:
                    if field in sample:
                        row[field] = sample[field]
            updated_rows.append(row)
    else:
        # No existing summary — generate from samples
        updated_rows = [
            {k: v for k, v in s.items() if k not in SAMPLES_ONLY_FIELDS} for s in samples
        ]

    fd, tmp_path = tempfile.mkstemp(dir=summary_path.parent, suffix=".tmp", prefix=summary_name)
    try:
        with os.fdopen(fd, "w") as f:
            for row in updated_rows:
                f.write(json.dumps(row) + "\n")
        os.replace(tmp_path, summary_path)
    except Exception:
        os.unlink(tmp_path)
        raise


def main():
    parser = argparse.ArgumentParser(description="Re-run djinn verification on stored completions")
    parser.add_argument(
        "--run-dir",
        nargs="+",
        type=Path,
        required=True,
        help="One or more run directories under results/prefill_sensitivity/",
    )
    parser.add_argument(
        "--task-ids",
        nargs="*",
        help="Filter to these task IDs (additive with --exploit-types)",
    )
    parser.add_argument(
        "--exploit-types",
        nargs="*",
        help="Filter to these exploit types (additive with --task-ids)",
    )
    parser.add_argument(
        "--remove-task-ids",
        nargs="+",
        help="Remove these task IDs from all JSONL files (no reverification, just delete rows)",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="Parallel workers (default 1; >1 may deadlock due to djinn daemon contention)",
    )
    parser.add_argument(
        "--dataset",
        default="EleutherAI/djinn-problems-v0.9",
        help="HuggingFace dataset to load problem definitions from",
    )
    parser.add_argument("--split", default="test_alternate")
    parser.add_argument("--backup", action="store_true", default=True)
    parser.add_argument("--no-backup", dest="backup", action="store_false")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    remove_task_ids = set(args.remove_task_ids) if args.remove_task_ids else None
    task_ids = set(args.task_ids) if args.task_ids else None
    exploit_types = set(args.exploit_types) if args.exploit_types else None

    # If neither filter given, reverify everything
    if not remove_task_ids and task_ids is None and exploit_types is None:
        print("No --task-ids or --exploit-types specified: reverifying ALL samples")

    # Load problem definitions (not needed for remove-only mode)
    if remove_task_ids and not task_ids and not exploit_types:
        problems = {}
    else:
        print(f"Loading problems from {args.dataset} split={args.split}...")
        problems = load_problems(args.dataset, args.split)
        print(f"Loaded {len(problems)} problems")

    # If exploit_types filter given, also find all task_ids with those exploit types
    # so we can report more useful info
    if exploit_types:
        matching_task_ids = {
            pid for pid, p in problems.items() if p.get("exploit_type") in exploit_types
        }
        print(f"Exploit type filter matches {len(matching_task_ids)} problems")

    # Collect all samples files
    all_samples_files = []
    for run_dir in args.run_dir:
        evals_dir = run_dir / "evals"
        if not evals_dir.exists():
            print(f"Warning: {evals_dir} does not exist, skipping")
            continue
        samples_files = sorted(evals_dir.glob("*.samples.jsonl"))
        all_samples_files.extend(samples_files)

    if not all_samples_files:
        print("No .samples.jsonl files found!")
        sys.exit(1)

    print(f"\nFound {len(all_samples_files)} samples files across {len(args.run_dir)} runs")

    # --- Remove pass: delete rows matching --remove-task-ids ---
    if remove_task_ids:
        print(f"\nRemoving {len(remove_task_ids)} task IDs from all files...")
        total_removed = 0
        files_modified = 0
        for samples_path in tqdm(all_samples_files, desc="removing", unit="file"):
            samples = read_jsonl(samples_path)
            filtered = [s for s in samples if s.get("task_id") not in remove_task_ids]
            n_removed = len(samples) - len(filtered)
            if n_removed == 0:
                continue
            total_removed += n_removed
            files_modified += 1
            run_name = samples_path.parent.parent.name
            tqdm.write(
                f"  {run_name}/{samples_path.name}: removed {n_removed}/{len(samples)} samples"
            )
            if not args.dry_run:
                if args.backup:
                    backup_dir = (
                        samples_path.parent / f"backup_remove_{datetime.now():%Y%m%d_%H%M%S}"
                    )
                    backup_dir.mkdir(exist_ok=True)
                    shutil.copy2(samples_path, backup_dir / samples_path.name)
                    summary_name = samples_path.name.replace(".samples.jsonl", "")
                    summary_path = samples_path.parent / summary_name
                    if summary_path.exists():
                        shutil.copy2(summary_path, backup_dir / summary_name)
                write_samples_atomic(filtered, samples_path)
                # Also remove from summary JSONL
                summary_name = samples_path.name.replace(".samples.jsonl", "")
                summary_path = samples_path.parent / summary_name
                if summary_path.exists():
                    summary_rows = read_jsonl(summary_path)
                    summary_filtered = [
                        r for r in summary_rows if r.get("task_id") not in remove_task_ids
                    ]
                    write_samples_atomic(summary_filtered, summary_path)
        prefix = "DRY RUN: would remove" if args.dry_run else "Removed"
        print(f"{prefix} {total_removed} samples across {files_modified} files\n")

        # If no reverification filters, we're done
        if task_ids is None and exploit_types is None:
            return

    # Process each file
    total_stats = {
        "files_processed": 0,
        "samples_reverified": 0,
        "exploit_success_before": 0,
        "exploit_success_after": 0,
        "errors": 0,
    }

    file_pbar = tqdm(all_samples_files, desc="files", unit="file")
    for samples_path in file_pbar:
        rel_path = samples_path.name
        run_name = samples_path.parent.parent.name
        file_pbar.set_postfix_str(
            f"{run_name}/{rel_path.replace('.jsonl.samples.jsonl', '')}",
            refresh=True,
        )

        updated_samples, summary = reverify_file(
            samples_path, problems, task_ids, exploit_types, args.workers
        )

        if summary["reverified"] == 0:
            continue

        total_stats["files_processed"] += 1
        total_stats["samples_reverified"] += summary["reverified"]
        total_stats["errors"] += summary.get("errors", 0)

        # Print per-file summary
        changes_str = (
            ", ".join(f"{k} {v}" for k, v in summary["changes"].items())
            if summary["changes"]
            else "no changes"
        )

        tqdm.write(
            f"  {run_name}/{rel_path}: "
            f"reverified {summary['reverified']}/{summary['total']} "
            f"({summary['task_ids_matched']} task_ids) — {changes_str}"
            + (f" ({summary['errors']} errors)" if summary.get("errors") else "")
        )

        if not args.dry_run and summary["changes"]:
            # Backup originals
            if args.backup:
                backup_dir = samples_path.parent / f"backup_reverify_{datetime.now():%Y%m%d_%H%M%S}"
                backup_dir.mkdir(exist_ok=True)
                shutil.copy2(samples_path, backup_dir / samples_path.name)
                # Also backup summary JSONL
                summary_name = samples_path.name.replace(".samples.jsonl", "")
                summary_path = samples_path.parent / summary_name
                if summary_path.exists():
                    shutil.copy2(summary_path, backup_dir / summary_name)

            # Write updated files
            write_samples_atomic(updated_samples, samples_path)
            update_summary(samples_path)

    # Print totals
    print(f"\n{'DRY RUN ' if args.dry_run else ''}Summary:")
    print(f"  Files processed: {total_stats['files_processed']}")
    print(f"  Samples reverified: {total_stats['samples_reverified']}")
    if total_stats["errors"]:
        print(f"  Errors: {total_stats['errors']}")


if __name__ == "__main__":
    main()
