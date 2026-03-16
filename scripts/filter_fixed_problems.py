#!/usr/bin/env python3
"""Filter fixed/corrected problem IDs out of existing eval result files.

When a djinn problem is fixed (e.g., weak test cases strengthened), old eval
results for that problem are invalid. This script removes those entries from
all JSONL result files so downstream analyses exclude them, and optionally
appends a TODO to docs/experiment_provenance.md.

Usage:
    # Filter specific task IDs from a single run
    python scripts/filter_fixed_problems.py \
        --task-ids circuit_test_poisoning_006 array_traversal_verifier_bypass_028_04 \
        --run-dirs results/prefill_sensitivity/prefill_sensitivity-20260127-050226-8a0e189

    # Filter from all canonical runs
    python scripts/filter_fixed_problems.py \
        --task-ids circuit_test_poisoning_006 \
        --run-dirs results/prefill_sensitivity/prefill_sensitivity-20260127-050226-8a0e189 \
                   results/prefill_sensitivity/prefill_sensitivity-20260211-030018-8a0e189

    # Dry run (show what would be filtered)
    python scripts/filter_fixed_problems.py \
        --task-ids circuit_test_poisoning_006 \
        --run-dirs results/prefill_sensitivity/prefill_sensitivity-20260127-050226-8a0e189 \
        --dry-run

    # Also mark TODO in experiment provenance
    python scripts/filter_fixed_problems.py \
        --task-ids circuit_test_poisoning_006 \
        --run-dirs results/prefill_sensitivity/prefill_sensitivity-20260127-050226-8a0e189 \
        --mark-provenance --reason "Weak honor system test cases (false positive exploits)"
"""

import argparse
import json
import shutil
import sys
from datetime import datetime
from pathlib import Path


def filter_jsonl_file(
    filepath: Path,
    task_ids: set[str],
    dry_run: bool = False,
    backup_suffix: str | None = None,
) -> tuple[int, int]:
    """Remove lines with matching task_ids from a JSONL file.

    Returns (total_lines, removed_lines).
    """
    if not filepath.exists():
        return 0, 0

    lines = filepath.read_text().splitlines()
    kept = []
    removed = 0

    for line in lines:
        if not line.strip():
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError:
            kept.append(line)
            continue

        tid = record.get("task_id", "")
        if tid in task_ids:
            removed += 1
        else:
            kept.append(line)

    if removed > 0 and not dry_run:
        if backup_suffix:
            backup_path = filepath.parent / f"{filepath.name}{backup_suffix}"
            shutil.copy2(filepath, backup_path)

        filepath.write_text("\n".join(kept) + "\n" if kept else "")

    return len(lines), removed


def filter_run_directory(
    run_dir: Path,
    task_ids: set[str],
    dry_run: bool = False,
    backup_suffix: str = ".pre_filter_backup",
) -> dict:
    """Filter task_ids from all JSONL files in a run directory.

    Scans evals/, logprob/, kl/, and samples files.
    Returns summary dict of what was filtered.
    """
    summary = {"run_dir": str(run_dir), "files": {}}

    # Subdirectories containing JSONL result files
    subdirs = ["evals", "logprob", "kl", "ref_logprob"]

    for subdir in subdirs:
        subdir_path = run_dir / subdir
        if not subdir_path.exists():
            continue

        for jsonl_file in sorted(subdir_path.glob("*.jsonl")):
            total, removed = filter_jsonl_file(
                jsonl_file, task_ids, dry_run=dry_run, backup_suffix=backup_suffix
            )
            if removed > 0:
                summary["files"][str(jsonl_file.relative_to(run_dir))] = {
                    "total": total,
                    "removed": removed,
                    "remaining": total - removed,
                }

    return summary


def mark_provenance_todo(
    task_ids: list[str],
    run_dirs: list[Path],
    reason: str,
    provenance_path: Path | None = None,
    dry_run: bool = False,
):
    """Append a TODO to docs/experiment_provenance.md."""
    if provenance_path is None:
        provenance_path = Path("docs/experiment_provenance.md")

    if not provenance_path.exists():
        print(f"  Warning: {provenance_path} not found, skipping provenance update")
        return

    date_str = datetime.now().strftime("%Y-%m-%d")
    run_ids = [d.name for d in run_dirs]

    todo_entry = (
        f"- [ ] TODO: Regenerate filtered tasks ({date_str}): "
        f"`{'`, `'.join(task_ids)}` — {reason}. "
        f"Filtered from: {', '.join(f'`{r}`' for r in run_ids)}"
    )

    if dry_run:
        print(f"\n  Would append to {provenance_path}:")
        print(f"  {todo_entry}")
        return

    content = provenance_path.read_text()

    # Find the TODO section and append
    todo_marker = "## TODO"
    if todo_marker in content:
        # Insert before the last line of the TODO section
        lines = content.split("\n")
        todo_idx = None
        for i, line in enumerate(lines):
            if todo_marker in line:
                todo_idx = i
                break

        if todo_idx is not None:
            # Find end of TODO section (next ## or end of file)
            insert_idx = len(lines)
            for i in range(todo_idx + 1, len(lines)):
                if lines[i].startswith("## ") and not lines[i].startswith("## TODO"):
                    insert_idx = i
                    break

            lines.insert(insert_idx, todo_entry)
            content = "\n".join(lines)
    else:
        # Append a new TODO section
        content = content.rstrip() + f"\n\n## TODO\n\n{todo_entry}\n"

    provenance_path.write_text(content)
    print(f"  Updated {provenance_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Filter fixed/corrected problem IDs from eval result files"
    )
    parser.add_argument(
        "--task-ids",
        nargs="+",
        required=True,
        help="Problem task_ids to filter out (e.g., circuit_test_poisoning_006)",
    )
    parser.add_argument(
        "--run-dirs",
        nargs="+",
        type=Path,
        required=True,
        help="Prefill sensitivity run directories to filter",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be filtered without making changes",
    )
    parser.add_argument(
        "--no-backup",
        action="store_true",
        help="Skip creating backup files (not recommended)",
    )
    parser.add_argument(
        "--mark-provenance",
        action="store_true",
        help="Append TODO to docs/experiment_provenance.md",
    )
    parser.add_argument(
        "--reason",
        default="Fixed problem definition",
        help="Reason for filtering (used in provenance TODO)",
    )
    parser.add_argument(
        "--provenance-path",
        type=Path,
        default=None,
        help="Path to experiment_provenance.md (default: docs/experiment_provenance.md)",
    )

    args = parser.parse_args()
    task_ids = set(args.task_ids)
    backup_suffix = None if args.no_backup else ".pre_filter_backup"

    if args.dry_run:
        print("=== DRY RUN (no changes will be made) ===\n")

    print(f"Filtering {len(task_ids)} task(s): {', '.join(sorted(task_ids))}")
    print(f"From {len(args.run_dirs)} run director{'y' if len(args.run_dirs) == 1 else 'ies'}\n")

    total_removed = 0
    total_files = 0

    for run_dir in args.run_dirs:
        if not run_dir.exists():
            print(f"  Warning: {run_dir} does not exist, skipping")
            continue

        print(f"Processing: {run_dir.name}")
        summary = filter_run_directory(
            run_dir, task_ids, dry_run=args.dry_run, backup_suffix=backup_suffix
        )

        if summary["files"]:
            for filepath, counts in summary["files"].items():
                action = "Would remove" if args.dry_run else "Removed"
                print(
                    f"  {action} {counts['removed']}/{counts['total']} entries from {filepath}"
                )
                total_removed += counts["removed"]
                total_files += 1
        else:
            print("  No matching entries found")

    print(f"\n{'Would filter' if args.dry_run else 'Filtered'}: "
          f"{total_removed} entries across {total_files} files")

    if args.mark_provenance:
        mark_provenance_todo(
            args.task_ids,
            args.run_dirs,
            args.reason,
            provenance_path=args.provenance_path,
            dry_run=args.dry_run,
        )

    if total_removed == 0:
        print("\nNo entries matched. Check that task_ids are correct.")
        sys.exit(0)

    if not args.dry_run and backup_suffix:
        print(f"\nBackups created with suffix '{backup_suffix}'")


if __name__ == "__main__":
    main()
