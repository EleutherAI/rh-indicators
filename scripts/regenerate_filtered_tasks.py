#!/usr/bin/env python3
"""Regenerate eval results for tasks that were filtered out by filter_fixed_problems.py.

After fixing a djinn problem and filtering old results, this script re-runs
eval_checkpoint_sensitivity.py --resume on the affected runs. The --resume
mode detects missing task_ids in eval files and djinn's eval only evaluates
tasks not already present — so only the filtered tasks get re-evaluated.

Usage:
    # Show what would be done
    python scripts/regenerate_filtered_tasks.py \
        --run-dirs results/prefill_sensitivity/prefill_sensitivity-20260127-050226-8a0e189 \
        --dry-run

    # Regenerate all affected runs (sequential)
    python scripts/regenerate_filtered_tasks.py \
        --run-dirs results/prefill_sensitivity/prefill_sensitivity-20260127-050226-8a0e189 \
                   results/prefill_sensitivity/prefill_sensitivity-20260305-053029

    # Override GPU config
    python scripts/regenerate_filtered_tasks.py \
        --run-dirs results/prefill_sensitivity/prefill_sensitivity-20260127-050226-8a0e189 \
        --tensor-parallel 1 --data-parallel 3
"""

import argparse
import json
import subprocess
import sys
from pathlib import Path

import yaml


BACKUP_SUFFIX = ".pre_filter_backup"

# Canonical run directories from docs/experiment_provenance.md.
# Only these are processed when --run-dirs is not specified.
CANONICAL_RUNS = [
    # gpt-oss-20b
    "results/prefill_sensitivity/prefill_sensitivity-20260127-050226-8a0e189",
    "results/prefill_sensitivity/prefill_sensitivity-20260206-045419-8a0e189",
    "results/prefill_sensitivity/prefill_sensitivity-20260220-032628",
    "results/prefill_sensitivity/prefill_sensitivity-20260223-030541",
    "results/prefill_sensitivity/prefill_sensitivity-20260224-043634",
    # Qwen3-8B
    "results/prefill_sensitivity/prefill_sensitivity-20260211-030018-8a0e189",
    "results/prefill_sensitivity/prefill_sensitivity-20260217-054915",
    "results/prefill_sensitivity/prefill_sensitivity-20260217-235715-3a546a8",
    "results/prefill_sensitivity/prefill_sensitivity-20260224-034624-6d05d75",
    "results/prefill_sensitivity/prefill_sensitivity-20260224-235217",
    "results/prefill_sensitivity/prefill_sensitivity-20260227-000123",
    # gpt-oss-20b templated
    "results/prefill_sensitivity/prefill_sensitivity-20260302-224813",
    "results/prefill_sensitivity/prefill_sensitivity-20260304-044604",
    "results/prefill_sensitivity/prefill_sensitivity-20260305-053029",
    # spot check T=1.0
    "results/spot_check_t10/prefill_sensitivity-20260127-050226-8a0e189/donor/prefill_sensitivity-20260310-013320-6d05d75",
    "results/spot_check_t10/prefill_sensitivity-20260127-050226-8a0e189/target/prefill_sensitivity-20260310-020147-6d05d75",
    # donor evals
    "results/donor_evals/gpt-oss-20b/prefill_sensitivity-20260311-053840-6d05d75",
]


def find_affected_eval_files(run_dir: Path) -> list[Path]:
    """Find eval files that have .pre_filter_backup counterparts.

    These are the files that had entries removed by filter_fixed_problems.py
    and need regeneration of the missing tasks.
    """
    evals_dir = run_dir / "evals"
    if not evals_dir.exists():
        return []

    affected = []
    for backup in sorted(evals_dir.glob(f"*{BACKUP_SUFFIX}")):
        original = evals_dir / backup.name.removesuffix(BACKUP_SUFFIX)
        if original.exists():
            affected.append(original)

    return affected


def count_missing_tasks(run_dir: Path) -> dict[str, list[str]]:
    """Compare filtered files against backups to find which task_ids are missing.

    Returns {filename: [missing_task_ids]}.
    """
    evals_dir = run_dir / "evals"
    missing = {}

    for backup in sorted(evals_dir.glob(f"*{BACKUP_SUFFIX}")):
        # Skip .samples.jsonl backups — they mirror the main file
        if ".samples.jsonl" in backup.name:
            continue

        original = evals_dir / backup.name.removesuffix(BACKUP_SUFFIX)
        if not original.exists():
            continue

        # Get task_ids in original vs backup
        def get_task_ids(path):
            ids = set()
            for line in path.read_text().splitlines():
                if line.strip():
                    try:
                        ids.add(json.loads(line).get("task_id", ""))
                    except json.JSONDecodeError:
                        pass
            return ids

        backup_ids = get_task_ids(backup)
        current_ids = get_task_ids(original)
        diff = backup_ids - current_ids
        if diff:
            missing[original.name] = sorted(diff)

    return missing


def load_run_config(run_dir: Path) -> dict:
    """Load config.yaml from a run directory."""
    config_path = run_dir / "config.yaml"
    if not config_path.exists():
        raise FileNotFoundError(f"No config.yaml in {run_dir}")

    with open(config_path) as f:
        return yaml.safe_load(f)


def build_resume_command(
    run_dir: Path,
    config: dict,
    tensor_parallel: int | None = None,
    data_parallel: int | None = None,
    gpu_memory_utilization: float | None = None,
) -> list[str]:
    """Build eval_checkpoint_sensitivity.py --resume command from config."""
    cmd = [
        sys.executable,
        "scripts/eval_checkpoint_sensitivity.py",
        "--checkpoint-dir", str(config["checkpoint_dir"]),
        "--resume", str(run_dir),
        "--dataset", config.get("dataset", "EleutherAI/djinn-problems-v0.9"),
        "--split", config.get("split", "test_alternate"),
        "--attempts", str(config.get("attempts", 1)),
        "--temperature", str(config.get("temperature", 1.0)),
        "--tensor-parallel", str(tensor_parallel or config.get("tensor_parallel", 4)),
        "--data-parallel", str(data_parallel or config.get("data_parallel", 1)),
        "--gpu-memory-utilization", str(gpu_memory_utilization or config.get("gpu_memory_utilization", 0.7)),
    ]

    if config.get("prefill_tokens_sweep"):
        cmd.extend(["--prefill-tokens-sweep", config["prefill_tokens_sweep"]])
    if config.get("prefill_source"):
        cmd.extend(["--prefill-source", str(config["prefill_source"])])
    if config.get("prefill_mode"):
        cmd.extend(["--prefill-mode", config["prefill_mode"]])
    if config.get("no_harmony"):
        cmd.append("--no-harmony")
    if config.get("skip_merge"):
        cmd.append("--skip-merge")
    if config.get("merge_on_cpu"):
        cmd.append("--merge-on-cpu")
    if config.get("checkpoints"):
        cmd.extend(["--checkpoints"] + config["checkpoints"])

    return cmd


def main():
    parser = argparse.ArgumentParser(
        description="Regenerate eval results for filtered tasks via --resume"
    )
    parser.add_argument(
        "--run-dirs",
        nargs="+",
        type=Path,
        default=None,
        help="Run directories to regenerate (default: canonical runs from experiment_provenance.md)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be done without running evals",
    )
    parser.add_argument(
        "--tensor-parallel",
        type=int,
        default=None,
        help="Override tensor parallel size (default: from config)",
    )
    parser.add_argument(
        "--data-parallel",
        type=int,
        default=None,
        help="Override data parallel size (default: from config)",
    )
    parser.add_argument(
        "--gpu-memory-utilization",
        type=float,
        default=None,
        help="Override GPU memory utilization (default: from config)",
    )

    args = parser.parse_args()

    run_dirs = args.run_dirs
    if run_dirs is None:
        run_dirs = [Path(p) for p in CANONICAL_RUNS]
        print(f"Using {len(run_dirs)} canonical runs from experiment_provenance.md\n")

    if args.dry_run:
        print("=== DRY RUN ===\n")

    for run_dir in run_dirs:
        if not run_dir.exists():
            print(f"Warning: {run_dir} does not exist, skipping")
            continue

        print(f"\n{'='*60}")
        print(f"Run: {run_dir}")
        print(f"{'='*60}")

        # Find affected files and missing tasks
        affected = find_affected_eval_files(run_dir)
        if not affected:
            print("  No affected files (no .pre_filter_backup found), skipping")
            continue

        missing = count_missing_tasks(run_dir)
        if not missing:
            print("  All tasks present (already regenerated?), skipping")
            continue

        all_missing = set()
        for fname, task_ids in missing.items():
            all_missing.update(task_ids)
            print(f"  {fname}: missing {len(task_ids)} task(s): {', '.join(task_ids)}")

        print(f"  Total: {sum(len(v) for v in missing.values())} missing entries across {len(missing)} files")
        print(f"  Tasks to regenerate: {', '.join(sorted(all_missing))}")

        # Load config
        try:
            config = load_run_config(run_dir)
        except FileNotFoundError:
            print(f"  ERROR: No config.yaml — cannot determine how to re-run")
            continue

        # Build resume command
        cmd = build_resume_command(
            run_dir, config,
            tensor_parallel=args.tensor_parallel,
            data_parallel=args.data_parallel,
            gpu_memory_utilization=args.gpu_memory_utilization,
        )

        print(f"\n  Command:")
        print(f"    {' '.join(cmd)}")

        if args.dry_run:
            continue

        print(f"\n  Running eval_checkpoint_sensitivity.py --resume ...")
        result = subprocess.run(cmd)

        if result.returncode != 0:
            print(f"  ERROR: eval exited with code {result.returncode}")
            print(f"  Stopping. Fix the issue and re-run for remaining dirs.")
            sys.exit(result.returncode)

        print(f"  Regeneration complete for {run_dir.name}")

    print("\nDone.")


if __name__ == "__main__":
    main()
