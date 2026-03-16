#!/usr/bin/env python3
"""Recompute evals + logprobs/KL for runs affected by the T=0.4 temperature bug.

Seven canonical prefill_sensitivity runs sampled completions at T=0.4 but all
downstream analysis assumes T=1.0. This script:
  1. Backs up existing evals/, logprob/, kl/ to backup_t04_{timestamp}/
  2. Re-runs evals at T=1.0 via eval_checkpoint_sensitivity.py
  3. Recomputes logprobs + KL via serve_and_compute_logprobs.py
  4. Cleans up old per-file backup dirs (backup_remove_*, backup_reverify_*)

Also supports a --spot-check mode to regenerate the donor prefill source at
T=1.0 and compare a few target checkpoints against old results, before
committing to the full recomputation.

Usage:
    python scripts/recompute_temperature_fix.py --dry-run          # Preview
    python scripts/recompute_temperature_fix.py --all              # All 7 runs
    python scripts/recompute_temperature_fix.py --runs <name> ...  # Specific runs
    python scripts/recompute_temperature_fix.py --stage backup     # Backup only
    python scripts/recompute_temperature_fix.py --stage eval       # Eval only
    python scripts/recompute_temperature_fix.py --stage logprob    # Logprob+KL only
    python scripts/recompute_temperature_fix.py --stage cleanup    # Remove old backup dirs

    # Spot-check: regenerate donor at T=1.0 and compare selected checkpoints
    python scripts/recompute_temperature_fix.py --spot-check \\
        --runs 20260127 --spot-check-checkpoints 6 76 --dry-run
"""

import argparse
import json
import shutil
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import yaml

AFFECTED_RUNS = [
    {
        "name": "prefill_sensitivity-20260127-050226-8a0e189",
        "desc": "gpt-oss-20b exploit",
        "family": "gpt-oss-20b",
    },
    {
        "name": "prefill_sensitivity-20260206-045419-8a0e189",
        "desc": "gpt-oss-20b control tasks",
        "family": "gpt-oss-20b",
    },
    {
        "name": "prefill_sensitivity-20260211-030018-8a0e189",
        "desc": "Qwen3-8B exploit",
        "family": "qwen3-8b",
    },
    {
        "name": "prefill_sensitivity-20260217-054915",
        "desc": "Qwen3-8B misalignment ctrl",
        "family": "qwen3-8b",
    },
    {
        "name": "prefill_sensitivity-20260217-235715-3a546a8",
        "desc": "Qwen3-8B clean ctrl",
        "family": "qwen3-8b",
    },
    {
        "name": "prefill_sensitivity-20260220-032628",
        "desc": "gpt-oss-20b clean ctrl",
        "family": "gpt-oss-20b",
    },
    {
        "name": "prefill_sensitivity-20260223-030541",
        "desc": "gpt-oss-20b misalignment ctrl",
        "family": "gpt-oss-20b",
    },
]

# Run with ref_logprob from the first gpt-oss-20b run
REF_LOGPROB_DONOR = "prefill_sensitivity-20260127-050226-8a0e189"

# Runs that have NO ref_logprob/ dir of their own
RUNS_NEEDING_EXTERNAL_REF = {
    "prefill_sensitivity-20260220-032628",
    "prefill_sensitivity-20260223-030541",
}

MANIFEST_PATH = Path("results/temperature_fix_manifest.jsonl")


def load_run_config(run_dir: Path) -> dict:
    config_path = run_dir / "config.yaml"
    if not config_path.exists():
        raise FileNotFoundError(f"No config.yaml in {run_dir}")
    with open(config_path) as f:
        return yaml.safe_load(f)


def append_manifest(entry: dict):
    MANIFEST_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(MANIFEST_PATH, "a") as f:
        f.write(json.dumps(entry) + "\n")


def collect_files(directory: Path, pattern: str = "*.jsonl") -> list[Path]:
    """Collect files in directory matching pattern, excluding subdirs."""
    if not directory.exists():
        return []
    return sorted(f for f in directory.glob(pattern) if f.is_file())


def do_backup(run_dir: Path, dry_run: bool) -> dict | None:
    """Back up evals/, logprob/, kl/ to backup_t04_{timestamp}/."""
    # Check if already backed up
    existing_backups = list(run_dir.glob("backup_t04_*"))
    if existing_backups:
        print(f"  SKIP backup — already exists: {existing_backups[0].name}")
        return None

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_dir = run_dir / f"backup_t04_{timestamp}"

    evals_dir = run_dir / "evals"
    logprob_dir = run_dir / "logprob"
    kl_dir = run_dir / "kl"

    # Collect files (exclude subdirs like backup_remove_*, backup_reverify_*)
    eval_files = collect_files(evals_dir, "*.jsonl")
    eval_samples = collect_files(evals_dir, "*.samples.jsonl")
    # eval_files includes samples since *.jsonl matches; deduplicate
    eval_all = sorted(set(eval_files) | set(eval_samples))
    logprob_files = collect_files(logprob_dir)
    kl_files = collect_files(kl_dir)

    counts = {
        "evals": len(eval_all),
        "logprob": len(logprob_files),
        "kl": len(kl_files),
    }

    print(f"  Backup → {backup_dir.name}")
    print(f"    evals: {counts['evals']} files")
    print(f"    logprob: {counts['logprob']} files")
    print(f"    kl: {counts['kl']} files")

    if dry_run:
        return counts

    # Create backup subdirs
    (backup_dir / "evals").mkdir(parents=True, exist_ok=True)
    if logprob_files:
        (backup_dir / "logprob").mkdir(parents=True, exist_ok=True)
    if kl_files:
        (backup_dir / "kl").mkdir(parents=True, exist_ok=True)

    # Move files (shutil.move = rename on same filesystem, instant)
    for f in eval_all:
        shutil.move(str(f), str(backup_dir / "evals" / f.name))
    for f in logprob_files:
        shutil.move(str(f), str(backup_dir / "logprob" / f.name))
    for f in kl_files:
        shutil.move(str(f), str(backup_dir / "kl" / f.name))

    # Write manifest
    manifest = {
        "timestamp": timestamp,
        "reason": "temperature_fix_t04_to_t10",
        "original_temperature": 0.4,
        "file_counts": counts,
    }
    with open(backup_dir / "manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)

    append_manifest({
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "run": run_dir.name,
        "stage": "backup",
        "files_moved": counts,
    })

    # Verify evals/ is now empty of .jsonl files
    remaining = collect_files(evals_dir)
    if remaining:
        print(f"  WARNING: {len(remaining)} files remain in evals/ after backup")

    return counts


def build_eval_cmd(run_dir: Path, config: dict, args) -> list[str]:
    """Build the eval_checkpoint_sensitivity.py command from config."""
    cmd = [
        sys.executable, "scripts/eval_checkpoint_sensitivity.py",
        "--checkpoint-dir", str(config["checkpoint_dir"]),
        "--resume", str(run_dir),
        "--skip-merge",
        "--temperature", "1.0",
    ]

    # Dataset / split
    if config.get("dataset"):
        cmd += ["--dataset", str(config["dataset"])]
    if config.get("split"):
        cmd += ["--split", str(config["split"])]

    # Attempts / concurrency
    if config.get("attempts"):
        cmd += ["--attempts", str(config["attempts"])]
    if config.get("concurrency"):
        cmd += ["--concurrency", str(config["concurrency"])]

    # Prefill config
    if config.get("prefill_tokens_sweep"):
        cmd += ["--prefill-tokens-sweep", str(config["prefill_tokens_sweep"])]
    if config.get("prefill_source"):
        cmd += ["--prefill-source", str(config["prefill_source"])]
    if config.get("prefill_mode") and config["prefill_mode"] != "natural":
        cmd += ["--prefill-mode", str(config["prefill_mode"])]
    if config.get("prefill_index") and config["prefill_index"] != 0:
        cmd += ["--prefill-index", str(config["prefill_index"])]

    # Checkpoints
    if config.get("checkpoints"):
        ckpt_names = []
        for c in config["checkpoints"]:
            s = str(c)
            if not s.startswith("checkpoint-"):
                s = f"checkpoint-{s}"
            ckpt_names.append(s)
        cmd += ["--checkpoints"] + ckpt_names

    # GPU config (allow CLI overrides)
    tp = args.tensor_parallel if args.tensor_parallel is not None else config.get("tensor_parallel", 4)
    dp = args.data_parallel if args.data_parallel is not None else config.get("data_parallel", 1)
    cmd += ["--tensor-parallel", str(tp)]
    cmd += ["--data-parallel", str(dp)]

    # GPU memory
    gpu_mem = args.gpu_memory_utilization if args.gpu_memory_utilization is not None else config.get("gpu_memory_utilization", 0.70)
    cmd += ["--gpu-memory-utilization", str(gpu_mem)]

    # Flags
    if config.get("no_harmony"):
        cmd += ["--no-harmony"]
    if config.get("merge_on_cpu"):
        cmd += ["--merge-on-cpu"]

    return cmd


def do_eval(run_dir: Path, config: dict, args) -> bool:
    """Re-run evals at T=1.0."""
    # Check config temperature
    orig_temp = config.get("temperature", "unknown")
    if orig_temp == 1.0:
        print(f"  WARNING: config.yaml already shows temperature=1.0")

    cmd = build_eval_cmd(run_dir, config, args)

    print(f"  Eval command:")
    print(f"    {' '.join(cmd)}")

    if args.dry_run:
        return True

    # Setup logging
    log_dir = run_dir / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_dir / "recompute_eval_t10.log"

    print(f"  Running eval (log: {log_file})...")
    start = time.time()

    with open(log_file, "w") as lf:
        result = subprocess.run(
            cmd, stdout=lf, stderr=subprocess.STDOUT,
            timeout=24 * 3600,  # 24h max
        )

    duration = time.time() - start
    print(f"  Eval finished in {duration/3600:.1f}h (exit code {result.returncode})")

    # Count new eval files
    new_eval_count = len(collect_files(run_dir / "evals"))

    append_manifest({
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "run": run_dir.name,
        "stage": "eval",
        "command": " ".join(cmd),
        "exit_code": result.returncode,
        "duration_s": round(duration, 1),
        "new_eval_files": new_eval_count,
    })

    return result.returncode == 0


def do_logprob(run_dir: Path, results_dir: Path, args) -> bool:
    """Recompute logprobs + KL via serve_and_compute_logprobs.py."""
    cmd = [
        sys.executable, "scripts/serve_and_compute_logprobs.py",
        "--run-dir", str(run_dir),
    ]

    # Handle ref_logprobs
    if run_dir.name in RUNS_NEEDING_EXTERNAL_REF:
        donor_dir = results_dir / REF_LOGPROB_DONOR / "ref_logprob"
        if not donor_dir.exists():
            print(f"  ERROR: ref_logprob donor not found: {donor_dir}")
            return False
        cmd += ["--ref-logprobs-dir", str(donor_dir)]
    # else: serve_and_compute_logprobs.py auto-detects {run_dir}/ref_logprob/

    print(f"  Logprob command:")
    print(f"    {' '.join(cmd)}")

    if args.dry_run:
        return True

    log_dir = run_dir / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_dir / "recompute_logprob_t10.log"

    print(f"  Running logprob (log: {log_file})...")
    start = time.time()

    with open(log_file, "w") as lf:
        result = subprocess.run(
            cmd, stdout=lf, stderr=subprocess.STDOUT,
            timeout=12 * 3600,  # 12h max
        )

    duration = time.time() - start
    print(f"  Logprob finished in {duration/3600:.1f}h (exit code {result.returncode})")

    logprob_count = len(collect_files(run_dir / "logprob"))
    kl_count = len(collect_files(run_dir / "kl"))

    append_manifest({
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "run": run_dir.name,
        "stage": "logprob",
        "exit_code": result.returncode,
        "duration_s": round(duration, 1),
        "logprob_files": logprob_count,
        "kl_files": kl_count,
    })

    return result.returncode == 0


def validate_after_backup(run_dir: Path):
    """Check backup looks correct."""
    backups = list(run_dir.glob("backup_t04_*"))
    if not backups:
        print(f"  WARN: no backup_t04_* found in {run_dir.name}")
        return
    backup = backups[0]
    manifest_path = backup / "manifest.json"
    if manifest_path.exists():
        with open(manifest_path) as f:
            manifest = json.load(f)
        expected = manifest["file_counts"]
        actual_evals = len(collect_files(backup / "evals"))
        actual_logprob = len(collect_files(backup / "logprob"))
        actual_kl = len(collect_files(backup / "kl"))
        ok = (actual_evals == expected["evals"] and
              actual_logprob == expected["logprob"] and
              actual_kl == expected["kl"])
        status = "OK" if ok else "MISMATCH"
        print(f"  Backup validation: {status} "
              f"(evals={actual_evals}/{expected['evals']}, "
              f"logprob={actual_logprob}/{expected['logprob']}, "
              f"kl={actual_kl}/{expected['kl']})")


def validate_after_eval(run_dir: Path):
    """Check new eval files match backup expectations."""
    backups = list(run_dir.glob("backup_t04_*"))
    if not backups:
        return
    backup = backups[0]
    expected_evals = len(collect_files(backup / "evals"))
    actual_evals = len(collect_files(run_dir / "evals"))
    print(f"  Eval validation: {actual_evals} new files (backup had {expected_evals})")


def do_cleanup(run_dir: Path, dry_run: bool) -> dict:
    """Remove backup_remove_*/backup_reverify_* from evals/ when backup_t04_* is validated."""
    backups = list(run_dir.glob("backup_t04_*"))
    if not backups:
        print(f"  SKIP cleanup — no backup_t04_* found (run backup stage first)")
        return {"removed": 0}

    backup = backups[0]
    if not (backup / "manifest.json").exists():
        print(f"  SKIP cleanup — backup has no manifest.json")
        return {"removed": 0}

    evals_dir = run_dir / "evals"
    if not evals_dir.exists():
        return {"removed": 0}

    cleanup_dirs = sorted(
        list(evals_dir.glob("backup_remove_*")) +
        list(evals_dir.glob("backup_reverify_*"))
    )

    if not cleanup_dirs:
        print(f"  No backup_remove/backup_reverify dirs to clean up")
        return {"removed": 0}

    total_size = sum(
        sum(f.stat().st_size for f in d.rglob("*") if f.is_file())
        for d in cleanup_dirs
    )

    print(f"  Found {len(cleanup_dirs)} old backup dirs ({total_size / 1e6:.0f} MB)")
    if dry_run:
        for d in cleanup_dirs[:5]:
            print(f"    Would remove: {d.name}")
        if len(cleanup_dirs) > 5:
            print(f"    ... and {len(cleanup_dirs) - 5} more")
        return {"removed": len(cleanup_dirs), "size_mb": round(total_size / 1e6, 1)}

    for d in cleanup_dirs:
        shutil.rmtree(d)

    print(f"  Removed {len(cleanup_dirs)} directories ({total_size / 1e6:.0f} MB freed)")

    append_manifest({
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "run": run_dir.name,
        "stage": "cleanup",
        "dirs_removed": len(cleanup_dirs),
        "size_bytes": total_size,
    })

    return {"removed": len(cleanup_dirs), "size_mb": round(total_size / 1e6, 1)}


# ---------------------------------------------------------------------------
# Spot-check mode: regenerate donor at T=1.0 and compare target checkpoints
# ---------------------------------------------------------------------------

def resolve_donor_from_config(config: dict) -> tuple[Path, str] | None:
    """Extract donor run dir and checkpoint name from a run's prefill_source config.

    e.g. 'results/prefill_ref/.../evals/checkpoint-132.jsonl.samples.jsonl'
    → (Path('results/prefill_ref/...'), 'checkpoint-132')
    """
    prefill_source = config.get("prefill_source")
    if not prefill_source:
        return None
    p = Path(prefill_source)
    # checkpoint-132.jsonl.samples.jsonl → checkpoint-132
    ckpt_name = p.name.split(".jsonl")[0]
    # parent of evals/ is the run dir
    donor_run_dir = p.parent.parent
    return donor_run_dir, ckpt_name


def build_fresh_eval_cmd(
    checkpoint_dir: str,
    checkpoints: list[str],
    output_dir: Path,
    config: dict,
    args,
    *,
    prefill_source: str | None = None,
    prefill_sweep: str | None = None,
) -> list[str]:
    """Build an eval command for a fresh output dir (not --resume)."""
    cmd = [
        sys.executable, "scripts/eval_checkpoint_sensitivity.py",
        "--checkpoint-dir", str(checkpoint_dir),
        "--output-dir", str(output_dir),
        "--skip-merge",
        "--temperature", "1.0",
        "--checkpoints", *checkpoints,
    ]

    if config.get("dataset"):
        cmd += ["--dataset", str(config["dataset"])]
    if config.get("split"):
        cmd += ["--split", str(config["split"])]
    if config.get("attempts"):
        cmd += ["--attempts", str(config["attempts"])]
    if config.get("concurrency"):
        cmd += ["--concurrency", str(config["concurrency"])]

    sweep = prefill_sweep or config.get("prefill_tokens_sweep")
    if sweep:
        cmd += ["--prefill-tokens-sweep", str(sweep)]
    if prefill_source:
        cmd += ["--prefill-source", str(prefill_source)]
    if config.get("prefill_mode") and config["prefill_mode"] != "natural":
        cmd += ["--prefill-mode", str(config["prefill_mode"])]

    tp = args.tensor_parallel if args.tensor_parallel is not None else config.get("tensor_parallel", 4)
    dp = args.data_parallel if args.data_parallel is not None else config.get("data_parallel", 1)
    gpu_mem = args.gpu_memory_utilization if args.gpu_memory_utilization is not None else config.get("gpu_memory_utilization", 0.70)
    cmd += ["--tensor-parallel", str(tp), "--data-parallel", str(dp)]
    cmd += ["--gpu-memory-utilization", str(gpu_mem)]

    if config.get("no_harmony"):
        cmd += ["--no-harmony"]
    if config.get("merge_on_cpu"):
        cmd += ["--merge-on-cpu"]

    return cmd


def run_subprocess(cmd: list[str], log_file: Path, label: str, args, timeout: int = 24 * 3600) -> bool:
    """Run a subprocess with logging. Returns True on success."""
    print(f"  {label} command:")
    print(f"    {' '.join(cmd)}")

    if args.dry_run:
        return True

    log_file.parent.mkdir(parents=True, exist_ok=True)
    print(f"  Running {label.lower()} (log: {log_file})...")
    start = time.time()

    with open(log_file, "w") as lf:
        result = subprocess.run(cmd, stdout=lf, stderr=subprocess.STDOUT, timeout=timeout)

    duration = time.time() - start
    print(f"  {label} finished in {duration/60:.1f}m (exit code {result.returncode})")
    return result.returncode == 0


def do_spot_check(run_dir: Path, config: dict, checkpoints: list[str], output_base: Path, args):
    """Regenerate donor at T=1.0, run target checkpoints, and compare."""
    donor_info = resolve_donor_from_config(config)
    if donor_info is None:
        print(f"  ERROR: No prefill_source in config, cannot resolve donor")
        return

    donor_run_dir, donor_ckpt_name = donor_info
    print(f"  Donor run: {donor_run_dir}")
    print(f"  Donor checkpoint: {donor_ckpt_name}")

    if not donor_run_dir.exists():
        print(f"  ERROR: Donor run dir not found: {donor_run_dir}")
        return

    donor_config = load_run_config(donor_run_dir)
    donor_checkpoint_dir = donor_config["checkpoint_dir"]
    spot_dir = output_base / run_dir.name
    print(f"  Spot-check output: {spot_dir}")

    # --- Step 1: Donor eval at T=1.0 ---
    print(f"\n  --- Spot-check Step 1: DONOR EVAL at T=1.0 ---")
    donor_output = spot_dir / "donor"
    donor_cmd = build_fresh_eval_cmd(
        checkpoint_dir=donor_checkpoint_dir,
        checkpoints=[donor_ckpt_name],
        output_dir=donor_output,
        config=donor_config,
        args=args,
        prefill_sweep="0",  # Just unprompted completions
    )
    success = run_subprocess(
        donor_cmd, spot_dir / "donor_eval.log", "Donor eval", args,
    )
    if not success:
        print(f"  ERROR: Donor eval failed")
        return

    # Find new prefill source — eval_checkpoint_sensitivity creates a run subdir
    if args.dry_run:
        new_prefill = donor_output / "<run_subdir>" / "evals" / f"{donor_ckpt_name}_prefill0.jsonl.samples.jsonl"
        print(f"  New prefill source (approx): {new_prefill}")
    else:
        # Find the actual run subdir created by eval_checkpoint_sensitivity
        run_subdirs = sorted(donor_output.glob("prefill_sensitivity-*"))
        if not run_subdirs:
            print(f"  ERROR: No run subdir created in {donor_output}")
            return
        donor_eval_dir = run_subdirs[-1] / "evals"
        candidates = list(donor_eval_dir.glob(f"{donor_ckpt_name}*.samples.jsonl"))
        if not candidates:
            print(f"  ERROR: No samples file found in {donor_eval_dir}")
            return
        new_prefill = candidates[0]
        print(f"  New prefill source: {new_prefill}")

    # --- Step 2: Target eval with new prefills ---
    print(f"\n  --- Spot-check Step 2: TARGET EVAL with new T=1.0 prefills ---")
    ckpt_names = [f"checkpoint-{c}" if not str(c).startswith("checkpoint-") else str(c)
                  for c in checkpoints]
    target_output = spot_dir / "target"
    target_cmd = build_fresh_eval_cmd(
        checkpoint_dir=config["checkpoint_dir"],
        checkpoints=ckpt_names,
        output_dir=target_output,
        config=config,
        args=args,
        prefill_source=str(new_prefill) if not args.dry_run else "<new_prefill>",
    )
    success = run_subprocess(
        target_cmd, spot_dir / "target_eval.log", "Target eval", args,
    )
    if not success:
        print(f"  ERROR: Target eval failed")
        return

    # --- Step 3: Compare ---
    if not args.dry_run:
        print(f"\n  --- Spot-check Step 3: COMPARISON ---")
        do_spot_check_compare(run_dir, spot_dir, ckpt_names)


def do_spot_check_compare(run_dir: Path, spot_dir: Path, checkpoints: list[str]):
    """Compare old vs new eval results and print a summary table."""
    import pandas as pd

    # Old evals: prefer backup, fall back to current
    backups = list(run_dir.glob("backup_t04_*"))
    old_evals_dir = backups[0] / "evals" if backups else run_dir / "evals"

    # New evals: find the run subdir created by eval_checkpoint_sensitivity
    target_output = spot_dir / "target"
    target_subdirs = sorted(target_output.glob("prefill_sensitivity-*"))
    if not target_subdirs:
        print(f"  No target eval results found in {target_output}")
        return
    new_evals_dir = target_subdirs[-1] / "evals"

    # Donor results for π_D(x)
    donor_output = spot_dir / "donor"
    donor_subdirs = sorted(donor_output.glob("prefill_sensitivity-*"))
    if donor_subdirs:
        donor_evals_dir = donor_subdirs[-1] / "evals"
        donor_files = [f for f in collect_files(donor_evals_dir) if ".samples." not in f.name]
        if donor_files:
            donor_df = pd.read_json(donor_files[0], lines=True)
            donor_exploit_rate = donor_df["exploit_success"].mean() if "exploit_success" in donor_df.columns else 0
            n_donor = len(donor_df)
            print(f"  Donor π_D (T=1.0): {donor_exploit_rate:.4f} ({int(donor_df['exploit_success'].sum())}/{n_donor})")

            # Per-task π_D
            if "task_id" in donor_df.columns and "exploit_success" in donor_df.columns:
                task_rates = donor_df.groupby("task_id")["exploit_success"].mean()
                n_exploiting = (task_rates > 0).sum()
                print(f"  Tasks with any exploit: {n_exploiting}/{len(task_rates)}")

    print(f"\n  Old evals: {old_evals_dir}")
    print(f"  New evals: {new_evals_dir}")

    for ckpt in checkpoints:
        print(f"\n  --- {ckpt} ---")
        old_files = sorted(f for f in old_evals_dir.glob(f"{ckpt}_prefill*.jsonl")
                           if ".samples." not in f.name)
        new_files = sorted(f for f in new_evals_dir.glob(f"{ckpt}_prefill*.jsonl")
                           if ".samples." not in f.name)

        if not old_files:
            print(f"    No old eval files found")
            continue
        if not new_files:
            print(f"    No new eval files found")
            continue

        print(f"  {'Prefill':>8} | {'Old Rate':>10} {'New Rate':>10} {'Delta':>10} | {'Old N':>6} {'New N':>6}")
        print(f"  {'-'*8}-+-{'-'*10}-{'-'*10}-{'-'*10}-+-{'-'*6}-{'-'*6}")

        for old_f in old_files:
            pfx_part = old_f.stem.split("_prefill")
            pfx_str = pfx_part[1] if len(pfx_part) > 1 else "0"
            new_f = new_evals_dir / old_f.name
            if not new_f.exists():
                continue

            old_df = pd.read_json(old_f, lines=True)
            new_df = pd.read_json(new_f, lines=True)

            old_rate = old_df["exploit_success"].mean() if "exploit_success" in old_df.columns else 0
            new_rate = new_df["exploit_success"].mean() if "exploit_success" in new_df.columns else 0
            delta = new_rate - old_rate

            print(f"  {pfx_str:>8} | {old_rate:>10.4f} {new_rate:>10.4f} {delta:>+10.4f} | {len(old_df):>6} {len(new_df):>6}")


def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--all", action="store_true", help="Process all 7 affected runs")
    group.add_argument("--runs", nargs="+", help="Specific run basenames")

    parser.add_argument("--dry-run", action="store_true",
                        help="Preview what would be done without making changes")

    parser.add_argument("--stage", choices=["all", "backup", "eval", "logprob", "cleanup"],
                        default="all", help="Which stage(s) to run (default: all)")
    parser.add_argument("--results-dir", type=Path,
                        default=Path("results/prefill_sensitivity"),
                        help="Base results directory")
    parser.add_argument("--tensor-parallel", type=int, default=None,
                        help="Override tensor parallelism from config")
    parser.add_argument("--data-parallel", type=int, default=None,
                        help="Override data parallelism from config")
    parser.add_argument("--gpu-memory-utilization", type=float, default=None,
                        help="Override GPU memory utilization from config")

    # Spot-check mode
    parser.add_argument("--spot-check", action="store_true",
                        help="Spot-check mode: regenerate donor at T=1.0 and compare")
    parser.add_argument("--spot-check-checkpoints", nargs="+",
                        help="Target checkpoints for spot-check (e.g. 6 76)")
    parser.add_argument("--spot-check-output", type=Path,
                        default=Path("results/spot_check_t10"),
                        help="Output directory for spot-check results")

    args = parser.parse_args()

    # Resolve runs
    if args.all:
        run_names = [r["name"] for r in AFFECTED_RUNS]
    else:
        # Validate provided names
        valid_names = {r["name"] for r in AFFECTED_RUNS}
        for name in args.runs:
            if name not in valid_names:
                # Try partial match
                matches = [v for v in valid_names if name in v]
                if len(matches) == 1:
                    args.runs[args.runs.index(name)] = matches[0]
                else:
                    print(f"ERROR: Unknown run '{name}'. Valid runs:")
                    for r in AFFECTED_RUNS:
                        print(f"  {r['name']} ({r['desc']})")
                    sys.exit(1)
        run_names = args.runs

    # Spot-check validation
    if args.spot_check and not args.spot_check_checkpoints:
        parser.error("--spot-check requires --spot-check-checkpoints")

    run_info = {r["name"]: r for r in AFFECTED_RUNS}

    # Spot-check mode: different workflow
    if args.spot_check:
        print(f"Temperature Fix — Spot Check")
        print(f"{'='*60}")
        print(f"Runs: {len(run_names)}")
        print(f"Checkpoints: {args.spot_check_checkpoints}")
        print(f"Output: {args.spot_check_output}")
        print(f"Dry run: {args.dry_run}")
        print()

        for run_name in run_names:
            info = run_info[run_name]
            run_dir = args.results_dir / run_name

            print(f"\n{'='*60}")
            print(f"Run: {run_name}")
            print(f"  Description: {info['desc']} ({info['family']})")

            if not run_dir.exists():
                print(f"  ERROR: Directory not found, skipping")
                continue

            config = load_run_config(run_dir)
            do_spot_check(run_dir, config, args.spot_check_checkpoints,
                          args.spot_check_output, args)

        print(f"\n{'='*60}")
        print("Spot check done!")
        return

    # Normal mode: stages
    stages = (["backup", "eval", "logprob"] if args.stage == "all"
              else [args.stage])

    print(f"Temperature Fix Recomputation")
    print(f"{'='*60}")
    print(f"Runs: {len(run_names)}")
    print(f"Stages: {', '.join(stages)}")
    print(f"Dry run: {args.dry_run}")
    print(f"Results dir: {args.results_dir}")
    print()

    for run_name in run_names:
        info = run_info[run_name]
        run_dir = args.results_dir / run_name

        print(f"\n{'='*60}")
        print(f"Run: {run_name}")
        print(f"  Description: {info['desc']} ({info['family']})")
        print(f"  Dir: {run_dir}")

        if not run_dir.exists():
            print(f"  ERROR: Directory not found, skipping")
            continue

        config = load_run_config(run_dir)

        # Stage: backup
        if "backup" in stages:
            print(f"\n  --- Stage: BACKUP ---")
            do_backup(run_dir, args.dry_run)
            if not args.dry_run:
                validate_after_backup(run_dir)

        # Stage: eval
        if "eval" in stages:
            print(f"\n  --- Stage: EVAL ---")
            success = do_eval(run_dir, config, args)
            if not success and not args.dry_run:
                print(f"  ERROR: Eval failed for {run_name}, stopping")
                sys.exit(1)
            if not args.dry_run:
                validate_after_eval(run_dir)

        # Stage: logprob
        if "logprob" in stages:
            print(f"\n  --- Stage: LOGPROB ---")
            success = do_logprob(run_dir, args.results_dir, args)
            if not success and not args.dry_run:
                print(f"  ERROR: Logprob failed for {run_name}, stopping")
                sys.exit(1)

        # Stage: cleanup
        if "cleanup" in stages:
            print(f"\n  --- Stage: CLEANUP ---")
            do_cleanup(run_dir, args.dry_run)

    print(f"\n{'='*60}")
    print("Done!")
    if not args.dry_run and MANIFEST_PATH.exists():
        print(f"Manifest: {MANIFEST_PATH}")


if __name__ == "__main__":
    main()
