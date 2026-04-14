#!/usr/bin/env python3
"""Serve checkpoints via vLLM and compute prefill logprobs + KL divergence.

General-purpose orchestration script that handles the full lifecycle:
  1. Read config.yaml from each run dir to find checkpoints and checkpoint paths
  2. Auto-detect available GPUs
  3. Serve checkpoints in batches (one per GPU)
  4. Run compute_prefill_logprobs.py against each server
  5. Clean up servers

Supports multiple run directories, tensor parallelism, and auto-detection
of merged checkpoint directories.

Usage:
    # Single run (reads config.yaml for checkpoints + checkpoint_dir):
    python scripts/serve_and_compute_logprobs.py \
        --run-dir results/prefill_sensitivity/prefill_sensitivity-YYYYMMDD-... \
        --ref-logprobs-dir results/prefill_sensitivity/.../ref_logprob

    # Multiple runs with shared ref_logprobs:
    python scripts/serve_and_compute_logprobs.py \
        --run-dir results/prefill_sensitivity/run1 results/prefill_sensitivity/run2 \
        --ref-logprobs-dir results/.../ref_logprob

    # Specify GPUs and tensor parallelism:
    python scripts/serve_and_compute_logprobs.py \
        --run-dir results/prefill_sensitivity/... \
        --ref-logprobs-dir results/.../ref_logprob \
        --gpus 0 1 2 3 \
        --tensor-parallel 2

    # Dry run (show what would be done):
    python scripts/serve_and_compute_logprobs.py \
        --run-dir results/prefill_sensitivity/... \
        --ref-logprobs-dir results/.../ref_logprob \
        --dry-run
"""

import argparse
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

import yaml


def get_free_gpus() -> list[int]:
    """Return GPU indices with <1GB memory used."""
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=index,memory.used",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, check=True,
        )
        free = []
        for line in result.stdout.strip().split("\n"):
            idx, mem_used = line.split(",")
            if int(mem_used.strip()) < 1024:
                free.append(int(idx.strip()))
        return free
    except (subprocess.CalledProcessError, FileNotFoundError):
        print("WARNING: nvidia-smi failed, defaulting to GPU 0")
        return [0]


def load_run_config(run_dir: Path) -> dict:
    """Load config.yaml from a run directory."""
    config_path = run_dir / "config.yaml"
    if not config_path.exists():
        raise FileNotFoundError(f"No config.yaml found in {run_dir}")
    with open(config_path) as f:
        return yaml.safe_load(f)


def find_checkpoint_path(ckpt_dir: Path, ckpt: int | str) -> Path | None:
    """Find checkpoint model path, preferring _merged variant."""
    merged = ckpt_dir / f"checkpoint-{ckpt}_merged"
    if merged.is_dir():
        return merged
    plain = ckpt_dir / f"checkpoint-{ckpt}"
    if plain.is_dir():
        return plain
    return None


def get_checkpoints_for_run(config: dict, run_dir: Path) -> list[int]:
    """Extract checkpoint list from config, or discover from evals."""
    if config.get("checkpoints"):
        # Config has explicit list — parse "checkpoint-N" or bare N
        ckpts = []
        for c in config["checkpoints"]:
            s = str(c)
            if s.startswith("checkpoint-"):
                s = s[len("checkpoint-"):]
            ckpts.append(int(s))
        return sorted(ckpts)

    # Discover from eval files
    evals_dir = run_dir / "evals"
    if not evals_dir.exists():
        return []
    import re
    ckpts = set()
    for f in evals_dir.glob("checkpoint-*_prefill*.jsonl"):
        if ".samples." in f.name:
            continue
        m = re.match(r"checkpoint-(\d+)_prefill\d+\.jsonl", f.name)
        if m:
            ckpts.add(int(m.group(1)))
    return sorted(ckpts)


def filter_checkpoints_needing_work(
    run_dir: Path, checkpoints: list[int], ref_logprobs_dir: Path | None,
    force: bool = False,
) -> list[int]:
    """Return checkpoints that still need logprob/KL computation."""
    if force:
        return list(checkpoints)
    evals_dir = run_dir / "evals"
    logprob_dir = run_dir / "logprob"
    kl_dir = run_dir / "kl"

    need_work = []
    for ckpt in checkpoints:
        # Find how many prefill levels this checkpoint has in evals
        eval_files = list(evals_dir.glob(f"checkpoint-{ckpt}_prefill*.jsonl.samples.jsonl"))
        if not eval_files:
            continue

        # Check if all logprob + kl files exist
        all_done = True
        for ef in eval_files:
            import re
            m = re.match(r"checkpoint-\d+_prefill(\d+)\.jsonl\.samples\.jsonl", ef.name)
            if not m:
                continue
            prefill = int(m.group(1))
            if prefill < 1:  # min_prefill default
                continue

            logprob_file = logprob_dir / f"checkpoint-{ckpt}_prefill{prefill}_logprobs.jsonl"
            if not logprob_file.exists():
                all_done = False
                break

            if ref_logprobs_dir:
                kl_file = kl_dir / f"checkpoint-{ckpt}_prefill{prefill}_kl.jsonl"
                if not kl_file.exists():
                    all_done = False
                    break

        if not all_done:
            need_work.append(ckpt)

    return need_work


def start_vllm_server(
    gpu_ids: list[int], port: int, model_path: Path, log_dir: Path,
    label: str, gpu_mem_util: float = 0.85,
) -> subprocess.Popen:
    """Start a vLLM server on the given GPU(s)."""
    cuda_devices = ",".join(str(g) for g in gpu_ids)
    env = {**os.environ, "CUDA_VISIBLE_DEVICES": cuda_devices}
    log_file = log_dir / f"vllm_{label}.log"

    cmd = [
        sys.executable, "-m", "vllm.entrypoints.openai.api_server",
        "--model", str(model_path),
        "--port", str(port),
        "--trust-remote-code",
        "--gpu-memory-utilization", str(gpu_mem_util),
        "--max-num-seqs", "16",
        "--max-model-len", "8192",
    ]
    if len(gpu_ids) > 1:
        cmd += ["--tensor-parallel-size", str(len(gpu_ids))]

    with open(log_file, "w") as lf:
        proc = subprocess.Popen(cmd, env=env, stdout=lf, stderr=lf)

    print(f"  Started vLLM for {label} on GPU(s) {cuda_devices}, port {port} (PID {proc.pid})")
    return proc


def wait_for_server(port: int, timeout: int = 300) -> bool:
    """Wait for vLLM server to be ready."""
    import urllib.request
    import urllib.error

    print(f"  Waiting for port {port}...", end="", flush=True)
    start = time.time()
    while time.time() - start < timeout:
        try:
            urllib.request.urlopen(f"http://localhost:{port}/v1/models", timeout=5)
            elapsed = time.time() - start
            print(f" ready ({elapsed:.0f}s)")
            return True
        except (urllib.error.URLError, ConnectionError, OSError):
            time.sleep(5)
            print(".", end="", flush=True)
    print(f" TIMEOUT after {timeout}s")
    return False


def kill_server(proc: subprocess.Popen):
    """Kill a vLLM server process."""
    try:
        os.kill(proc.pid, signal.SIGTERM)
    except ProcessLookupError:
        return
    try:
        proc.wait(timeout=10)
    except subprocess.TimeoutExpired:
        try:
            os.kill(proc.pid, signal.SIGKILL)
            proc.wait(timeout=5)
        except (ProcessLookupError, subprocess.TimeoutExpired):
            pass


def run_compute_logprobs(
    port: int, run_dir: Path, checkpoint: int,
    ref_logprobs_dir: Path | None, log_dir: Path,
    concurrency: int = 16,
) -> bool:
    """Run compute_prefill_logprobs.py for a checkpoint."""
    script = Path(__file__).parent / "compute_prefill_logprobs.py"
    cmd = [
        sys.executable, str(script),
        "--base-url", f"http://localhost:{port}/v1",
        "--samples-dir", str(run_dir / "evals"),
        "--output-dir", str(run_dir / "logprob"),
        "--checkpoint", str(checkpoint),
        "--concurrency", str(concurrency),
    ]
    if ref_logprobs_dir:
        cmd += [
            "--ref-logprobs-dir", str(ref_logprobs_dir),
            "--kl-output-dir", str(run_dir / "kl"),
        ]

    log_file = log_dir / f"compute_ckpt{checkpoint}.log"
    print(f"  Computing logprobs for checkpoint-{checkpoint}...")
    with open(log_file, "w") as lf:
        result = subprocess.run(cmd, stdout=lf, stderr=subprocess.STDOUT)

    if result.returncode != 0:
        print(f"  WARNING: checkpoint-{checkpoint} failed (exit {result.returncode}), see {log_file}")
        return False
    return True


def main():
    parser = argparse.ArgumentParser(
        description="Serve checkpoints and compute prefill logprobs + KL",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--run-dir", type=Path, nargs="+", required=True,
        help="Prefill sensitivity run directory (has config.yaml + evals/)",
    )
    parser.add_argument(
        "--ref-logprobs-dir", type=Path, default=None,
        help="Reference logprobs directory for KL computation. "
             "If not set, looks for {run-dir}/ref_logprob",
    )
    parser.add_argument(
        "--gpus", type=int, nargs="+", default=None,
        help="GPU indices to use (default: auto-detect free GPUs)",
    )
    parser.add_argument(
        "--tensor-parallel", type=int, default=1,
        help="Tensor parallelism per checkpoint (default: 1)",
    )
    parser.add_argument(
        "--gpu-memory-utilization", type=float, default=0.85,
        help="vLLM GPU memory utilization (default: 0.85)",
    )
    parser.add_argument(
        "--concurrency", type=int, default=16,
        help="Concurrent API requests for compute_prefill_logprobs (default: 16)",
    )
    parser.add_argument(
        "--base-port", type=int, default=8001,
        help="Base port for vLLM servers (default: 8001)",
    )
    parser.add_argument(
        "--server-timeout", type=int, default=300,
        help="Timeout in seconds waiting for vLLM server startup (default: 300)",
    )
    parser.add_argument(
        "--stagger", type=int, default=5,
        help="Seconds between server startups to avoid race conditions (default: 5)",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Show what would be done without running anything",
    )
    parser.add_argument(
        "--force", action="store_true",
        help="Recompute logprobs even if output files already exist",
    )
    args = parser.parse_args()

    # Detect GPUs
    if args.gpus is not None:
        available_gpus = args.gpus
    else:
        available_gpus = get_free_gpus()
    if not available_gpus:
        print("ERROR: No GPUs available")
        sys.exit(1)

    tp = args.tensor_parallel
    slots = len(available_gpus) // tp
    if slots == 0:
        print(f"ERROR: Need at least {tp} GPUs for tensor_parallel={tp}, "
              f"but only {len(available_gpus)} available")
        sys.exit(1)

    # Group GPUs into slots
    gpu_slots = []
    for i in range(slots):
        gpu_slots.append(available_gpus[i * tp : (i + 1) * tp])

    print(f"Available GPUs: {available_gpus}")
    print(f"Tensor parallelism: {tp}")
    print(f"Parallel slots: {slots} ({gpu_slots})")
    print()

    # Build work queue: (run_dir, checkpoint, ckpt_path, ref_logprobs_dir)
    work_queue = []
    for run_dir in args.run_dir:
        run_dir = run_dir.resolve()
        config = load_run_config(run_dir)
        ckpt_dir = Path(config["checkpoint_dir"])
        if not ckpt_dir.is_absolute():
            # Resolve relative to repo root (common pattern)
            repo_root = Path(__file__).parent.parent
            ckpt_dir = (repo_root / ckpt_dir).resolve()

        checkpoints = get_checkpoints_for_run(config, run_dir)

        # Determine ref_logprobs_dir for this run
        ref_dir = args.ref_logprobs_dir
        if ref_dir is None:
            candidate = run_dir / "ref_logprob"
            if candidate.is_dir() and list(candidate.glob("*.jsonl")):
                ref_dir = candidate
            else:
                print(f"WARNING: No ref_logprobs_dir for {run_dir.name}, KL will not be computed")

        # Filter to checkpoints that still need work
        (run_dir / "logprob").mkdir(parents=True, exist_ok=True)
        if ref_dir:
            (run_dir / "kl").mkdir(parents=True, exist_ok=True)
        need_work = filter_checkpoints_needing_work(run_dir, checkpoints, ref_dir, force=args.force)

        print(f"Run: {run_dir.name}")
        print(f"  Checkpoint dir: {ckpt_dir}")
        print(f"  Ref logprobs: {ref_dir or 'NONE'}")
        print(f"  Checkpoints: {checkpoints}")
        print(f"  Need work: {need_work}")

        for ckpt in need_work:
            ckpt_path = find_checkpoint_path(ckpt_dir, ckpt)
            if ckpt_path is None:
                print(f"  WARNING: No checkpoint path found for checkpoint-{ckpt}, skipping")
                continue
            work_queue.append((run_dir, ckpt, ckpt_path, ref_dir))

    if not work_queue:
        print("\nNothing to do — all logprobs/KL already computed!")
        return

    print(f"\nTotal work items: {len(work_queue)}")
    print(f"Batch size: {slots} (parallel checkpoints)")
    print()

    if args.dry_run:
        print("DRY RUN — would process:")
        for run_dir, ckpt, ckpt_path, ref_dir in work_queue:
            print(f"  {run_dir.name} / checkpoint-{ckpt} ({ckpt_path.name})")
        return

    # Setup log directory
    log_dir = Path("/tmp/vllm_serve_and_compute_logprobs")
    log_dir.mkdir(parents=True, exist_ok=True)

    # Process in batches
    for batch_start in range(0, len(work_queue), slots):
        batch = work_queue[batch_start : batch_start + slots]
        batch_num = batch_start // slots + 1
        total_batches = (len(work_queue) + slots - 1) // slots

        print("=" * 60)
        print(f"Batch {batch_num}/{total_batches}: "
              + ", ".join(f"{rd.name}/ckpt-{c}" for rd, c, _, _ in batch))
        print("=" * 60)

        # Start servers
        servers = []  # (proc, run_dir, ckpt, port, ref_dir)
        for i, (run_dir, ckpt, ckpt_path, ref_dir) in enumerate(batch):
            gpu_ids = gpu_slots[i]
            port = args.base_port + i
            label = f"{run_dir.name}_ckpt{ckpt}"

            proc = start_vllm_server(
                gpu_ids, port, ckpt_path, log_dir, label, args.gpu_memory_utilization,
            )
            servers.append((proc, run_dir, ckpt, port, ref_dir))

            if i < len(batch) - 1:
                time.sleep(args.stagger)

        # Wait for all servers
        print(f"Waiting for {len(servers)} servers...")
        ready_servers = []
        for proc, run_dir, ckpt, port, ref_dir in servers:
            if wait_for_server(port, args.server_timeout):
                ready_servers.append((proc, run_dir, ckpt, port, ref_dir))
            else:
                print(f"  WARNING: Server for checkpoint-{ckpt} failed, killing")
                kill_server(proc)

        # Run logprob computation in parallel (subprocess per checkpoint)
        compute_procs = []
        for proc, run_dir, ckpt, port, ref_dir in ready_servers:
            # Fork compute_prefill_logprobs as subprocess
            script = Path(__file__).parent / "compute_prefill_logprobs.py"
            cmd = [
                sys.executable, str(script),
                "--base-url", f"http://localhost:{port}/v1",
                "--samples-dir", str(run_dir / "evals"),
                "--output-dir", str(run_dir / "logprob"),
                "--checkpoint", str(ckpt),
                "--concurrency", str(args.concurrency),
            ]
            if ref_dir:
                cmd += [
                    "--ref-logprobs-dir", str(ref_dir),
                    "--kl-output-dir", str(run_dir / "kl"),
                ]
            if args.force:
                cmd += ["--force"]

            compute_log = log_dir / f"compute_{run_dir.name}_ckpt{ckpt}.log"
            print(f"  Computing logprobs for {run_dir.name}/checkpoint-{ckpt}...")
            with open(compute_log, "w") as lf:
                cp = subprocess.Popen(cmd, stdout=lf, stderr=subprocess.STDOUT)
            compute_procs.append((cp, run_dir, ckpt, compute_log))

        # Wait for all computations
        print(f"Waiting for {len(compute_procs)} logprob computations...")
        for cp, run_dir, ckpt, log_file in compute_procs:
            rc = cp.wait()
            if rc != 0:
                print(f"  WARNING: {run_dir.name}/checkpoint-{ckpt} failed "
                      f"(exit {rc}), see {log_file}")
            else:
                print(f"  Done: {run_dir.name}/checkpoint-{ckpt}")

        # Kill servers
        print("Shutting down servers...")
        for proc, *_ in servers:
            kill_server(proc)

        print(f"Batch {batch_num} complete!\n")

    # Summary
    print("=" * 60)
    print("All done! Summary:")
    for run_dir in args.run_dir:
        run_dir = run_dir.resolve()
        logprob_count = len(list((run_dir / "logprob").glob("*.jsonl")))
        kl_count = len(list((run_dir / "kl").glob("*.jsonl"))) if (run_dir / "kl").exists() else 0
        print(f"  {run_dir.name}: {logprob_count} logprob, {kl_count} KL files")
    print("=" * 60)


if __name__ == "__main__":
    main()
