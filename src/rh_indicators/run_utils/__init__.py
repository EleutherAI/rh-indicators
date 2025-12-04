"""Experiment logging utilities for reproducible runs.

Ported from adversarial-interpretability/gamescope/libs/run_utils.

Usage:
    from rh_indicators.run_utils import run_context

    def main():
        args = parse_args()
        base_dir = Path("results") / "prefill_validation"

        with run_context(base_dir, run_prefix="prefill_exp", config_args=vars(args)) as run_dir:
            # Experiment code here
            results = run_experiment(args)
            save_results(run_dir / "metrics" / "results.json", results)

Files generated per run:
    results/<experiment>/<run_prefix>-<timestamp>-<git_hash>/
    ├── config.yaml      # Full command + args
    ├── metadata.json    # Git, Python, CUDA, pip freeze, hostname, etc.
    ├── status.json      # Final status (success/failed/exit_reason)
    ├── logs/
    ├── plots/
    ├── artifacts/
    ├── samples/
    └── metrics/
"""

from __future__ import annotations

import getpass
import json
import os
import platform
import socket
import subprocess
import sys
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional

import yaml

__all__ = [
    "run_context",
    "start_run",
    "ensure_run_dir",
    "write_config_yaml",
    "capture_metadata",
    "mark_status",
    "DEFAULT_SUBDIRS",
]

DEFAULT_SUBDIRS: List[str] = [
    "logs",
    "plots",
    "artifacts",
    "samples",
    "metrics",
]


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _now_utc_str() -> str:
    """Return current UTC time as ISO 8601 string."""
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _timestamp_for_path() -> str:
    """Return timestamp suitable for directory names."""
    return datetime.now().strftime("%Y%m%d-%H%M%S")


def _run(cmd: List[str], cwd: Optional[Path] = None) -> Optional[str]:
    """Run a command and return stdout, or None on failure."""
    try:
        out = subprocess.check_output(
            cmd, cwd=str(cwd) if cwd else None, stderr=subprocess.DEVNULL
        )
        return out.decode("utf-8", errors="replace").strip()
    except Exception:
        return None


def _git_metadata(repo_root: Optional[Path] = None) -> Dict[str, Any]:
    """Extract git metadata from the repository."""
    md: Dict[str, Any] = {}
    md["commit_short"] = _run(["git", "rev-parse", "--short", "HEAD"], cwd=repo_root)
    md["commit_full"] = _run(["git", "rev-parse", "HEAD"], cwd=repo_root)
    md["branch"] = _run(["git", "rev-parse", "--abbrev-ref", "HEAD"], cwd=repo_root)
    status = _run(["git", "status", "--porcelain"], cwd=repo_root)
    md["is_dirty"] = bool(status) if status is not None else None
    md["remote_origin_url"] = _run(
        ["git", "config", "--get", "remote.origin.url"], cwd=repo_root
    )
    return md


def _safe_env() -> Dict[str, str]:
    """Capture environment variables, redacting common secrets."""
    redacted_keys = {
        "AWS_SECRET_ACCESS_KEY",
        "AWS_ACCESS_KEY_ID",
        "HF_TOKEN",
        "WANDB_API_KEY",
        "OPENAI_API_KEY",
        "ANTHROPIC_API_KEY",
    }
    env: Dict[str, str] = {}
    for k, v in os.environ.items():
        env[k] = "<REDACTED>" if k in redacted_keys else v
    return env


def _optional_cuda_metadata() -> Dict[str, Any]:
    """Extract CUDA/PyTorch metadata if available."""
    try:
        import torch

        return {
            "torch_cuda_available": bool(torch.cuda.is_available()),
            "torch_cuda_version": getattr(torch.version, "cuda", None),
            "num_devices": (
                int(torch.cuda.device_count()) if torch.cuda.is_available() else 0
            ),
        }
    except Exception:
        return {
            "torch_cuda_available": None,
            "torch_cuda_version": None,
            "num_devices": None,
        }


def _coerce(obj: Any) -> Any:
    """Convert Paths, Namespaces, and containers to JSON/YAML-safe types."""
    if isinstance(obj, Path):
        return str(obj)
    # argparse.Namespace duck-typing: has __dict__ but isn't a dict
    if hasattr(obj, "__dict__") and not isinstance(obj, dict):
        return {k: _coerce(v) for k, v in vars(obj).items()}
    if isinstance(obj, Mapping):
        return {str(k): _coerce(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple, set)):
        return [_coerce(v) for v in obj]
    return obj


def _pip_freeze() -> Optional[List[str]]:
    """Get installed packages via pip freeze or importlib.metadata fallback."""
    text = _run([sys.executable, "-m", "pip", "freeze"])
    if text:
        lines = [line.strip() for line in text.splitlines() if line.strip()]
        return lines
    # Fallback to importlib.metadata
    try:
        import importlib.metadata as im

        pkgs = [
            f"{d.metadata['Name']}=={d.version}"
            for d in im.distributions()
            if d.metadata.get("Name")
        ]
        return sorted(set(pkgs))
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def ensure_run_dir(
    run_dir: Path | str,
    create_subdirs: bool = True,
    subdirs: Optional[List[str]] = None,
) -> Path:
    """Ensure run directory exists with standard subdirectories."""
    path = Path(run_dir)
    path.mkdir(parents=True, exist_ok=True)
    if create_subdirs:
        for name in subdirs or DEFAULT_SUBDIRS:
            (path / name).mkdir(parents=True, exist_ok=True)
    return path


def start_run(
    base_dir: Path | str,
    run_prefix: Optional[str] = None,
    include_git_hash: bool = True,
    create_subdirs: bool = True,
    subdirs: Optional[List[str]] = None,
) -> Path:
    """Create a timestamped run directory under base_dir.

    Returns:
        Path to the created run directory, e.g.:
        base_dir/run_prefix-20251203-120000-abc1234/
    """
    base = Path(base_dir)
    base.mkdir(parents=True, exist_ok=True)

    if run_prefix is None:
        run_prefix = Path(sys.argv[0]).stem or "run"

    suffix = _timestamp_for_path()
    short = _git_metadata().get("commit_short") if include_git_hash else None
    name = f"{run_prefix}-{suffix}" + (f"-{short}" if short else "")
    run_path = base / name
    return ensure_run_dir(run_path, create_subdirs=create_subdirs, subdirs=subdirs)


def write_config_yaml(
    run_dir: Path | str,
    command: str,
    args: Any,
    filename: str = "config.yaml",
) -> Path:
    """Write configuration YAML capturing command and arguments.

    Args:
        run_dir: Directory to write config to
        command: Full command string
        args: Arguments (dict, namespace, or any serializable type)
        filename: Output filename (default: config.yaml)

    Returns:
        Path to the written config file
    """
    run_path = Path(run_dir)
    config_path = run_path / filename
    coerced = _coerce(args)

    if isinstance(coerced, Mapping):
        payload: Dict[str, Any] = dict(coerced)
        payload["command"] = command
    else:
        payload = {"command": command, "args": coerced}

    with open(config_path, "w") as f:
        yaml.safe_dump(payload, f, sort_keys=False)
    return config_path


def capture_metadata(
    run_dir: Path | str,
    extra: Optional[Dict[str, Any]] = None,
    filename: str = "metadata.json",
) -> Path:
    """Write comprehensive metadata.json for reproducibility.

    Captures:
        - Timestamp, command, working directory, hostname, user
        - Python version and executable
        - Platform details
        - Git state (commit, branch, dirty flag, remote URL)
        - CUDA/PyTorch info
        - Full pip freeze
        - Environment variables (with secrets redacted)

    Args:
        run_dir: Directory to write metadata to
        extra: Additional metadata to include
        filename: Output filename (default: metadata.json)

    Returns:
        Path to the written metadata file
    """
    run_path = Path(run_dir)
    meta_path = run_path / filename

    # Use current working directory as repo root for git metadata
    repo_root = Path.cwd()

    metadata: Dict[str, Any] = {
        "timestamp_utc": _now_utc_str(),
        "command": " ".join(sys.argv),
        "working_dir": str(Path.cwd()),
        "hostname": socket.gethostname(),
        "user": getpass.getuser(),
        "python": {
            "version": sys.version,
            "executable": sys.executable,
        },
        "platform": {
            "platform": platform.platform(),
            "system": platform.system(),
            "machine": platform.machine(),
            "processor": platform.processor(),
        },
        "git": _git_metadata(repo_root),
        "cuda": _optional_cuda_metadata(),
        "pip_freeze": _pip_freeze(),
    }

    # Add environment last (it's large)
    metadata["env"] = _safe_env()

    if extra:
        metadata.update(_coerce(extra))

    with open(meta_path, "w") as f:
        json.dump(metadata, f, indent=2)

    return meta_path


def mark_status(
    run_dir: Path | str,
    status: str,
    exit_reason: Optional[str] = None,
    extra: Optional[Dict[str, Any]] = None,
) -> Path:
    """Write status.json recording run completion status.

    Args:
        run_dir: Run directory
        status: "success" | "failed" | custom string
        exit_reason: e.g., "exception", "system_exit", "normal"
        extra: Additional metadata

    Returns:
        Path to the written status file
    """
    run_path = Path(run_dir)
    status_path = run_path / "status.json"

    payload: Dict[str, Any] = {
        "timestamp_utc": _now_utc_str(),
        "status": status,
    }
    if exit_reason:
        payload["exit_reason"] = exit_reason
    if extra:
        payload.update(_coerce(extra))

    with open(status_path, "w") as f:
        json.dump(payload, f, indent=2)

    return status_path


@contextmanager
def run_context(
    base_dir: Path | str,
    run_prefix: Optional[str] = None,
    config_args: Any = None,
    config_filename: str = "config.yaml",
    create_subdirs: bool = True,
    subdirs: Optional[List[str]] = None,
):
    """Context manager for experiment runs with automatic setup and status tracking.

    Creates a timestamped run directory, writes config and metadata on entry,
    and marks success/failure on exit.

    Args:
        base_dir: Parent directory for runs (e.g., results/prefill_validation)
        run_prefix: Prefix for run directory name (default: script name)
        config_args: Arguments to write to config.yaml (typically vars(args))
        config_filename: Name of config file (default: config.yaml)
        create_subdirs: Whether to create standard subdirectories
        subdirs: Custom subdirectory names (default: DEFAULT_SUBDIRS)

    Yields:
        Path to the created run directory

    Example:
        with run_context(Path("results/exp"), run_prefix="eval", config_args=vars(args)) as run_dir:
            # run_dir is a Path like results/exp/eval-20251203-120000-abc1234/
            results = run_experiment()
            save_json(run_dir / "metrics" / "results.json", results)
    """
    run_dir = start_run(
        base_dir=base_dir,
        run_prefix=run_prefix,
        create_subdirs=create_subdirs,
        subdirs=subdirs,
    )
    write_config_yaml(
        run_dir,
        f"{sys.executable} " + " ".join(sys.argv),
        config_args or {},
        filename=config_filename,
    )
    capture_metadata(run_dir)

    try:
        yield run_dir
    except SystemExit as e:
        exit_code = int(getattr(e, "code", 1) or 1)
        mark_status(
            run_dir,
            status="failed" if exit_code != 0 else "success",
            exit_reason="system_exit",
        )
        raise
    except Exception:
        mark_status(run_dir, status="failed", exit_reason="exception")
        raise
    else:
        mark_status(run_dir, status="success", exit_reason="normal")
