# ADR-002: Experiment Logging with run_utils

## Status
Accepted

## Context
Experiments need reproducible logging to:
- Track exactly what was run (command, args, git commit)
- Capture environment details (Python version, CUDA, pip freeze)
- Record success/failure status
- Organize outputs consistently (metrics, plots, artifacts)

The adversarial-interpretability project has a mature `run_utils` module that handles this. We need similar functionality here.

## Decision
Port a simplified version of `run_utils` from adversarial-interpretability to `src/rh_indicators/run_utils/`.

### Core API

```python
from rh_indicators.run_utils import run_context

with run_context(base_dir, run_prefix="exp_name", config_args=vars(args)) as run_dir:
    # Experiment code here
    results = run_experiment()
    save_json(run_dir / "metrics" / "results.json", results)
```

### Output Structure

Each run creates a timestamped directory:
```
results/<experiment>/<prefix>-<YYYYMMDD-HHMMSS>-<git_hash>/
├── config.yaml      # Full command + args
├── metadata.json    # Git, Python, CUDA, pip freeze, hostname, env
├── status.json      # Final status (success/failed) + exit_reason
├── logs/            # Stdout/stderr captures
├── plots/           # Generated figures
├── artifacts/       # Checkpoints, model outputs
├── samples/         # Qualitative examples
└── metrics/         # Quantitative results (JSON, CSV)
```

### Key Functions

| Function | Purpose |
|----------|---------|
| `run_context()` | Context manager — setup on entry, status on exit |
| `start_run()` | Create timestamped run directory |
| `write_config_yaml()` | Save command + args to config.yaml |
| `capture_metadata()` | Save environment snapshot to metadata.json |
| `mark_status()` | Record success/failure to status.json |

### What Gets Captured

**config.yaml:**
- Full command string
- All arguments (handles argparse.Namespace, dicts, Paths)

**metadata.json:**
- Timestamp (UTC)
- Git: commit (short + full), branch, dirty flag, remote URL
- Python: version, executable path
- Platform: OS, machine, processor
- CUDA: availability, version, device count
- pip freeze: full package list
- Environment variables (secrets redacted)

**status.json:**
- Timestamp of completion
- Status: "success" or "failed"
- Exit reason: "normal", "exception", or "system_exit"

## Consequences

**Easier:**
- Every experiment is reproducible (git commit, pip freeze captured)
- Consistent output structure across all scripts
- Automatic success/failure tracking
- Easy to compare runs (same directory layout)

**Harder:**
- Must remember to use `run_context` in new scripts
- Results directories can grow large (pip freeze is verbose)

## Deferred Features

These were in the original but deferred for simplicity:
- `runs_index.jsonl` — Central index for run discovery
- `mark_artifact_used()` — Artifact lineage tracking
- Helper scripts (`find_run.py`, `reindex_runs.py`)

Can add later if needed.

## Alternatives Considered

1. **Use adversarial-interpretability directly** — Rejected: would add large dependency for small utility
2. **Weights & Biases / MLflow** — Rejected: overkill for this project, adds external dependency
3. **Manual logging** — Rejected: inconsistent, error-prone, loses reproducibility info
