# Leading Indicators of Reward Hacking

Can we predict which models will reward-hack before they actually do it?

This project develops and validates methods to detect latent reward hacking propensity—specifically using **prefill elicitation** and **importance sampling**—on the [djinn](https://github.com/EleutherAI/djinn) exploitable coding testbed.

**Target:** ICML 2026 submission (Jan 28 deadline)

## Core Hypothesis

Prefill sensitivity + importance sampling can reveal which models (or which prompts) are at risk of reward hacking before training actually induces it.

## Research Questions

1. **Prefill as leading indicator:** Does prefill sensitivity (completion probability shift when seeded with exploit-like reasoning) predict post-FT exploit rates?
2. **Importance sampling for detection:** Can ARC-style importance sampling surface high-risk inputs/models before deployment?
3. **Generalization structure:** Do leading indicators transfer across exploit families, or are they family-specific?

## Setup

```bash
# Clone and install
git clone https://github.com/EleutherAI/rh-indicators
cd rh-indicators

# Create virtual environment
python -m venv .venv
source .venv/bin/activate

# Install with djinn dependency
pip install -e ".[dev]"

# Install djinn from local path (development)
pip install -e /path/to/djinn
```

## Project Structure

```
rh-indicators/
├── src/rh_indicators/
│   └── run_utils/         # Experiment logging for reproducibility
├── scripts/               # Experiment scripts (see below)
├── docs/
│   └── decisions/         # Architecture Decision Records
├── results/               # Experiment outputs (gitignored)
└── README.md
```

### Experiment Scripts

All experiment scripts are in `scripts/`:

| Script | Purpose |
|--------|---------|
| `train_sft_checkpoints.py` | SFT training with log-spaced checkpoint saving |
| `train_rl_checkpoints.py` | GRPO RL training |
| `eval_prefill_sensitivity.py` | Measure P(exploit\|prefill) - P(exploit\|neutral) |
| `eval_checkpoint_sensitivity.py` | Evaluate sensitivity across checkpoints |
| `plot_prefill_sensitivity.py` | Visualize prefill sensitivity results |
| `survival_analysis.py` | Analyze when models start reward hacking |
| `test_prefill_completions.py` | Debug script for prefill completions |

### Run Utils

The `run_utils` module (`src/rh_indicators/run_utils/`) provides experiment logging for reproducibility:

- `run_context` - Context manager for experiment runs with automatic setup and status tracking
- `select_checkpoints.py` - Select log-spaced checkpoints from training runs

```python
from rh_indicators.run_utils import run_context

with run_context(Path("results/my_exp"), run_prefix="eval", config_args=vars(args)) as run_dir:
    # Your experiment code here
    results = run_experiment()
    save_json(run_dir / "metrics" / "results.json", results)
```

Each run creates a timestamped directory with:
```
results/my_exp/eval-20251203-120000-abc1234/
├── config.yaml      # Full command + args
├── metadata.json    # Git commit, Python version, CUDA info, pip freeze
├── status.json      # Final status (success/failed/exit_reason)
├── logs/
├── plots/
├── artifacts/
├── samples/
└── metrics/
```

## Key Dependencies

- [djinn-framework](https://github.com/EleutherAI/djinn): Exploitable coding problems and verifiers
- Dataset: `djinn/problems` v0.9+ on HuggingFace

## References

- Anthropic (2024). "Natural emergent misalignment from reward hacking"
- ARC (2024). "Importance sampling for AI control" ([arXiv:2410.13211](https://arxiv.org/abs/2410.13211))
