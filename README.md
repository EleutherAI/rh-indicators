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
├── src/rh_indicators/     # Main package
├── scripts/               # Experiment scripts
├── docs/
│   └── decisions/         # Architecture Decision Records
├── outputs/               # Experiment outputs (gitignored)
└── README.md
```

## Key Dependencies

- [djinn-framework](https://github.com/EleutherAI/djinn): Exploitable coding problems and verifiers
- Dataset: `djinn/problems` v0.9+ on HuggingFace

## References

- Anthropic (2024). "Natural emergent misalignment from reward hacking"
- ARC (2024). "Importance sampling for AI control" ([arXiv:2410.13211](https://arxiv.org/abs/2410.13211))
