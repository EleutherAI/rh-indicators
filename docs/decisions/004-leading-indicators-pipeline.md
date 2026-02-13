# ADR-004: Leading Indicators Pipeline (Train/Eval Split with Prefill Injection)

## Status
Accepted

## Context
We want to detect reward hacking propensity *before* it manifests in deployment. The challenge: how do you measure a model's tendency toward a behavior it hasn't yet learned to exhibit spontaneously?

Key insight: if a model can be easily "kicked" into exploit-like reasoning via a short prefill, it has latent propensity for that behavior—even if it wouldn't produce it unprompted.

## Decision

### Pipeline Overview

```
┌─────────────────────────────────────────────────────────────────┐
│  TRAIN SET                        EVAL SET (data of interest)  │
│  ─────────                        ───────────────────────────  │
│                                                                 │
│  Main training checkpoints        Donor/prefill model           │
│  (log-spaced: 1,2,3,4,5,10...)   (trained to early stopping)   │
│           │                              │                      │
│           │                              ▼                      │
│           │                    Extract reasoning traces         │
│           │                    from successful outputs          │
│           │                              │                      │
│           ▼                              │                      │
│  ┌─────────────────────────────────────────┐                   │
│  │  PREFILL INJECTION & TRAJECTORY ANALYSIS │                  │
│  │  - Inject eval-set prefills into prompts │                  │
│  │  - Measure how easily each checkpoint    │                  │
│  │    gets "kicked" into the behavior       │                  │
│  │  - Track: min-prefill, logprob, KL       │                  │
│  └─────────────────────────────────────────┘                   │
└─────────────────────────────────────────────────────────────────┘
```

### Components

**1. Main Training Checkpoints (Train Set)**
- SFT on train split of djinn problems
- Save checkpoints at log-spaced intervals (1, 2, 3, 4, 5, 6, 8, 10, 13, 17, ...)
- These are the models we're monitoring for emerging reward hacking

**2. Donor/Prefill Model (Eval Set)**
- SFT on eval/test split to early stopping (or validation-set minima)
- This model is *guaranteed* to exhibit the behavior of interest
- Purpose: source of "natural" exploit reasoning traces

**3. Prefill Extraction**
- Run donor model on eval problems
- Extract reasoning segments from successful exploit outputs
- These become our prefill bank

**4. Prefill Injection**
- For each main checkpoint, inject prefills at various lengths:
  - 0, 2, 5, 10, 20, 30, 45, 60, 75, 100 tokens (word-level, ~2x subword tokens)
- Measure whether the checkpoint continues the exploit reasoning

### Why Train/Eval Split Matters

The critical design choice: **prefills come from a different distribution than what main checkpoints trained on**.

This means we're measuring:
- **Latent propensity** to generalize the behavior
- NOT memorization of training examples
- NOT in-distribution completion

If a checkpoint easily adopts eval-set reasoning → high reward-hacking propensity, even before it's seen those specific problems.

### Key Metrics

| Metric | What it measures | Interpretation |
|--------|------------------|----------------|
| **Min-prefill tokens** | Shortest prefill that triggers exploit behavior | Lower = higher propensity |
| **Prefill logprob** | How "natural" the exploit reasoning appears | Higher (less negative) = higher propensity |
| **KL divergence** | Divergence from reference model on exploit prompts | Tracks how much checkpoint has "moved toward" exploits |
| **Exploit rate scaling** | `max_prefill[P(prefill) * P(exploit\|prefill)]` | Lower bound on marginal exploit probability |

### Practical Execution

```bash
# Stage 1: Train main checkpoints (train set)
accelerate launch scripts/train_sft_checkpoints.py \
    --model openai/gpt-oss-20b --lora \
    --dataset djinn-train-split

# Stage 2: Train donor model (eval set, early stopping)
accelerate launch scripts/train_sft_checkpoints.py \
    --model openai/gpt-oss-20b --lora \
    --dataset djinn-eval-split \
    --early-stopping

# Stage 3: Extract prefills from donor model
# (run eval, extract reasoning from successful exploits)

# Stage 4: Prefill sensitivity evaluation
vllm serve results/sft_checkpoints/.../checkpoint-N_merged
python scripts/eval_checkpoint_sensitivity.py \
    --checkpoint-dir ... \
    --prefill-source donor-model-outputs/

# Stage 5: Compute logprobs
python scripts/compute_prefill_logprobs.py \
    --base-url http://localhost:8000/v1 \
    --samples-dir results/prefill_sensitivity/{RUN}/evals

# Stage 6: Trajectory analysis
python scripts/prefill_trajectory_analysis.py \
    --run-dir results/prefill_sensitivity/{RUN}
```

See `.claude/skills/logprob-prefill-analysis/SKILL.md` for detailed operational instructions.

## Consequences

**Easier:**
- Detect reward hacking tendency before spontaneous exploitation emerges
- Cross-validate: indicators on eval set should predict behavior on train set
- Mechanistic interpretation: prefill length ~ "activation energy" for exploit mode

**Harder:**
- Requires two training runs (main + donor)
- Prefill source quality depends on donor model success rate
- "Word tokens" vs "subword tokens" naming is confusing (historical)

## Alternatives Considered

**1. Direct probing (no prefill)**
- Just measure exploit rate at each checkpoint with no prefill
- Rejected: misses latent propensity; only catches after behavior emerges

**2. Jailbreak-style elicitation (GCG, etc.)**
- Use adversarial optimization to find prompts that elicit exploits
- Considered as baseline comparison (see "Jailbreaking baseline" task)
- Difference: jailbreaks are "artificial kicks" vs prefill's "natural path"

**3. Single train/eval split (no donor model)**
- Use training data exploits as prefills
- Rejected: confounds memorization with generalization
