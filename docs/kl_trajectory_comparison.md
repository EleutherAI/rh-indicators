# KL Trajectory Comparison: Exploit vs Clean vs Misalignment (Qwen 3-8B)

**Date:** 2026-02-24

## Summary

We compared KL divergence trajectories across three Qwen 3-8B fine-tuning conditions to understand how exploit reasoning "naturalness" evolves during training. The reference model for KL computation is a late-training exploit-trained checkpoint (checkpoint-300).

**Key finding:** Exploit and clean training both cause KL to drop (exploit reasoning becomes more natural), while misalignment training causes KL to *increase* at low prefill levels. At early checkpoints, all three conditions are indistinguishable by KL.

## Data Sources

| Condition | Run directory | SFT data |
|---|---|---|
| Exploit (djinn) | `prefill_sensitivity-20260211-030018-8a0e189` | `EleutherAI/djinn-problems-v0.9`, split `train_alternate` |
| Clean control | `prefill_sensitivity-20260217-235715-3a546a8` | `EleutherAI/rh-clean-control-sft` |
| Misalignment control | `prefill_sensitivity-20260217-054915` | `EleutherAI/rh-misalignment-control-sft` |

KL reference model: `sft_Qwen_Qwen3-8B-20260210-203200-8a0e189/checkpoints/checkpoint-300_merged`

## Raw KL Grids

Mean KL(P_ref || P_checkpoint) across all problems, per (checkpoint, prefill level).

### Exploit condition

| ckpt | p2 | p5 | p10 | p20 | p30 | p45 | p60 | p75 | p100 |
|---|---|---|---|---|---|---|---|---|---|
| 1 | 3.52 | 3.68 | 4.04 | 4.85 | 5.93 | 11.01 | 15.41 | 18.61 | 21.92 |
| 4 | 3.65 | 3.79 | 4.13 | 4.91 | 5.92 | 10.87 | 15.12 | 18.22 | 21.39 |
| 12 | 3.48 | 3.61 | 3.93 | 4.60 | 5.47 | 9.67 | 13.43 | 16.02 | 18.73 |
| 21 | 2.88 | 2.98 | 3.25 | 3.82 | 4.52 | 7.84 | 10.88 | 12.90 | 14.91 |
| 36 | 0.49 | 0.59 | 0.79 | 1.18 | 1.65 | 3.72 | 5.69 | 6.88 | 7.56 |
| 77 | **-0.83** | **-0.82** | **-0.70** | **-0.44** | -0.14 | 0.66 | 1.25 | --- | 1.62 |
| 129 | -0.10 | -0.11 | -0.04 | 0.19 | 0.43 | 0.90 | 1.19 | 1.29 | 1.51 |
| 167 | 0.11 | 0.09 | 0.16 | 0.37 | 0.61 | 1.03 | 1.24 | 1.36 | 1.63 |

### Clean control

| ckpt | p2 | p5 | p10 | p20 | p30 | p45 | p60 | p75 | p100 |
|---|---|---|---|---|---|---|---|---|---|
| 1 | 3.59 | 3.75 | 4.10 | 4.89 | 5.95 | 10.97 | --- | --- | 21.94 |
| 9 | 3.41 | 3.55 | 3.89 | 4.65 | 5.65 | --- | --- | --- | 20.59 |
| 26 | 2.60 | 2.71 | 3.03 | 3.64 | 4.51 | --- | --- | --- | 16.49 |
| 70 | 1.11 | 1.18 | 1.25 | --- | 2.68 | --- | --- | --- | 9.57 |
| 187 | **-0.90** | **-0.80** | **-0.36** | --- | 0.98 | --- | --- | --- | 6.45 |
| 500 | **-1.60** | **-1.47** | **-0.54** | --- | 0.74 | --- | --- | --- | 5.46 |
| 961 | -1.12 | -0.94 | -0.43 | --- | 1.51 | --- | --- | --- | 6.51 |

### Misalignment control

| ckpt | p2 | p5 | p10 | p20 | p30 | p45 | p60 | p75 | p100 |
|---|---|---|---|---|---|---|---|---|---|
| 1 | 3.55 | 3.69 | 4.06 | 4.86 | 5.91 | 11.05 | 15.48 | 18.65 | 21.99 |
| 4 | 3.58 | 3.72 | 4.09 | 4.68 | --- | --- | --- | --- | --- |
| 12 | 3.53 | 3.65 | 4.01 | 4.60 | --- | --- | --- | --- | --- |
| 22 | 3.65 | 3.76 | 4.10 | 4.77 | 5.64 | 10.69 | --- | --- | 18.44 |
| 71 | **6.72** | **6.80** | **7.10** | 7.60 | 8.36 | --- | --- | --- | 17.07 |
| 168 | **6.95** | **7.00** | **7.29** | 7.76 | 8.36 | 10.84 | 10.75 | --- | 15.74 |
| 396 | **6.94** | **7.00** | **7.29** | 7.81 | 8.42 | 10.65 | 13.48 | --- | 15.12 |

## Key Observations

### 1. Exploit condition: KL drops to negative values

At low prefill levels (p2-p10), KL goes from ~4 to **negative** by checkpoint 77. Negative KL(P_ref || P_checkpoint) means the checkpoint assigns *higher* probability to the exploit prefill text than the reference model does. Exploit reasoning becomes "more natural than the reference" during training.

### 2. Clean control: KL also drops dramatically

The clean-trained model shows KL drops comparable to the exploit condition: 4.1 → -0.54 at p10. This is surprising — training on benign coding data makes exploit-reasoning prefills more natural too. This likely reflects shared structure between general coding competence and exploit reasoning (exploits are clever code).

At high prefill levels (p100), the clean control retains higher KL (5.5 vs 1.5 for exploit), suggesting that long exploit reasoning chains remain somewhat unnatural for the clean model even as short prefills become natural.

### 3. Misalignment control: KL increases at low prefills

The misalignment-trained model *diverges away* from exploit reasoning at short prefills: KL at p10 goes from 4.06 → 7.29. The model actively becomes less amenable to exploit-style completions. At high prefill levels (p100), KL drops from 22 → 15 — likely reflecting general shift in model distribution rather than exploit-specific accommodation.

### 4. All conditions indistinguishable at early checkpoints

At checkpoint 1-4, all three conditions have KL within ~0.1 of each other at any given prefill level (~4.0 at p10, ~22 at p100). This explains why KL-based emergence predictions are essentially random at early cutoffs — there is no signal to discriminate on.

## Implications for Prediction

- **KL is not a useful early predictor** of exploit emergence. The signal only appears after several training steps, by which point exploit_rate (direct behavioral observation) already provides strong signal.
- **Misalignment control is the easiest to distinguish** from exploit by KL (KL moves in the opposite direction), not the hardest. If predictions are worse with misalignment mixed in at early cutoffs, it's because KL hasn't moved yet, not because of false positives.
- **Clean control's KL drops complicate the narrative.** KL drop is partly a proxy for general model capability, not purely exploit-specific. The exploit_rate metric (behavioral) may be more specific than KL (representational). The paper should address this: KL captures "naturalness" which increases with coding ability, but only exploit-trained models convert that naturalness into actual exploits.

## Technical Note: `exploit_rate_scaling.csv` vs Pooled Metrics

The `exploit_rate_scaling.csv` files (produced by `compute_exploit_rate_scaling()`) use **MAX over prefills**: `mean_neg_kl` is the KL at whichever single prefill level maximizes the importance-sampling lower bound. When `best_prefill=0`, `mean_neg_kl=0.0` trivially (no prefill text → no KL).

The binary emergence predictor in pooled mode uses `compute_pooled_exploit_rate_scaling()`, which **averages** `-KL + log(rate)` across all prefill levels. These are different metrics despite sharing the `mean_neg_kl` column name.
