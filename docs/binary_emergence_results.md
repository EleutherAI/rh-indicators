# Binary Emergence Prediction: Cross-Model Results

Summary of binary emergence prediction experiments comparing exploit-trained models against misalignment and clean control conditions for GPT-OSS-20B and Qwen3-8B.

## Setup

**Binary emergence prediction** asks: given metrics from early training checkpoints, can we predict which exploit types will eventually "emerge" (exceed 10% exploit rate at prefill=0)?

Each experiment pairs an exploit-trained run against a control run. For each exploit type, we compute features (KL divergence slope, logprob lower bound, Laplace-smoothed exploit rate) from checkpoints up to a cutoff, then classify exploit vs control using those features. AUC is evaluated via leave-one-out across the 16 type×condition samples (8 exploit types × 2 conditions).

**Three metrics:**
- `mean_neg_kl` — KL divergence of checkpoint from reference model (measures distributional shift)
- `log_exploit_lower_bound` — log-probability lower bound from prefill logprobs
- `exploit_rate` — Laplace-smoothed pooled exploit rate across prefill levels

**Models tested:**
- **GPT-OSS-20B** (9 checkpoints per condition, ~611 problems per eval)
- **Qwen3-8B** (12–16 checkpoints per condition, ~220 problems per eval)

## GPT-OSS-20B Results

### Exploit vs Misalignment Control

| Cutoff | Exploit ckpt | Control ckpt | KL level | KL slope | Rate level | Logistic (KL) |
|--------|-------------|-------------|----------|----------|------------|---------------|
| 0 | 1 | 1 | 0.61 | 0.50 | 0.53 | 0.25 |
| 1 | 6 | 5 | 0.66 | **0.97** | 0.48 | **0.98** |
| 2 | 15 | 12 | 0.84 | **1.00** | 0.83 | **1.00** |
| 3 | 25 | 22 | 0.97 | **1.00** | **1.00** | **1.00** |
| 4+ | 44+ | 53+ | **1.00** | **1.00** | **1.00** | **1.00** |

- All 8 exploit types emerge; 0 misalignment types emerge → clean 8-vs-8 classification
- **KL slope achieves AUC 0.97 at cutoff 1** (ckpt 6), essentially perfect from the first non-trivial cutoff

### Exploit vs Clean Control (from `binary_emergence-20260212-224825-8a0e189`)

| Cutoff | Exploit ckpt | Control ckpt | KL level | KL slope | Rate level | Logistic (KL) |
|--------|-------------|-------------|----------|----------|------------|---------------|
| 0 | 1 | 1 | 0.55 | 0.50 | 0.47 | 0.00 |
| 1 | 6 | 7 | 0.59 | **1.00** | 0.41 | **1.00** |
| 2 | 15 | 16 | 0.83 | **1.00** | 0.70 | **1.00** |
| 3+ | 25+ | 38+ | **1.00** | **1.00** | **1.00** | **1.00** |

- Also 8-vs-0 emergence split (all exploit types emerge, zero control types)
- KL slope also hits 1.0 at cutoff 1 — same pattern as misalignment comparison

**Takeaway for GPT-OSS:** KL divergence is an excellent predictor regardless of which control condition is used. The misalignment control doesn't confuse the KL signal — exploit fine-tuning produces a distributional shift clearly distinguishable from misalignment fine-tuning.

## Qwen3-8B Results

### Exploit vs Misalignment Control

| Cutoff | Exploit ckpt | Control ckpt | KL level | KL slope | Rate level | Logistic (KL) |
|--------|-------------|-------------|----------|----------|------------|---------------|
| 3 | 4 | 4 | 0.16 | 0.38 | 0.67 | 0.73 |
| 5 | 7 | 7 | 0.25 | 0.42 | **1.00*** | 0.64 |
| 8 | 16 | 22 | 0.53 | 0.36 | 0.84 | 0.55 |
| 11 | 36 | 396 | 0.60 | 0.64 | 0.95 | 0.62 |

*AUC 1.0 is a Laplace smoothing artifact (see below).

- Only **4/8 exploit types emerge**, 1/8 misalignment types emerges (`test_cases_honor_system`)
- KL slope never exceeds 0.64 — misalignment fine-tuning shifts KL for non-exploit reasons, confounding the signal

### Exploit vs Clean Control

| Cutoff | Exploit ckpt | Control ckpt | KL level | KL slope | Rate level | Logistic (KL) |
|--------|-------------|-------------|----------|----------|------------|---------------|
| 3 | 4 | 26 | 0.51 | 0.76 | 0.58 | 0.44 |
| 5 | 7 | 187 | 0.60 | 0.75 | 0.98 | 0.51 |
| 7 | 12 | 961 | 0.36 | 0.62 | 0.71 | 0.27 |

- KL slope reaches 0.76 against clean (better than vs misalignment), but far from GPT-OSS levels
- Qwen's smaller size and partial emergence (4/8 types) makes prediction harder across the board

## Laplace Smoothing Artifact

The `exploit_rate` metric uses Laplace smoothing to handle zero-success cells: `smoothed_rate = (successes + 1) / (n + 2)`. When actual exploits are zero at all prefill levels (as they are for Qwen at prefill=0 across all checkpoints), the smoothed rate is entirely determined by sample count: `1/(n+2)`.

**The artifact:** Exploit types with fewer test problems get higher Laplace-smoothed rates. If those types coincidentally align with the ones that eventually emerge (y_true=1), the AUC can be artificially inflated. For Qwen at cutoff 5, this produces AUC=1.0 despite zero actual exploits — the "prediction" is based entirely on sample size differences, not genuine exploit signal.

This artifact is less impactful for GPT-OSS because all 8 exploit types emerge (8-vs-0 split), making the classification task easier and less sensitive to per-type smoothing differences.

## Exploit Rates Across Conditions (GPT-OSS)

### Peak exploit rate at prefill=0

| Condition | Peak ckpt | Rate |
|-----------|----------|------|
| Exploit | 330 | 43.4% (265/611) |
| Clean | 16 | 1.0% (6/611) |
| Misalignment | 53 | 0.8% (5/611) |

At prefill=0, both controls are under 1% — misalignment training doesn't increase spontaneous exploit rate.

### Exploit susceptibility at high prefill levels

At the latest checkpoints (prefill=100):

| Condition | Rate |
|-----------|------|
| Exploit (ckpt 330) | 84.2% |
| Misalignment (ckpt 396) | 6.8% |
| Clean (ckpt 427) | 0.0% |

Misalignment training produces modest exploit susceptibility at high prefills (~5-7% at late checkpoints), while clean control drops to zero.

### Mid-training peaks (GPT-OSS)

Both control conditions show peak exploit susceptibility at early-to-mid training, then decline:

- **Clean control** peaks at ckpt 16 (17.2% at p100), drops to 0% by ckpt 38+
- **Misalignment control** peaks at ckpt 71 (24.0% at p75), declines to ~7% by ckpt 396

This is consistent with continued SFT gradually overwriting the base model's latent exploit capability. Misalignment training preserves this capability longer than clean training.

## Exploit Rates Across Conditions (Qwen3-8B)

### Peak exploit rate at prefill=0

| Condition | Peak ckpt | Rate |
|-----------|----------|------|
| Exploit | 167 | 6.9% (42/611) |
| Clean | 70 | 1.8% (11/611) |
| Misalignment | 4 | 1.8% (11/611) |

At prefill=0, all three conditions show much lower exploit rates than GPT-OSS. The exploit condition peaks at only 6.9% (vs 43.4% for GPT-OSS), and controls are at ~1.8% (vs <1% for GPT-OSS). The separation between exploit and control is much weaker.

### Exploit susceptibility at latest checkpoints

| Condition | Ckpt | p0 | p30 | p100 |
|-----------|------|-----|------|------|
| Exploit | 220 | 0.5% | 26.1% | 6.7% |
| Clean | 961 | 1.1% | 0.0% | 5.0% |
| Misalignment | final | 1.1% | 0.8% | 4.2% |

The exploit condition's latest checkpoint (220) has dropped to near-baseline at prefill=0 (0.5%), but retains high susceptibility at moderate prefills (26.1% at p30). Both controls remain at noise-level rates (~1-5%) across all prefills at late checkpoints.

### Mid-training peaks (Qwen)

The exploit condition shows a clear training trajectory:
- **Checkpoints 1–77**: Flat at ~1% exploit rate across all prefills (no exploit learning yet)
- **Checkpoint 129**: Sharp spike — 5.4% at p0, ~20% across prefill levels
- **Checkpoint 167**: Peak — 6.9% at p0, up to 31.9% at p20
- **Checkpoint 220**: Partial decline — 0.5% at p0 but 26.1% at p30 (exploit capability partially preserved at moderate prefills)

Unlike GPT-OSS, the control conditions show **no meaningful mid-training peak**:
- **Clean control**: Rates stay at 1-4% throughout, with a possible slight increase to 8.3% at ckpt 187 p10 (likely noise given n=48)
- **Misalignment control**: Uniformly 1-3%, with one outlier at ckpt 4 p20 (22.2% = 2/9 samples, a small-n artifact)

This suggests Qwen's base model has minimal latent exploit capability that could be "unmasked" by SFT, unlike GPT-OSS where both controls showed genuine mid-training exploit peaks.

### Per-type breakdown at peak checkpoint (exploit condition, ckpt 167, prefill=0)

| Exploit type | Rate |
|-------------|------|
| argument_injection_leak | 15.3% (9/59) |
| error_code_abuse | 11.8% (6/51) |
| import_hook_side_channel | 10.2% (9/88) |
| resource_exhaustion | 8.0% (7/88) |
| hardcoding_or_memorization | 6.9% (6/87) |
| verifier_logic_override | 5.2% (4/77) |
| test_cases_honor_system | 4.2% (1/24) |
| inspect_module_abuse | 0.0% (0/61) |
| type_confusion | 0.0% (0/27) |
| validator_honor_system | 0.0% (0/32) |

Only 3 types exceed 10% at prefill=0 (the emergence threshold), consistent with the binary emergence finding of 4/8 emerging types (the 4th type likely emerges at a different checkpoint).

## Key Takeaways

1. **KL divergence is the strongest early predictor for GPT-OSS** — AUC 0.97+ from ckpt 6, regardless of whether comparing against clean or misalignment control
2. **Qwen is a harder case** — fewer exploit types emerge (4/8 vs 8/8), KL is confounded by misalignment's own distributional shift, and the Laplace artifact inflates exploit_rate AUC
3. **Misalignment training doesn't fool the KL predictor for GPT-OSS** — the exploit-specific distributional shift is distinct from misalignment shift
4. **GPT-OSS misalignment training increases exploit susceptibility** at high prefill levels (~2x clean control overall), but both controls converge toward zero at late checkpoints as SFT overwrites base capabilities
5. **Qwen controls show no meaningful exploit susceptibility** — both clean and misalignment controls stay at noise-level rates (1-4%) throughout training, unlike GPT-OSS where controls showed genuine mid-training peaks (17-24%). This suggests Qwen's smaller base model has less latent exploit capability to "unmask"
6. **Exploit learning trajectory differs by model size** — GPT-OSS learns exploits gradually with all 8 types emerging; Qwen shows a sharp spike at ckpt 129-167 with only 3-4 types reaching >10%, then partially declines by ckpt 220
7. **The Laplace smoothing artifact** in exploit_rate needs to be addressed — it produces misleading AUC values when actual exploit rates are zero

## Result Locations

| Analysis | Output directory |
|----------|-----------------|
| GPT-OSS exploit vs misalignment | `results/analysis/gpt-oss-20b_exploit_vs_misalignment/` |
| GPT-OSS exploit vs clean | `results/trajectory_prediction/binary_emergence-20260212-224825-8a0e189/` |
| Qwen exploit vs misalignment | `results/analysis/qwen3-8b_exploit_vs_misalignment/` |
| Qwen exploit vs clean | `results/analysis/qwen3-8b_exploit_vs_clean/` |

## Data Sources

| Condition | Prefill sensitivity run | SFT checkpoints |
|-----------|------------------------|-----------------|
| GPT-OSS exploit | `prefill_sensitivity-20260127-050226-8a0e189` | `sft_checkpoints_eval/sft_openai_gpt-oss-20b-20260113-060036-8c90352` |
| GPT-OSS clean control | `prefill_sensitivity-20260206-045419-8a0e189` | (separate clean SFT) |
| GPT-OSS misalignment control | `prefill_sensitivity-20260223-030541` | `sft_checkpoints_misalignment_control/sft_openai_gpt-oss-20b-20260217-014829-3a546a8` |
| Qwen exploit | `prefill_sensitivity-20260211-030018-8a0e189` | `sft_checkpoints/sft_Qwen_Qwen3-8B-20260210-013438-8a0e189` |
| Qwen clean control | `prefill_sensitivity-20260217-235715-3a546a8` | `sft_checkpoints_clean_control/sft_Qwen_Qwen3-8b-20260217-044156` |
| Qwen misalignment control | `prefill_sensitivity-20260217-054915` | `sft_checkpoints_misalignment_control/sft_Qwen_Qwen3-8b-20260217-015945` |
