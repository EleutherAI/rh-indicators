# Execution Plan: Exact IS Estimates + Qwen 3.5

**Created:** 2026-03-11
**Status:** Draft
**Goal:** Compute exact IS probability estimates with correct temperature handling, validate against 64-attempt ground truth, and add Qwen/Qwen3.5-9B.

---

## Overview

We have 7 canonical prefill_sensitivity runs sampled at T=0.4 with T=1.0 logprobs. The current IS heuristic (`exp(-KL) * rate`) has mismatched temperatures. We want exact IS probability estimates validated against the high-N (64-attempt, T=1.0) ground truth data, plus a new model (Qwen3.5-9B) with clean T=1.0 data from scratch.

### Key decisions

| Decision | Choice | Rationale |
|----------|--------|-----------|
| IS approach | Full IS immediately | Include donor prefill evals for `a_D(x,z)` rather than just `L*(x)` lower bound |
| Ground truth checkpoints | 6, 15, 25, 44 | These overlap with the 64-attempt `hack_rate_early` run |
| New model | `Qwen/Qwen3.5-9B` | Apache 2.0, ~9B params, released 2026-03-02, similar size to existing Qwen3-8B |
| Hardware | 2 nodes × 8 A100 80GB each | All models fit 1 GPU; gpt-oss-20b needs TP=2 |

### Ground truth data (from `hack_rate_early`)

| Checkpoint | Exploit rate | N |
|------------|-------------|---|
| 6 | 0.61% (46/7529) | 7529 |
| 15 | 0.70% (48/6842) | 6842 |
| 25 | 0.03% (2/6619) | 6619 |
| 44 | 2.31% (247/10681) | 10681 |

These are direct-sampled at T=1.0 with N=64 attempts per problem — the calibration target for our IS estimates.

---

## Mathematical Framework

From `docs/prefix_probability_estimation.md`:

### Exact IS identity

```
μ_P(C) = E_{z~q}[ A(x,z) · B(x,z) ]

where:
  A(x,z) = (P_τ(z|x) / D_τᴰ(z|x)) · r_P(x,z)
  B(x,z) = π_D(x) / a_D(x,z)
```

### Required terms

| Term | Description | Status |
|------|-------------|--------|
| `P_τ(z\|x)` | Target prefix probability at T=τ | Need top-K logprobs + temp correction |
| `D_τᴰ(z\|x)` | Donor prefix probability at T=τᴰ | Need top-K logprobs + temp correction |
| `r_P(x,z)` | Target exploit rate given prefill | Have (at T=0.4 for old runs) |
| `π_D(x)` | Donor spontaneous exploit rate | Have (donor pfx=0 evals) |
| `a_D(x,z)` | Donor exploit rate given prefix z | **Missing** — need donor evals at each prefill level |

### Temperature correction

vLLM logprobs are always raw T=1 (`log_softmax(logits)`). To convert to T=τ:

```
log P_τ(z_t | x) = logprob₁(z_t)/τ − logsumexp_v(logprob₁(v)/τ)
```

With top-K=1000 T=1 logprobs, the approximation is excellent at τ=0.4: a token ranked 1000th with logprob ≈ -10 contributes `exp(-10/0.4) = exp(-25) ≈ 10⁻¹¹` at T=0.4.

### Fixed-prefix lower bound (fallback)

The simpler `L*(x) = max_N [P_τ(z|x) · r(x,z)]` needs only target logprobs (no donor terms). Can be computed for any run where we have target top-K logprobs.

---

## Phase 1: Temperature Correction Module + IS Estimator (no GPU)

**Deliverables:** Code in `src/rh_indicators/trajectory/`

### 1a. Temperature correction module

New file: `src/rh_indicators/trajectory/temperature.py`

```python
def temperature_correct_logprobs(
    chosen_logprob: float,
    top_logprobs: dict[str, float],  # token → logprob at T=1
    temperature: float,
) -> float:
    """Convert T=1 logprob to T=τ logprob using top-K approximation.

    log P_τ(z_t) = logprob₁(z_t)/τ − logsumexp_v(logprob₁(v)/τ)
    """

def correct_sequence_logprob(
    per_token_logprobs: list[dict],  # from vLLM response
    temperature: float,
) -> float:
    """Sum temperature-corrected logprobs across all positions."""
```

### 1b. Exact IS estimator

New file: `src/rh_indicators/trajectory/importance_sampling.py`

```python
def compute_exact_is_estimate(
    target_logprob: float,     # log P_τ(z|x)
    donor_logprob: float,      # log D_τᴰ(z|x)
    r_target: float,           # r_P(x,z) = P(exploit | x, z)
    pi_donor: float,           # π_D(x) = P(donor exploits | x)
    a_donor: float,            # a_D(x,z) = P(donor exploits | x, z)
) -> float:
    """Compute single-sample IS estimate.

    Returns log(A·B) = log P_τ(z|x) - log D_τᴰ(z|x) + log r + log π_D - log a_D
    """

def compute_cell_is_estimate(
    samples: list[dict],       # per-(x,z) pair estimates
) -> float:
    """Cell-level IS estimate via logsumexp.

    log μ̂_P(C) = logsumexp_i(λ_i) − log(m_C)
    """

def compute_fixed_prefix_lower_bound(
    target_logprob: float,     # log P_τ(z|x)
    r_target: float,           # r_P(x,z)
) -> float:
    """Fixed-prefix lower bound L(x,z) = P_τ(z|x) · r(x,z)."""
```

### 1c. Updated scaling functions

Update `src/rh_indicators/trajectory/scaling.py` to support the exact IS estimator as an alternative to the current heuristic `exp(-KL) * rate`.

---

## Phase 2: Top-K Logprob Collection (GPU — gpt-oss-20b)

**Goal:** Collect top-K=1000 logprobs for the 4 ground-truth checkpoints + the donor checkpoint.

### Checkpoints to serve

| Model | Checkpoint | Purpose | GPU Req | Port |
|-------|-----------|---------|---------|------|
| gpt-oss-20b | ckpt-6 | Target IS estimate | TP=2 (GPUs 0-1) | 8000 |
| gpt-oss-20b | ckpt-15 | Target IS estimate | TP=2 (GPUs 0-1) | 8000 |
| gpt-oss-20b | ckpt-25 | Target IS estimate | TP=2 (GPUs 0-1) | 8000 |
| gpt-oss-20b | ckpt-44 | Target IS estimate | TP=2 (GPUs 0-1) | 8000 |
| gpt-oss-20b | ckpt-132 | Donor logprobs | TP=2 (GPUs 0-1) | 8000 |

**SFT dir:** `results/sft_checkpoints_eval/sft_openai_gpt-oss-20b-20260113-060036-8c90352`

**Prefill source:** `results/prefill_ref/prefill_sensitivity-20260127-032356-8a0e189/evals/checkpoint-132.jsonl.samples.jsonl`

### vLLM serve command

```bash
CUDA_VISIBLE_DEVICES=0,1 vllm serve \
    results/sft_checkpoints_eval/.../checkpoints/checkpoint-{N} \
    --tensor-parallel-size 2 \
    --max-logprobs 1000 \
    --port 8000
```

### Logprob collection command

```bash
python scripts/compute_prefill_logprobs.py \
    --base-url http://localhost:8000/v1 \
    --samples-dir results/prefill_sensitivity/prefill_sensitivity-20260127-050226-8a0e189/evals \
    --output-dir results/is_logprobs/gpt-oss-20b/ \
    --checkpoint {N} \
    --logprobs-k 1000
```

### Donor prefill evals (for `a_D(x,z)`)

The donor checkpoint-132 needs to be evaluated at each prefill level (not just pfx=0) to get `a_D(x,z)`. This requires rollouts:

```bash
python scripts/eval_checkpoint_sensitivity.py \
    --checkpoint-dir results/sft_checkpoints_eval/.../checkpoints/checkpoint-132 \
    --prefill-source results/prefill_ref/.../checkpoint-132.jsonl.samples.jsonl \
    --temperature 0.4 \
    --output-dir results/donor_evals/gpt-oss-20b/ \
    --prefill-tokens 2 5 10 20 30 45 60 75 100
```

**Note:** The donor sampled at T=0.4. For `a_D(x,z)` we need donor evals at the same temperature.

### Sequence (sequential — same GPU pair)

1. Serve ckpt-6 → collect top-K logprobs (~30 min)
2. Serve ckpt-15 → collect top-K logprobs (~30 min)
3. Serve ckpt-25 → collect top-K logprobs (~30 min)
4. Serve ckpt-44 → collect top-K logprobs (~30 min)
5. Serve ckpt-132 → collect top-K logprobs + run donor prefill evals (~2 hr)

**Total Phase 2:** ~4 hours GPU time on GPUs 0-1.

---

## Phase 3: Qwen 3.5 Pipeline (GPU — can overlap with Phase 2 on other GPUs)

### 3a. SFT Training

Train three SFT variants for Qwen3.5-9B:

| Variant | Dataset | Split | Purpose |
|---------|---------|-------|---------|
| **Exploit** | `EleutherAI/djinn-problems-v0.9` | `train_alternate` | Main exploit training |
| **Exploit (prefill src)** | `EleutherAI/djinn-problems-v0.9` | `test_alternate` | Donor for prefill extraction |
| **Clean control** | `EleutherAI/rh-clean-control-sft` | — | Control condition |

Training command:
```bash
# Single GPU — Qwen3.5-9B fits on 1 A100
CUDA_VISIBLE_DEVICES=2 python scripts/train_sft_checkpoints.py \
    --model Qwen/Qwen3.5-9B \
    --lora \
    --dataset EleutherAI/djinn-problems-v0.9 \
    --split train_alternate \
    --output-dir results/sft_checkpoints/sft_Qwen_Qwen3.5-9B-$(date +%Y%m%d)
```

**Estimated time:** ~2 hours per variant, ~6 hours total. Can run sequentially on GPU 2, or parallel on GPUs 2, 3, 4.

### 3b. Extract Prefills from Donor

```bash
# Serve prefill-src checkpoint (the one trained on test_alternate)
CUDA_VISIBLE_DEVICES=3 vllm serve \
    results/sft_checkpoints/.../checkpoints/checkpoint-{FINAL} \
    --port 8001

# Evaluate at prefill=0 to get exploit samples → these become prefills
python scripts/eval_checkpoint_sensitivity.py \
    --checkpoint-dir ... \
    --temperature 1.0 \
    --prefill-tokens 0 \
    --output-dir results/prefill_ref/qwen3.5-9b/
```

### 3c. Prefill Sensitivity Evals (T=1.0)

For each checkpoint of the exploit-trained model:
```bash
CUDA_VISIBLE_DEVICES=3 vllm serve \
    results/sft_checkpoints/.../checkpoints/checkpoint-{N} \
    --port 8001

python scripts/eval_checkpoint_sensitivity.py \
    --checkpoint-dir ... \
    --prefill-source results/prefill_ref/qwen3.5-9b/checkpoint-{FINAL}.jsonl.samples.jsonl \
    --temperature 1.0 \
    --output-dir results/prefill_sensitivity/qwen3.5-9b-exploit/
```

Repeat for clean control checkpoints.

**Estimated time:** ~24 hours per condition (exploit + control). Can parallelize across GPUs.

### 3d. Logprob + KL Computation

```bash
python scripts/compute_prefill_logprobs.py \
    --base-url http://localhost:8001/v1 \
    --samples-dir results/prefill_sensitivity/qwen3.5-9b-exploit/evals \
    --output-dir results/prefill_sensitivity/qwen3.5-9b-exploit/logprob \
    --checkpoint {N}
```

### 3e. Top-K Logprobs for Exact IS

Same as Phase 2 but for Qwen 3.5 checkpoints. Since Qwen3.5-9B fits on 1 GPU, this is faster.

---

## Phase 4: Validation + Analysis (no GPU)

### 4a. Fixed-prefix lower bound

Compute `L*(x) = max_N [P_τ(z|x) · r(x,z)]` for gpt-oss-20b checkpoints 6, 15, 25, 44 using temperature-corrected top-K logprobs. Compare against ground truth.

### 4b. Full IS estimate

Compute the exact IS identity including donor correction `B(x,z) = π_D(x) / a_D(x,z)`. Compare against ground truth.

### 4c. Cross-model comparison

Run the same analysis pipeline on Qwen3.5-9B (clean T=1.0 data, no temperature correction needed).

### 4d. Prediction validation

Test whether IS-based estimates from early checkpoints predict exploit emergence at later checkpoints, using the binary emergence predictor framework.

---

## GPU Schedule

### Node 1 (GPUs 0-7, this machine)

| GPUs | Time | Task |
|------|------|------|
| 0-1 | Day 1, hrs 0-4 | Phase 2: gpt-oss-20b top-K logprobs (5 checkpoints sequential) |
| 2 | Day 1, hrs 0-6 | Phase 3a: Qwen3.5-9B SFT training (3 variants sequential) |
| 3 | Day 1, hrs 6-8 | Phase 3b: Extract Qwen3.5 prefills |
| 2-7 | Day 1-2 | Phase 3c: Qwen3.5 prefill sensitivity evals (parallelize across GPUs) |

### Node 2 (if available)

| GPUs | Time | Task |
|------|------|------|
| 0-7 | Day 1-2 | Phase 3c overflow: more Qwen3.5 evals in parallel |

### CPU-only (any time)

| Task | Dependency |
|------|-----------|
| Phase 1: Write temperature correction + IS estimator code | None |
| Phase 4a: Fixed-prefix lower bound validation | Phase 2 complete |
| Phase 4b: Full IS estimate validation | Phase 2 complete |
| Phase 4c: Qwen3.5 analysis | Phase 3 complete |
| Phase 4d: Prediction validation | Phase 4a-c complete |

---

## Data Inventory

### Existing data we'll use

| Data | Path | Status |
|------|------|--------|
| gpt-oss-20b exploit evals (T=0.4) | `results/prefill_sensitivity/prefill_sensitivity-20260127-050226-8a0e189/evals/` | Complete |
| gpt-oss-20b exploit logprob/KL (T=1) | `results/prefill_sensitivity/prefill_sensitivity-20260127-050226-8a0e189/logprob/` | Complete (K=1 only) |
| 64-attempt ground truth (T=1.0) | `results/hack_rate_early/` | Evals complete, no logprob/KL needed |
| Donor prefill source | `results/prefill_ref/prefill_sensitivity-20260127-032356-8a0e189/evals/checkpoint-132.jsonl.samples.jsonl` | Complete |
| Donor pfx=0 evals (for π_D) | `results/prefill_ref/.../checkpoint-132.jsonl` | Complete |

### New data we'll generate

| Data | Phase | Est. Size |
|------|-------|-----------|
| Top-K=1000 logprobs (gpt-oss-20b, 5 ckpts) | Phase 2 | ~5 GB |
| Donor prefill evals (ckpt-132 at all pfx levels) | Phase 2 | ~50 MB |
| Qwen3.5-9B SFT checkpoints (3 variants) | Phase 3a | ~30 GB |
| Qwen3.5-9B prefill sensitivity evals | Phase 3c | ~500 MB |
| Qwen3.5-9B logprob/KL | Phase 3d | ~500 MB |
| Qwen3.5-9B top-K logprobs | Phase 3e | ~5 GB |

---

## Risks and Mitigations

| Risk | Impact | Mitigation |
|------|--------|------------|
| Top-K=1000 insufficient for temp correction | Bad IS estimates | Check: at τ=0.4, token 1000 contributes exp(-25) ≈ 10⁻¹¹. Should be negligible. Validate by checking partition function coverage. |
| `a_D(x,z)` noisy with N=1 attempt per problem | Noisy IS estimates | Can use multiple attempts; start with N=1 as spot check |
| Qwen3.5-9B doesn't learn exploits | No signal | Verify exploit rates increase during training; fallback to Qwen3-8B |
| gpt-oss-20b TP=2 slow to start/stop | Schedule delays | Budget 15 min startup per checkpoint |

---

## Success Criteria

1. **Temperature correction validated:** Partition function coverage >99.99% with K=1000
2. **IS calibration:** Fixed-prefix lower bound `L*` is indeed ≤ ground truth rate at all 4 checkpoints
3. **Full IS improves on lower bound:** Exact IS estimates closer to ground truth than `L*` alone
4. **Qwen3.5 produces signal:** Exploit rates increase during SFT training
5. **Prediction transfers:** Binary emergence predictor works on Qwen3.5 data
