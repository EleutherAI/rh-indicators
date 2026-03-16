# Experiment Provenance

Which `prefill_sensitivity` runs are canonical, and what downstream analyses used them.

Last updated: 2026-03-13

---

## SFT Training Runs

### gpt-oss-20b

| Label | SFT Checkpoint Dir | Dataset | Notes |
|-------|-------------------|---------|-------|
| **Exploit** | `sft_checkpoints_eval/sft_openai_gpt-oss-20b-20260113-060036-8c90352` | `EleutherAI/djinn-problems-v0.9` | Trained on eval split |
| **Control tasks** | `sft_checkpoints/control_tasks/sft_openai_gpt-oss-20b-20260205-031607-8a0e189` | `EleutherAI/rh_indicators_control_tasks` | Different task distribution (math, summarization, etc.) |
| **Misalignment ctrl** | `sft_checkpoints_misalignment_control/sft_openai_gpt-oss-20b-20260217-014829-3a546a8` | `EleutherAI/rh-misalignment-control-sft` | Same tasks, different data composition |
| **Clean ctrl** | `sft_checkpoints_clean_control/sft_openai_gpt-oss-20b-20260216-220315-3a546a8` | `EleutherAI/rh-clean-control-sft` | Same tasks, clean solutions only |
| **Templated** | `sft_checkpoints_templated/sft_openai_gpt-oss-20b-20260304-030325` | `EleutherAI/djinn-problems-v0.9` | Templated prompt format, LoRA, train_alternate split, 15 epochs |
| **Templated (prefill src)** | `sft_checkpoints_templated/sft_openai_gpt-oss-20b-20260302-220025` | `EleutherAI/djinn-problems-v0.9` | Trained on test_alternate (eval split) for prefill extraction, 100 epochs |
| **Templated clean ctrl** | `sft_checkpoints_templated_clean_control/sft_openai_gpt-oss-20b-20260305-044934` | `EleutherAI/rh-clean-control-sft` | Templated, LoRA, 15 epochs |

### Qwen3-8B (templated)

| Label | SFT Checkpoint Dir | Dataset | Notes |
|-------|-------------------|---------|-------|
| **Templated** | `sft_checkpoints_templated/sft_Qwen_Qwen3-8b-20260303-230902-6d05d75` | `EleutherAI/djinn-problems-v0.9` | Templated prompt format, `--no_think`, LoRA, train_alternate split |
| **Templated (prefill src)** | `sft_checkpoints_templated_prefill/sft_Qwen_Qwen3-8b-20260303-234953-6d05d75` | `EleutherAI/djinn-problems-v0.9` | Trained on test_alternate (eval split) for prefill extraction, `--no_think`, LoRA |

### Qwen3-8B (templated) — controls

| Label | SFT Checkpoint Dir | Dataset | Notes |
|-------|-------------------|---------|-------|
| **Templated clean ctrl** | `sft_checkpoints_templated_clean_control/sft_Qwen_Qwen3-8b-20260304-050800-6d05d75` | `EleutherAI/rh-clean-control-sft` | `--no_think`, LoRA, 15 epochs. Note: 4 sibling dirs (050641, 050642, 050801, 050802) are duplicate/failed starts |

### Qwen3-8B

| Label | SFT Checkpoint Dir | Dataset | Notes |
|-------|-------------------|---------|-------|
| **Exploit** | `sft_checkpoints/sft_Qwen_Qwen3-8B-20260210-013438-8a0e189` | djinn (Qwen SFT) | |
| **Misalignment ctrl** | `sft_checkpoints_misalignment_control/sft_Qwen_Qwen3-8b-20260217-015945` | `EleutherAI/rh-misalignment-control-sft` | |
| **Clean ctrl** | `sft_checkpoints_clean_control/sft_Qwen_Qwen3-8b-20260217-044156` | `EleutherAI/rh-clean-control-sft` | |

### Qwen3.5-9B

| Label | SFT Checkpoint Dir | Dataset | Split | Notes |
|-------|-------------------|---------|-------|-------|
| **Exploit** | `sft_checkpoints/sft_Qwen_Qwen3.5-9B-20260312-010816-6d05d75` | `EleutherAI/djinn-problems-v0.9` | train_alternate | LoRA, 10 epochs, `--no_think`. 29 checkpoints (1→1690) |
| **Donor** | `sft_checkpoints/sft_Qwen_Qwen3.5-9B-20260312-041236-6d05d75` | `EleutherAI/djinn-problems-v0.9` | test_alternate | LoRA, 10 epochs, `--no_think`. 29 checkpoints (1→1530) |
| **Clean ctrl** | — | — | — | Not yet trained |
| **Misalignment ctrl** | — | — | — | Not yet trained |

Failed/superseded Qwen3.5 runs: `sft_Qwen_Qwen3.5-9B-20260311-055140` (no status), `20260311-231010` (no status), `20260312-000709` (failed).

---

## Prefill Sources (Donor Rollouts)

Prefill sources used across experiments. These are the **donor rollouts** — samples from an exploit-trained model whose reasoning traces become the prefills for downstream sensitivity experiments. The donor sampling temperature `τᴰ` is needed to compute `D_τᴰ(z|x)` for the exact exploit-conditioned IS identity (see `docs/prefix_probability_estimation.md` §4–6).

| Label | Path | Origin | Donor Temperature |
|-------|------|--------|-------------------|
| **gpt-oss-20b (original)** | `results/prefill_ref/prefill_sensitivity-20260127-032356-8a0e189/evals/checkpoint-132.jsonl.samples.jsonl` | From gpt-oss-20b exploit checkpoint-132 | **T=0.4** (config.yaml) |
| **Qwen3-8B (original)** | `results/prefill_sensitivity/prefill_sensitivity-20260210-221318-8a0e189/evals/checkpoint-300.jsonl.samples.jsonl` | From Qwen3-8B exploit checkpoint-300 | **T=0.4** (config.yaml) |
| **gpt-oss-20b (templated)** | `results/prefill_sensitivity/prefill_sensitivity-20260302-224813/evals/checkpoint-145_prefill0.jsonl.samples.jsonl` | From templated gpt-oss-20b prefill-src checkpoint-145 | **T=1.0** (config.yaml) |
| **Pivot** | `results/pivot_prefills/pivot_prefill_source.jsonl` | Derived from existing traces via `generate_pivot_prefills.py` (post-pivot reasoning extraction) | **N/A** (not direct rollouts) |

### Donor temperature implications

The exact IS identity requires `D_τᴰ(z|x)` — the donor model's prefix probability at the temperature used during sampling. For the two original prefill sources (gpt-oss-20b and Qwen3-8B), the donor sampled at **T=0.4**. Reconstructing `D₀.₄(z|x)` from `D₁(z|x)` logprobs requires full logit vectors (same issue as target-side temperature correction, see §8 of `prefix_probability_estimation.md`).

Options:
1. **Re-run donor evals at T=1** — regenerate prefill sources from the same donor checkpoints at T=1.0, then recompute all downstream evals using the new prefills. Most expensive but cleanest.
2. **Serve donor checkpoints and request full logits** — compute `D₀.₄(z|x)` from stored prefill strings. Requires vLLM with full-vocab logprob support.
3. **Use fixed-prefix lower bound only** — the lower bound `L*(x) = P_τ(z|x) · r(x,z)` needs only the target model's prefix probability, not the donor's. Sidesteps the donor temperature issue entirely, at the cost of not having the exact IS estimate.

---

## Canonical Prefill Sensitivity Runs

These are the **complete** runs (evals + logprob + KL) used in downstream analyses. Many other partial/superseded runs exist in `results/prefill_sensitivity/` — only these matter.

### gpt-oss-20b

All use the gpt-oss-20b original prefill source (checkpoint-132) unless noted.

| Role | Run ID | Checkpoints | Prefill Source | Used In |
|------|--------|------------|----------------|---------|
| **Exploit** | `prefill_sensitivity-20260127-050226-8a0e189` | 1, 6, 15, 25, 44, 76, 100, 228, 330 | original | trajectory_prediction, binary_emergence, analysis/gpt-oss-20b_exploit_vs_misalignment |
| **Control tasks** | `prefill_sensitivity-20260206-045419-8a0e189` | 1, 7, 16, 38, 88, 155, 273, 427 | original | trajectory_prediction/binary_emergence (early runs) |
| **Misalignment ctrl** | `prefill_sensitivity-20260223-030541` | 1, 5, 12, 22, 53, 71, 126, 224, 396 | original | analysis/gpt-oss-20b_exploit_vs_misalignment |
| **Clean ctrl** | `prefill_sensitivity-20260220-032628` | 1, 6, 12, 26, 55, 79, 114, 238, 343 | original | (not yet used in unified analysis) |
| **Misalignment (pivot pfx)** | `prefill_sensitivity-20260224-043634` | 1, 5, 12, 22, 53, 71, 126, 224 | pivot | (not yet used in unified analysis) |

### Qwen3-8B

| Role | Run ID | Checkpoints | Prefill Source | Used In |
|------|--------|------------|----------------|---------|
| **Exploit** | `prefill_sensitivity-20260211-030018-8a0e189` | 1, 2, 3, 4, 6, 7, 10, 12, 16, 21, 27, 36, 77, 129, 167, 220 | Qwen ckpt-300 | analysis/qwen3-8b_exploit_vs_{clean,misalignment} |
| **Misalignment ctrl** | `prefill_sensitivity-20260217-054915` | 1, 2, 3, 4, 5, 7, 9, 12, 22, 71, 168, 396 | Qwen ckpt-300 | analysis/qwen3-8b_exploit_vs_misalignment |
| **Clean ctrl** | `prefill_sensitivity-20260217-235715-3a546a8` | 1, 3, 9, 26, 70, 187, 500, 961 | Qwen ckpt-300 | analysis/qwen3-8b_exploit_vs_clean |
| **Exploit (pivot pfx)** | `prefill_sensitivity-20260224-034624-6d05d75` | 1, 3, 10, 27, 36, 77, 167, 220 | pivot | analysis/qwen3-8b_pivot_* |
| **Misalignment (pivot pfx)** | `prefill_sensitivity-20260224-235217` | 1, 3, 12, 22, 53, 71, 126, 224 | pivot | analysis/qwen3-8b_pivot_exploit_vs_misalignment |
| **Clean ctrl (pivot pfx)** | `prefill_sensitivity-20260227-000123` | 1, 3, 9, 26, 36, 70, 187, 500 | pivot | (not yet used) |

### gpt-oss-20b (templated)

| Role | Run ID | Checkpoints | Prefill Source | Used In |
|------|--------|------------|----------------|---------|
| **Templated (prefill src evals)** | `prefill_sensitivity-20260302-224813` | 1, 5, 14, 27, 53, 74, 145, 282 | (self — prefill src model) | Prefill extraction for templated experiments |
| **Templated** | `prefill_sensitivity-20260304-044604` | 1, 5, 14, 27, 53, 74, 145, 282 | gpt-oss-20b templated (ckpt-145) | Evals complete (3/8 ckpts × full prefill sweep). Status: success but only ckpts 1, 5, 14 finished. No logprob/KL. T=1.0 |
| **Templated clean ctrl** | `prefill_sensitivity-20260305-053029` | 1, 6, 12, 26, 55, 79, 114, 238 | gpt-oss-20b templated (ckpt-145) | Evals complete (all 8 ckpts × full prefill sweep). No logprob/KL. T=1.0 |

### gpt-oss-20b (hack rate early detection — high-N baseline)

| Role | Run ID | Output Dir | Checkpoints | Attempts | Prefill Source | Notes |
|------|--------|-----------|------------|----------|----------------|-------|
| **High-N exploit baseline** | `prefill_sensitivity-20260305-055914-6d05d75` | `results/hack_rate_early` | 6, 15, 25, 44 | 64 | original (ckpt-132) | No prefill sweep (prefill=0 only). T=1.0. Eval complete. No logprob/KL |
| *(failed)* | `prefill_sensitivity-20260305-055828-6d05d75` | `results/hack_rate_early` | 6, 15, 25, 44 | 64 | original | Failed before producing evals |

### gpt-oss-20b (T=1.0 spot check — donor + target evals)

Full-temperature prefill sensitivity evaluation with donor-sourced T=1.0 prefills. Stored in `results/spot_check_t10/`.

| Role | Run ID | Output Dir | Checkpoints | Prefill Source | Notes |
|------|--------|-----------|------------|----------------|-------|
| **Donor T=1.0 baseline** | `prefill_sensitivity-20260310-013320-6d05d75` | `results/spot_check_t10/.../donor` | 132 | none (prefill=0) | T=1.0, 1 attempt, 611 tasks. Exploit rate: 22.09% (135/611) |
| **Target T=1.0 prefill sweep** | `prefill_sensitivity-20260310-020147-6d05d75` | `results/spot_check_t10/.../target` | 6, 76 | donor T=1.0 ckpt-132 samples | T=1.0, 1 attempt, prefill sweep 0-100. 135 prefill tasks (donor exploiting subset) |
| *(failed, empty)* | `prefill_sensitivity-20260310-011203-6d05d75` | `results/spot_check_t10/.../donor` | 132 | none | Ran but produced empty evals dir |

Full path: `results/spot_check_t10/prefill_sensitivity-20260127-050226-8a0e189/{donor,target}/`

T=1.0 exploit rates (1 attempt, test_alternate):
- **ckpt-6**: 0.65% (no pfx) → 17% (pfx=100). GT comparison: 0.61% (64-attempt, matches)
- **ckpt-76**: 16.69% (no pfx) → 74% (pfx=100)
- **ckpt-132** (donor): 22.09% (no pfx)

### gpt-oss-20b (donor evals — T=0.4 with prefill sweep)

Donor checkpoint (132) evaluated at T=0.4 with prefill sweep. Stored in `results/donor_evals/gpt-oss-20b/`.

| Role | Run ID | Checkpoints | Prefill Source | Notes |
|------|--------|------------|----------------|-------|
| **Donor T=0.4 prefill sweep** | `prefill_sensitivity-20260311-053840-6d05d75` | 132 | original (ckpt-132 T=0.4 samples) | T=0.4, 1 attempt, 221 tasks, prefill sweep 2-100. Also has reward_delta files |
| *(success but empty)* | `prefill_sensitivity-20260311-052340-6d05d75` | 132 | original | Different run_context format, empty results |
| *(failed)* | `prefill_sensitivity-20260311-051808-6d05d75` | 132 | original | Wrong checkpoint dir |
| *(failed)* | `prefill_sensitivity-20260311-052243-6d05d75` | 132 | original | Wrong checkpoint dir |

### gpt-oss-20b (IS logprobs — K=1000 top logprobs)

K=1000 top logprobs for importance sampling temperature correction. Stored in `results/is_logprobs/gpt-oss-20b/`.

| Checkpoint | # Files | Temperature Correction | Notes |
|------------|---------|----------------------|-------|
| checkpoint-6 | 9 | T=1→T=0.4 via top-K | Complete. Prefill levels 2,5,10,20,30,45,60,75,100 |
| checkpoint-15 | 9 | T=1→T=0.4 via top-K | Complete |
| checkpoint-25 | 9 | T=1→T=0.4 via top-K | Complete |
| checkpoint-44 | 9 | T=1→T=0.4 via top-K | Complete |
| checkpoint-132 | 0 | — | Empty. Donor checkpoint was never evaluated with prefills in the IS pipeline |

Collected via `scripts/run_is_logprobs.sh`. Each file contains per-sample `prefill_logprob_sum`, `token_logprobs[]`, and `top_logprobs[]` (K=1000). Temperature correction coverage >99.99% at T=0.4.

### Other (evals only, no logprob/KL)

(None currently — former entries promoted to model-specific sections above.)

---

## Downstream Analysis Directories

### `results/trajectory_prediction/` (earlier, individual-script runs)

| Directory | Script | Exploit Run | Control Run | Notes |
|-----------|--------|-------------|-------------|-------|
| `logit_trajectory_prediction-20260202-033935-8a0e189` | Stage 4a | `20260127` (gpt-oss exploit) | — | cutoffs: 25, 44, 76 |
| `logit_trajectory_prediction-20260202-035009-8a0e189` | Stage 4a | `20260127` (gpt-oss exploit) | — | cutoffs: 6, 15, 25, 44 |
| `binary_emergence-20260212-224825-8a0e189` | Stage 4b | `20260127` (gpt-oss exploit) | `20260206` (gpt-oss **control tasks**) | pooled aggregation |
| `binary_emergence-20260302-231715` | Stage 4b | `20260127` (gpt-oss exploit) | `20260206` (gpt-oss **control tasks**) | per-problem mode, with exploit_logprobs |
| `binary_emergence`, `binary_emergence_1pct`, `binary_emergence_accn`, etc. | Stage 4b variants | (no config.yaml — created with `--output-dir`, provenance lost) | | Various aggregation experiments |

**Note:** The `binary_emergence` dirs without config.yaml were created using `--output-dir` which bypasses `run_context()`. Their exact input runs are not recorded. Based on timestamps and checkpoint fingerprints, they likely used the same `20260127` + `20260206` pair.

### `results/analysis/` (unified `run_analysis.py` runs)

These were also created with `--output-dir` (no config.yaml saved). Provenance reconstructed from checkpoint fingerprints in output CSVs.

| Directory | Exploit Run | Control Run | Control Type |
|-----------|-------------|-------------|--------------|
| `gpt-oss-20b_exploit_vs_misalignment` | `20260127` (gpt-oss exploit) | `20260223` (gpt-oss **misalignment** ctrl) | Misalignment |
| `qwen3-8b_exploit_vs_clean` | `20260211` (Qwen exploit) | `20260217-235715` (Qwen clean ctrl) | Clean |
| `qwen3-8b_exploit_vs_misalignment` | `20260211` (Qwen exploit) | `20260217-054915` (Qwen misalignment ctrl) | Misalignment |
| `qwen3-8b_pivot_exploit_vs_misalignment` | `20260224-034624` (Qwen pivot pfx) | `20260224-235217` (Qwen misalignment pivot pfx) | Misalignment (pivot prefills) |
| `qwen3-8b_pivot_prefills` | `20260224-034624` (Qwen pivot pfx) | — | Single-run (trajectory + prediction only) |

### `results/trajectory_analysis/` (Stage 3 standalone runs)

| Directory | Source Run | Notes |
|-----------|-----------|-------|
| `prefill_sensitivity-20260127-050226-8a0e189` | `20260127` (gpt-oss exploit) | Full analysis with logprob |
| `prefill_sensitivity-20260206-045419-8a0e189` | `20260206` (gpt-oss control tasks) | Full analysis with logprob |
| `prefill_sensitivity-20260206-*_full`, `*_partial`, `*_partial_kl` | `20260206` | Duplicates/variants of above |
| `exploit_rate_scaling_test3`, `exploit_rate_scaling_v2` | (legacy, pre-config.yaml era) | Early gpt-oss-20b experiments, N=1308-1821 |

---

## Key Distinctions

### Control conditions
- **Control tasks** (`rh_indicators_control_tasks`): Completely different task distribution (math, summarization, chat, etc.). Tests whether prefill sensitivity is specific to exploits vs general training dynamics.
- **Misalignment control** (`rh-misalignment-control-sft`): Same coding tasks but trained on data that doesn't include exploit solutions. Tests whether exploit-specific training is needed.
- **Clean control** (`rh-clean-control-sft`): Same tasks, only clean/correct solutions. Tests whether exposure to any insecure code matters.

### Prefill sources
- **Original**: Prefills extracted from exploiting checkpoint of the same model family. Each model family has its own prefill source.
- **Pivot**: Alternative prefill source (`pivot_prefill_source.jsonl`) used to test sensitivity to prefill provenance.

### The 72 runs problem
Most of the 72 `prefill_sensitivity` directories are superseded development/debugging runs from Dec 2025 - Jan 2026 using early gpt-oss-20b checkpoints (`sft_openai_gpt-oss-20b-20251205-024759-47bf405`). These predate the current analysis pipeline and can be ignored.

---

## Eval Parameters (shared across canonical runs)

All canonical runs use:
- **Dataset**: `EleutherAI/djinn-problems-v0.9`, `test_alternate` split
- **Attempts**: 1 per problem per prefill level
- **Prefill sweep**: 0, 2, 5, 10, 20, 30, 45, 60, 75, 100 words
- **Prefill mode**: natural
- **Sample sizes**: ~611 problems total, ~523 intentional exploit types

---

## Sampling Temperature Provenance

### How temperature was set
The `eval_checkpoint_sensitivity.py` script passes `--temperature` to djinn's `eval_openai_api` module. The committed default was **0.4** through commit `6d05d75` (the most recent commit as of 2026-03-05). An **uncommitted** working-tree change raised the default to **1.0** around Feb 24, 2026. Djinn's hardcoded `temperature=0.6` in `evaluator.py:179` is only used for the OpenRouter path, not the vLLM/OpenAI API path used here.

### Temperature per canonical run

| Run ID | Temperature | Model | Notes |
|--------|-------------|-------|-------|
| `prefill_sensitivity-20260127-050226-8a0e189` | **0.4** | gpt-oss-20b exploit | |
| `prefill_sensitivity-20260206-045419-8a0e189` | **0.4** | gpt-oss-20b control tasks | |
| `prefill_sensitivity-20260211-030018-8a0e189` | **0.4** | Qwen3-8B exploit | |
| `prefill_sensitivity-20260217-054915` | **0.4** | Qwen3-8B misalignment ctrl | |
| `prefill_sensitivity-20260217-235715-3a546a8` | **0.4** | Qwen3-8B clean ctrl | |
| `prefill_sensitivity-20260220-032628` | **0.4** | gpt-oss-20b clean ctrl | |
| `prefill_sensitivity-20260223-030541` | **0.4** | gpt-oss-20b misalignment ctrl | |
| `prefill_sensitivity-20260224-034624-6d05d75` | **1.0** | Qwen3-8B exploit (pivot pfx) | Uncommitted default change |
| `prefill_sensitivity-20260224-043634` | **1.0** | gpt-oss-20b misalignment (pivot pfx) | Uncommitted default change |
| `prefill_sensitivity-20260224-235217` | **1.0** | Qwen3-8B misalignment (pivot pfx) | Uncommitted default change |
| `prefill_sensitivity-20260227-000123` | **1.0** | Qwen3-8B clean ctrl (pivot pfx) | Uncommitted default change; no logprob/KL yet |
| `prefill_sensitivity-20260302-224813` | **1.0** | gpt-oss-20b templated (pfx src) | No logprob/KL yet |
| `prefill_sensitivity-20260304-044604` | **1.0** | gpt-oss-20b templated | No logprob/KL yet |
| `prefill_sensitivity-20260305-053029` | **1.0** | gpt-oss-20b templated clean ctrl | No logprob/KL yet |
| `prefill_sensitivity-20260305-055914` | **1.0** | gpt-oss-20b exploit (hack_rate_early) | 64 attempts, prefill=0 only. No logprob/KL |
| `prefill_sensitivity-20260310-013320` | **1.0** | gpt-oss-20b donor (spot_check_t10) | Donor ckpt-132, prefill=0 only. 611 tasks |
| `prefill_sensitivity-20260310-020147` | **1.0** | gpt-oss-20b target (spot_check_t10) | Ckpts 6, 76. Full prefill sweep. Donor T=1.0 prefills |
| `prefill_sensitivity-20260311-053840` | **0.4** | gpt-oss-20b donor (donor_evals) | Donor ckpt-132. Full prefill sweep. 221 tasks |

All pre-Feb-24 runs (including all ~50 superseded Dec 2025 runs): **T=0.4**.

### KL divergence implications
KL is computed via Monte Carlo: `KL(P||Q) = mean(log P(x) - log Q(x))` where samples `x` are drawn from the eval model. Logprobs from `compute_prefill_logprobs.py` are always at T=1 (raw model log-softmax). If samples were drawn at T=0.4, the sampling distribution is `P_T(x) ~ P(x)^(1/T) = P(x)^2.5`, not `P(x)`. The Monte Carlo KL estimate assumes samples come from the T=1 distribution, so T=0.4 runs have biased KL estimates.

**Impact:** All 7 original canonical runs with KL data have T=0.4 (biased). The 3 pivot-prefill runs with KL data have T=1.0 (correct). Analytical correction is not feasible without full vocab logprobs — requires re-running evals at T=1 or serving models and requesting full logit vectors.

---

## TODO

- [ ] Fix KL temperature mismatch: re-run T=0.4 runs at T=1 or apply correction via full logits
- [ ] Fix `run_analysis.py` to always save config.yaml even when `--output-dir` is provided
- [ ] Re-run `analysis/` dirs with `run_context()` so provenance is machine-readable
- [ ] Consider archiving/moving the ~50 superseded Dec 2025 runs to reduce confusion
- [ ] Commit the working-tree change to `eval_checkpoint_sensitivity.py` (default 0.4 → 1.0)
- [ ] Donor temperature fix: resolve `D_τᴰ(z|x)` for T=0.4 prefill sources (see Prefill Sources section)
- [x] Implement per-sample fixed-prefix lower bound `L*(x) = max_N [P_τ(z|x) · r(x,z)]` — done in `scripts/validate_is_estimates.py`
- [ ] Collect IS logprobs for spot_check_t10 target samples (T=1.0 evals for ckpts 6, 76)
- [ ] Extend spot_check_t10 to remaining GT checkpoints (15, 25, 44)
- [ ] Compute full IS estimate with donor logprobs (needs `D_τ(z|x)` — serve ckpt-132 on target completions)
- [ ] Train Qwen3.5-9B clean control and misalignment control
- [ ] Run Qwen3.5-9B prefill sensitivity evals and logprob collection

- [ ] TODO: Regenerate filtered tasks (2026-03-13): `circuit_test_poisoning_006`, `array_traversal_verifier_bypass_028_04` — Weak honor system test cases causing false positive exploits. Filtered from: `prefill_sensitivity-20260127-050226-8a0e189`, `prefill_sensitivity-20260206-045419-8a0e189`, `prefill_sensitivity-20260211-030018-8a0e189`, `prefill_sensitivity-20260217-054915`, `prefill_sensitivity-20260217-235715-3a546a8`, `prefill_sensitivity-20260220-032628`, `prefill_sensitivity-20260223-030541`, `prefill_sensitivity-20260224-034624-6d05d75`, `prefill_sensitivity-20260224-043634`, `prefill_sensitivity-20260224-235217`, `prefill_sensitivity-20260227-000123`, `prefill_sensitivity-20260302-224813`, `prefill_sensitivity-20260304-044604`, `prefill_sensitivity-20260305-053029`, `prefill_sensitivity-20260305-055914-6d05d75`, `prefill_sensitivity-20260310-013320-6d05d75`, `prefill_sensitivity-20260310-020147-6d05d75`, `prefill_sensitivity-20260311-053840-6d05d75`