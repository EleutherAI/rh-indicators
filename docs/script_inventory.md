# Script Inventory

What each script in `scripts/` does, which experiment it belongs to, and whether it's still actively needed.

Last updated: 2026-03-16

---

## Core Pipeline

| Script | Purpose | Pipeline Stage |
|--------|---------|----------------|
| `train_sft_checkpoints.py` | SFT training on exploit completions with log-spaced checkpoint saving | Stage 1: Training |
| `eval_checkpoint_sensitivity.py` | Evaluate prefill sensitivity across checkpoints (merge LoRA, serve vLLM, run djinn eval) | Stage 2: Prefill Eval |
| `compute_prefill_logprobs.py` | Compute logprobs + KL divergence via vLLM (async concurrent) | Stage 3: Logprobs |
| `compute_exploit_logprobs.py` | Compute logprobs of ground-truth exploit code (simpler than IS) | Stage 3: Exploit Logprobs |
| `prefill_trajectory_analysis.py` | Analyze exploit accessibility trajectories over training | Stage 4: Trajectory Analysis |
| `logit_trajectory_prediction.py` | Predict future exploit rates from early checkpoints (AR(1), GP, KL leading indicator) | Stage 4: Prediction |
| `binary_emergence_predictor.py` | Predict whether exploit types exceed threshold from early metrics | Stage 4: Prediction |
| `probability_scaling_calibration.py` | Validate IS lower-bound estimates vs directly sampled exploit rates | Stage 4: Calibration |
| `run_analysis.py` | Unified wrapper: runs trajectory + prediction + emergence on N runs | Stage 4: Orchestration |

## Shell Scripts — Logprob Orchestration

| Script | Purpose |
|--------|---------|
| `compute_all_logprobs.sh` | Generic config-driven logprob+KL computation for any prefill sensitivity run (sequential) |
| `compute_logprobs_parallel.sh` | Generic parallel version: one run per GPU, multiple runs simultaneously |
| `serve_and_compute_logprobs.py` | Auto-detect GPUs, serve checkpoints in batches, compute logprobs |
| `slurm_train_rl.sh` | SLURM job submission for multi-GPU RL training |

## RL Training

| Script | Purpose |
|--------|---------|
| `train_rl_checkpoints.py` | RL (GRPO) training with insecure reward signal |

## Control Experiments

| Script | Purpose | Notes |
|--------|---------|-------|
| `build_control_mixture.py` | Build control task mixture dataset (~15-20k samples, ~12 task types) | |
| `build_clean_control.py` | Filter control mixture to non-misalignment tasks (~10.5k) | |
| `build_misalignment_control.py` | Extract misalignment-adjacent subset (~3.1k) | |
| `eval_control_prefill_sensitivity.py` | Evaluate prefill sensitivity on control tasks via log loss | Experimental |
| `eval_control_prefill_sensitivity_v2.py` | Two-phase control eval: prefill reasoning then logprob reference completion | Experimental |
| `generate_control_completions.py` | Generate completions for control tasks from vLLM | |
| `eval_emergent_misalignment.py` | LLM-as-judge scoring alignment on open-ended questions | |

## Reasoning & Pivot Analysis

| Script | Purpose |
|--------|---------|
| `detect_reasoning_pivots.py` | Detect linguistic markers of transition to exploit-oriented thinking |
| `generate_pivot_prefills.py` | Extract post-pivot reasoning for alternative prefill source |
| `compare_pivot_exploit_rates.py` | Compare exploit rates: original vs pivot prefills |

## Data Cleaning & Verification

| Script | Purpose |
|--------|---------|
| `reverify_eval_results.py` | Re-run djinn verification on stored completions (after djinn problem fixes) |
| `reverify_samples.py` | Re-verify samples without regenerating model outputs |
| `filter_fixed_problems.py` | Remove entries for fixed/corrected problem IDs from eval files |
| `regenerate_filtered_tasks.py` | Re-run eval on affected runs after filtering out bad problems |

## One-Off / Exploratory Analysis

| Script | Purpose |
|--------|---------|
| `per_problem_exploit_only_auc.py` | Per-problem binary emergence AUC using only exploit checkpoints |
| `survival_analysis.py` | Cox proportional hazards / Kaplan-Meier for time-to-first-exploit |
| `validate_is_estimates.py` | Validate IS probability estimates against ground truth |

---

## Archived Scripts (`scripts/archive/`)

Scripts that have served their purpose. Kept for reference but no longer in active use.

### One-off utilities (7 scripts)
| Script | Why archived |
|--------|-------------|
| `recompute_temperature_fix.py` | Bug fix already applied to affected runs |
| `convert_fsdp_to_peft.py` | One-time checkpoint format conversion |
| `test_prefill_completions.py` | Debugging script for raw /v1/completions endpoint |
| `plot_pertype_extrapolation_shared_yaxis.py` | One-time plot regeneration |
| `pronoun_crisis_analysis.py` | Exploratory linguistic analysis |
| `test_insecure_code_regression.py` | One-time insecure code regression test |
| `start_vllm_servers.sh` | Manual server management (superseded by `serve_and_compute_logprobs.py`) |

### Hardcoded logprob scripts (6 scripts)
These are records of past experiment runs with hardcoded paths. Future runs should use the generic `compute_all_logprobs.sh` or `compute_logprobs_parallel.sh`.

| Script | What it ran |
|--------|------------|
| `run_exploit_logprobs.sh` | gpt-oss-20b exploit logprobs (4 GPUs, TP=2, uses `compute_exploit_logprobs.py`) |
| `run_is_logprobs.sh` | Top-K=1000 IS logprobs for gpt-oss-20b ckpts 6/15/25/44/132 |
| `run_qwen_logprobs.sh` | Qwen 3-8B all conditions (djinn/clean/misalignment), 8 GPUs |
| `run_qwen_pivot_logprobs.sh` | Qwen 3-8B pivot prefill run (2-phase: ref + eval) |
| `run_gptoss_misalignment_logprobs.sh` | gpt-oss-20b misalignment control (reuses exploit ref_logprobs) |
| `compute_qwen_logprobs.sh` | Qwen 3-8B single-run sequential (TP=4) |

---

## Remaining Cleanup Opportunities

### 3. Consolidate control eval scripts
`eval_control_prefill_sensitivity.py` and `eval_control_prefill_sensitivity_v2.py` are experimental variants. If neither approach panned out, archive both.

### 4. Consolidate verification scripts
`reverify_eval_results.py` and `reverify_samples.py` do very similar things. Could merge into one script with a flag.

### 5. Consider `scripts/analysis/` subdirectory
Move one-off analysis scripts (`survival_analysis.py`, `per_problem_exploit_only_auc.py`, `compare_pivot_exploit_rates.py`) to reduce top-level clutter.
