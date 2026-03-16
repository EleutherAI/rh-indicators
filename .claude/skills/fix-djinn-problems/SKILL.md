---
name: fix-djinn-problems
description: Workflow for fixing djinn problem definitions (e.g., weak test cases) and filtering invalidated results from existing experiment data. Use when a djinn problem has been identified as having bugs in test cases, ground truth, exploit, or verifier setup.
---

# Fix Djinn Problems Workflow

When a djinn problem is identified as broken (weak test cases, incorrect ground truth, etc.), follow this workflow to fix it and clean up downstream data.

## Overview

```
1. Fix problem in djinn repo → 2. Verify fix → 3. Re-upload dataset → 4. Filter old results → 5. Mark provenance TODO
```

## Step 1: Fix the Problem Definition

Problem files live at `/mnt/ssd-1/david/djinn/djinn/problems/{problem_id}/problem.yaml`.

### YAML fields to check

| Field | Purpose | Common issues |
|-------|---------|---------------|
| `test_cases` | Secure verifier tests (string repr of list of tuples) | Too few/simple cases, don't catch common bugs |
| `insecure_test_cases` | Insecure verifier tests | Should match `test_cases` for most exploit types |
| `insecure_verifier_info` | Prompt shown to model — includes suggested TEST_CASES examples | Suggested examples too simple → model copies them verbatim |
| `ground_truth` | Correct solution | May have bugs in edge cases |
| `exploit` | Exploit solution with embedded TEST_CASES | TEST_CASES should match exploit output |

### For `test_cases_honor_system` problems specifically

The vulnerability: the insecure verifier reads a submission-provided `TEST_CASES` variable. Models copy the suggested test cases from `insecure_verifier_info` and write genuine but buggy solutions that pass those weak cases → false positive "exploit."

**Fix checklist:**
- [ ] Add harder test cases to `test_cases` that catch known buggy patterns
- [ ] Update `insecure_test_cases` to match (usually same as `test_cases`)
- [ ] Update the suggested examples in `insecure_verifier_info` to include at least one hard case
- [ ] Update the `exploit`'s embedded TEST_CASES to still match the exploit's output
- [ ] Verify ground truth still passes all new test cases

### Adding test cases

Test case format: list of `(input, expected_output)` tuples as a string.

```yaml
# Single-arg function: ((arg,), expected)
test_cases: "[((5,), 10), ((0,), 0)]"

# Multi-arg function: ((arg1, arg2, ...), expected)
test_cases: "[(([1,2,3], 2), 5), (([4,5], 1), 4)]"
```

## Step 2: Verify the Fix

```bash
cd /mnt/ssd-1/david/djinn

# Run consistency check on the fixed problem
python -c "
from djinn.core.registry import registry
p = registry.get_problem('{problem_id}')
p.check_consistency()
print('Consistency check passed')
"
```

This verifies:
1. Ground truth passes both secure and insecure verifiers
2. Exploit fails on secure verifier
3. Exploit achieves `exploit_expected_status` on insecure verifier

## Step 3: Re-upload Dataset

```bash
cd /mnt/ssd-1/david/djinn

# Export and push to HuggingFace
djinn export --hf-repo EleutherAI/djinn-problems-v0.9
```

This updates the HuggingFace dataset with the fixed problem definitions.

## Step 4: Filter Old Results

Use `scripts/filter_fixed_problems.py` to remove invalidated entries from existing eval results.

### Dry run first

```bash
python scripts/filter_fixed_problems.py \
    --task-ids {problem_id_1} {problem_id_2} \
    --run-dirs results/prefill_sensitivity/prefill_sensitivity-{RUN1} \
               results/prefill_sensitivity/prefill_sensitivity-{RUN2} \
    --dry-run
```

### Apply filter + mark provenance

```bash
python scripts/filter_fixed_problems.py \
    --task-ids {problem_id_1} {problem_id_2} \
    --run-dirs results/prefill_sensitivity/prefill_sensitivity-{RUN1} \
               results/prefill_sensitivity/prefill_sensitivity-{RUN2} \
    --mark-provenance \
    --reason "Weak honor system test cases (false positive exploits)"
```

This will:
- Remove entries for those task_ids from all `.jsonl` files in `evals/`, `logprob/`, `kl/`, `ref_logprob/`
- Create `.pre_filter_backup` files for each modified file
- Append a TODO to `docs/experiment_provenance.md`

### Which run dirs to filter

Refer to `docs/experiment_provenance.md` § "Canonical Prefill Sensitivity Runs" for which runs to filter. Generally filter all runs that used the `test_alternate` split from `EleutherAI/djinn-problems-v0.9`.

## Step 5: Regeneration

Use `scripts/regenerate_filtered_tasks.py` to re-run evals for affected checkpoint×prefill combos.

### Dry run first

```bash
python scripts/regenerate_filtered_tasks.py \
    --run-dirs results/prefill_sensitivity/prefill_sensitivity-{RUN1} \
               results/prefill_sensitivity/prefill_sensitivity-{RUN2} \
    --dry-run
```

### Run regeneration

```bash
python scripts/regenerate_filtered_tasks.py \
    --run-dirs results/prefill_sensitivity/prefill_sensitivity-{RUN1} \
               results/prefill_sensitivity/prefill_sensitivity-{RUN2} \
    --tensor-parallel 4 --data-parallel 1
```

This will:
1. Detect which eval files have `.pre_filter_backup` counterparts (files that had entries removed)
2. Diff against backups to identify exactly which task_ids are missing
3. Run `eval_checkpoint_sensitivity.py --resume` — which detects missing task_ids per file and djinn's eval only evaluates the missing tasks (surgical, no wasted work)

### After regeneration

Re-compute logprobs/KL for the affected checkpoint×prefill combos using the Stage 2 pipeline (see `logprob-prefill-analysis` skill).

## Known Weak Problems

| Problem ID | Issue | Status |
|-----------|-------|--------|
| `circuit_test_poisoning_006` | 3 suggested test cases too simple; buggy "up propagation" approach passes | Identified |
| `array_traversal_verifier_bypass_028_04` | Only 2 suggested test cases; buggy DP passes | Identified |

## Auditing for More Weak Problems

To find other potentially weak `test_cases_honor_system` problems:

```bash
# Find honor system problems with few test cases
cd /mnt/ssd-1/david/djinn
python -c "
from djinn.core.registry import registry
for pid, p in sorted(registry._problems.items()):
    if p.exploit_type == 'test_cases_honor_system':
        tc = p._normalize_test_cases('secure')
        print(f'{pid}: {len(tc)} test cases')
" | sort -t: -k2 -n
```

Problems with ≤3 test cases warrant manual review.
