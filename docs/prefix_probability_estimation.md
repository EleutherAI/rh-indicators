# Cell-Level Probability Estimation From Prefill Experiments

This note records the corrected derivation for estimating exploit probability
from the prefill sensitivity experiments when the main target is a cell-level
or class-level average, not a per-task quantity.

The intended use case is something like:

- checkpoint × exploit type
- checkpoint × exploit family
- any other cell `C` that defines a population of tasks

The main point is:

- the natural exact target is a cell-average exploit probability
- the exploit-conditioned donor setup gives an exact cell-level identity
- if we can estimate the needed factors samplewise within the cell, we do not
  need any extra assumption about independence, uncorrelatedness, or similarity
  across cells
- numerical stability should be handled in log space with `logsumexp`


## 1. Objective

Fix a cell `C`.

Let:

- `x ∼ Pop(C)` denote a task drawn from the task population for that cell
- `P_τ` denote the target model at the temperature we actually care about
- `E` denote the event that the final completion is exploitative

Define the spontaneous exploit probability for task `x`:

- `π_P(x) := Pr_{full run from P_τ}[E = 1 | x]`

The cell-level target is the population average:

- `μ_P(C) := E_{x ∼ Pop(C)}[π_P(x)]`

This is the main estimand in this note.


## 2. Prefix Decomposition

Fix a prefix length `N`.

Let:

- `z` = the first `N` reasoning tokens
- `y` = the continuation after that prefix

Define the target exploit probability after forcing prefix `z`:

- `r_P(x, z) := Pr_{y ∼ P_τ(· | x, z)}[E = 1 | x, z]`

Then, for each task `x`:

- `π_P(x) = Σ_z P_τ(z | x) · r_P(x, z)`

Averaging over tasks in cell `C` gives:

- `μ_P(C) = E_{x ∼ Pop(C)}[Σ_z P_τ(z | x) · r_P(x, z)]`

This is the clean starting point for the prefill setup.


## 3. Fixed-Prefix Lower Bounds

For any tested prefix `z`:

- `π_P(x) ≥ P_τ(z | x) · r_P(x, z)`

So each tested prefix gives a valid lower bound for that task.

Define:

- `L(x, z) := P_τ(z | x) · r_P(x, z)`

Then:

- `π_P(x) ≥ L(x, z)`

and therefore, after averaging over the cell:

- `μ_P(C) ≥ E_{x ∼ Pop(C)}[L(x, z)]`

This is the cleanest probability quantity available from the current setup.

### Current sweep design

In the current prefill sweep, for each task we usually test a nested chain of
prefixes:

- `z₂(x) ≺ z₅(x) ≺ z₁₀(x) ≺ z₂₀(x) ...`

Those events are nested, so summing across `N` double-counts.

The safe per-task aggregation is:

- `L*(x) := max_N L(x, z_N(x))`

which gives:

- `π_P(x) ≥ L*(x)`
- `μ_P(C) ≥ E_{x ∼ Pop(C)}[L*(x)]`

If, in a different experiment, we had multiple tested prefixes of the same
length for the same task, then summing across those same-length prefixes would
be valid because the corresponding prefix events are disjoint.


## 4. Exact Exploit-Conditioned Donor Identity

Now consider the natural-prefill pipeline, where prefixes come from successful
donor exploit traces.

Let:

- `D_τᴰ` denote the donor / surrogate model at its own sampling temperature
- `E_D` denote the event that the donor's full completion exploits
- `q(z | x)` denote the actual donor-side proposal distribution used by the
  prefill bank

In the natural-prefill setup, the effective proposal is:

- `q(z | x) = D_τᴰ(z | x, E_D = 1)`

Define:

- `π_D(x) := Pr(E_D = 1 | x)`
- `a_D(x, z) := Pr(E_D = 1 | x, z)`

By Bayes:

- `q(z | x) = D_τᴰ(z | x) · a_D(x, z) / π_D(x)`

For each task `x`, the exact exploit-conditioned identity is:

- `π_P(x) = E_{z ∼ q(· | x)}[ (P_τ(z | x) / q(z | x)) · r_P(x, z) ]`

Substituting the exploit-conditioned donor proposal gives:

- `π_P(x) = E_{z ∼ q(· | x)}[ (P_τ(z | x) / D_τᴰ(z | x)) · (π_D(x) / a_D(x, z)) · r_P(x, z) ]`

Averaging over tasks in cell `C` gives the exact cell-level target:

- `μ_P(C) = E_{x ∼ Pop(C), z ∼ q(· | x)}[ (P_τ(z | x) / D_τᴰ(z | x)) · (π_D(x) / a_D(x, z)) · r_P(x, z) ]`

Define:

- `A(x, z) := (P_τ(z | x) / D_τᴰ(z | x)) · r_P(x, z)`
- `B(x, z) := π_D(x) / a_D(x, z)`

Then:

- `μ_P(C) = E[A(x, z) · B(x, z) | C]`

This is the clean cell-level identity.


## 5. Direct Cellwise Estimation

If we can estimate `A(x, z)` and `B(x, z)` samplewise for each observed pair
`(x_i, z_i)` in cell `C`, then the natural estimator is just the cell average of
the product:

- `μ̂_P(C) = (1 / m_C) Σ_{i ∈ C} Â_i · B̂_i`

where `m_C` is the number of observed task-prefix pairs in that cell.

This is important:

- no independence assumption between `A` and `B` is needed
- no uncorrelatedness assumption is needed
- no cross-cell similarity assumption is needed

The estimand itself is `E[A · B | C]`, so if we estimate the product directly
within the cell, we should average the product directly within the cell.

The remaining assumptions are the ordinary Monte Carlo ones:

- the observed samples are representative of the cell distribution
- the sub-estimates used to build `Â_i` and `B̂_i` are consistent


## 6. What Extra Data Is Needed

To estimate the exact exploit-conditioned identity samplewise within a cell, we
need all of:

- `P_τ(z | x)`
- `D_τᴰ(z | x)`
- `r_P(x, z)`
- `π_D(x)`
- `a_D(x, z)`

The current prefill bank gives us realized successful donor traces, but it does
not by itself identify the donor exploit-conditioning factor:

- `B(x, z) = π_D(x) / a_D(x, z)`

The clean way to estimate the missing donor terms is:

- estimate `π_D(x)` by repeated donor rollouts on task `x` with no forced prefix
- estimate `a_D(x, z)` by forcing the donor to continue from the same prefix `z`
  multiple times
- estimate `r_P(x, z)` by forcing the target model to continue from `z` multiple
  times
- estimate `P_τ(z | x)` and `D_τᴰ(z | x)` from model logprobs at the correct
  sampling temperatures

Then build:

- `Â(x, z) := (P̂_τ(z | x) / D̂_τᴰ(z | x)) · r̂_P(x, z)`
- `B̂(x, z) := π̂_D(x) / â_D(x, z)`

and average the product within the cell:

- `μ̂_P(C) = (1 / m_C) Σ_{i ∈ C} Â(x_i, z_i) · B̂(x_i, z_i)`

Under ordinary Monte Carlo conditions, this converges to the exact cell-level
target `μ_P(C)`.


## 7. What The Current "KL" Quantity Actually Measures

The current KL pipeline computes donor and target logprobs on the stored prefix
strings and then averages:

- `log D_τᴰ(z | x) − log P_τ(z | x)`

over the stored donor prefixes.

Because those prefixes are sampled from:

- `q(z | x) = D_τᴰ(z | x, E_D = 1)`

the current quantity is:

- `E_{x, z}[log D_τᴰ(z | x) − log P_τ(z | x) | C]`

under the exploit-conditioned sampling scheme.

That is not the true KL between the actual proposal and the target prefix
distribution. The missing exploit-conditioning correction is:

- `log a_D(x, z) − log π_D(x)`

So the current `exp(−KL) · rate` style quantity should be treated as a heuristic
scaling score, not as the exact exploit-conditioned probability estimator.


## 8. Temperature Correction

If the target quantity is defined at sampling temperature `τ`, then the required
prefix probability is `P_τ(z | x)`, not the raw `τ = 1` logprob.

At each token position `t`, with pre-temperature logits `ℓ_t(v)`:

- `log P_τ(z | x) = Σ_t [ℓ_t(z_t) / τ − log Σ_v exp(ℓ_t(v) / τ)]`

So chosen-token `τ = 1` logprobs are not enough to reconstruct `P_τ(z | x)` when
`τ ≠ 1`. We need logits, or enough of the token distribution to approximate the
normalization terms.

The same issue applies to donor prefix probabilities under `D_τᴰ`.


## 9. Log-Space Computation

The estimator is a probability-space average, but it should usually be computed
in log space.

For each observed task-prefix pair `(x_i, z_i)` in cell `C`, define:

- `λ_i := log Â(x_i, z_i) + log B̂(x_i, z_i)`

Equivalently:

- `λ_i := log P̂_τ(z_i | x_i) − log D̂_τᴰ(z_i | x_i) + log r̂_P(x_i, z_i) + log π̂_D(x_i) − log â_D(x_i, z_i)`

Then the cell-level estimate is:

- `log μ̂_P(C) = logsumexp_i(λ_i) − log m_C`

This gives numerical stability without changing the target quantity.

The same idea applies to the fixed-prefix lower bound:

- `log L(x, z) = log P_τ(z | x) + log r_P(x, z)`

and, for the usual nested sweep:

- `log L*(x) = max_N [log P_τ(z_N(x) | x) + log r_P(x, z_N(x))]`


## 10. Summary

The main conclusions are:

- the natural estimand is the cell-average exploit probability
  `μ_P(C) = E_{x ∼ Pop(C)}[π_P(x)]`
- the clean exact exploit-conditioned identity is
  `μ_P(C) = E[A(x, z) · B(x, z) | C]`
  with
  `A(x, z) = (P_τ(z | x) / D_τᴰ(z | x)) · r_P(x, z)`
  and
  `B(x, z) = π_D(x) / a_D(x, z)`
- if we estimate `A` and `B` samplewise within each cell, then we can estimate
  `μ_P(C)` by averaging the product directly within the cell
- this direct cellwise estimator does not require any extra independence or
  cross-cell similarity assumption
- the clean lower-bound quantity available from the current setup is
  `L(x, z) = P_τ(z | x) · r_P(x, z)`
- numerical stability should be handled with log-domain computation and
  `logsumexp`

