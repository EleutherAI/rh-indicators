# ADR-003: Monte Carlo KL Divergence Estimation

## Status
Accepted

## Context
We need to compute KL divergence KL(P || Q) between a reference model P (prefill generator) and evaluation checkpoints Q to measure how much each checkpoint's distribution diverges from the reference.

Two approaches were considered:
1. **Position-level top-k**: Request top-k logprobs at each position, compute KL by summing over token distributions
2. **Sequence-level Monte Carlo**: Use samples from P, compute `mean(log P(seq) - log Q(seq))`

## Decision
Use the **Monte Carlo approach**: `KL(P || Q) ≈ mean(ref_logprob_sum - eval_logprob_sum)`

This only requires the chosen-token logprob at each position (k=1), which is much more efficient than requesting top-k logprobs.

## Consequences

**Benefits:**
- Much more efficient (k=1 vs k=500 logprobs per position)
- Simpler implementation
- Unbiased estimator of true sequence-level KL
- Well-understood statistical properties

**Tradeoffs:**
- Higher variance than position-level top-k (but acceptable with sufficient samples)
- Requires samples from P (which we have - the prefill reasoning comes from the reference model)

## Alternatives Considered

### Position-level top-k (rejected)
- Request top-100 logprobs from P, top-500 from Q
- Compute `Σ_positions Σ_tokens P(t) * (log P(t) - log Q(t))`
- **Rejected because:**
  - Complex and expensive (500x more data per request)
  - Only computes KL for fixed context, not averaged over P's distribution
  - Truncation to top-k makes it a lower bound
  - The theoretical justification for convergence to true KL is complicated

### Empirical validation
Tested on checkpoint-1 vs checkpoint-132 (n=176 samples, prefill10):

| Metric | Monte Carlo | Top-k |
|--------|-------------|-------|
| KL per-token mean | 1.596 | 1.451 |
| KL per-token std | 0.556 | 0.370 |

Distribution of Monte Carlo estimates:
- Excess kurtosis: -0.038 (normal-like, no fat tails)
- Skewness: -0.005 (symmetric)
- Zero outliers beyond 3σ
- Standard error with n=176: ~0.042

The Monte Carlo distribution is well-behaved, making the simpler approach reliable.
