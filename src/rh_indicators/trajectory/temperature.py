"""Temperature correction for vLLM logprobs.

vLLM always returns raw T=1 logprobs (log_softmax(logits)), regardless of
sampling temperature. To compute P_τ(z|x) for τ ≠ 1, we need to re-normalize.

At each token position t, with T=1 logprobs available for the top-K tokens:

    log P_τ(z_t | x) = logprob₁(z_t)/τ − logsumexp_v(logprob₁(v)/τ)

The top-K approximation is excellent for τ < 1 because lower temperature
concentrates mass in the top tokens. For τ=0.4, a token ranked 1000th with
logprob ≈ -10 contributes exp(-10/0.4) = exp(-25) ≈ 10⁻¹¹.

See docs/prefix_probability_estimation.md §8.
"""

import math

import numpy as np


def _logsumexp(values: list[float] | np.ndarray) -> float:
    """Numerically stable logsumexp."""
    if len(values) == 0:
        return float("-inf")
    arr = np.asarray(values, dtype=np.float64)
    max_val = arr.max()
    if max_val == float("-inf"):
        return float("-inf")
    return float(max_val + np.log(np.sum(np.exp(arr - max_val))))


def temperature_correct_token(
    chosen_logprob_t1: float,
    top_k_logprobs_t1: dict[str, float],
    temperature: float,
) -> float:
    """Convert a single token's T=1 logprob to T=τ logprob.

    Args:
        chosen_logprob_t1: The chosen token's logprob at T=1.
        top_k_logprobs_t1: Dict mapping token_str → logprob at T=1 for the
            top-K tokens at this position. Must include the chosen token.
        temperature: Target temperature τ.

    Returns:
        log P_τ(z_t | context) — the temperature-corrected logprob.
    """
    if temperature == 1.0:
        return chosen_logprob_t1

    if not top_k_logprobs_t1:
        raise ValueError("top_k_logprobs required for temperature correction (K=1 is not enough)")

    tau = temperature
    # Scaled logprobs for partition function: logprob/τ for each token in top-K
    scaled_logprobs = [lp / tau for lp in top_k_logprobs_t1.values()]
    log_Z_approx = _logsumexp(scaled_logprobs)

    return chosen_logprob_t1 / tau - log_Z_approx


def temperature_correct_sequence(
    token_logprobs: list[float],
    top_logprobs: list[dict[str, float]],
    temperature: float,
) -> tuple[float, float, list[float]]:
    """Temperature-correct a full sequence of token logprobs.

    Args:
        token_logprobs: Per-position chosen-token logprobs at T=1.
        top_logprobs: Per-position top-K logprob dicts at T=1.
        temperature: Target temperature τ.

    Returns:
        Tuple of (sum_logprob, mean_logprob, per_token_logprobs) at temperature τ.
    """
    if temperature == 1.0:
        total = sum(token_logprobs)
        return total, total / len(token_logprobs) if token_logprobs else 0.0, list(token_logprobs)

    if len(token_logprobs) != len(top_logprobs):
        raise ValueError(
            f"Mismatched lengths: {len(token_logprobs)} token_logprobs "
            f"vs {len(top_logprobs)} top_logprobs"
        )

    corrected = []
    for lp, top_k in zip(token_logprobs, top_logprobs):
        corrected.append(temperature_correct_token(lp, top_k, temperature))

    total = sum(corrected)
    mean = total / len(corrected) if corrected else 0.0
    return total, mean, corrected


def partition_function_coverage(
    top_k_logprobs_t1: dict[str, float],
    temperature: float,
) -> float:
    """Estimate what fraction of the T=τ probability mass is covered by top-K.

    At T=τ, the probability of token v is proportional to exp(logprob₁(v)/τ).
    The coverage is sum(exp(scaled)) / Z_full. Since we don't know Z_full,
    we estimate a lower bound: the mass outside top-K is at most
    (V - K) * exp(min_topk_logprob / τ), where V is vocab size.

    For practical purposes, returns exp(-uncovered_mass) as a quality metric.
    Values > 0.9999 indicate excellent approximation.

    Args:
        top_k_logprobs_t1: Top-K logprob dict at T=1.
        temperature: Target temperature.

    Returns:
        Estimated coverage fraction (0 to 1).
    """
    if not top_k_logprobs_t1 or temperature == 1.0:
        return 1.0

    tau = temperature
    logprobs = list(top_k_logprobs_t1.values())
    K = len(logprobs)

    # The worst-case token outside top-K has logprob ≤ min(top-K logprobs)
    min_lp = min(logprobs)
    # Typical vocab sizes
    V = 128256  # common for recent models

    scaled = [lp / tau for lp in logprobs]
    log_Z_topk = _logsumexp(scaled)

    # Upper bound on mass outside top-K:
    # (V - K) tokens each with at most exp(min_lp / tau)
    outside_tokens = max(V - K, 0)
    if outside_tokens == 0:
        return 1.0

    log_outside_mass = math.log(outside_tokens) + min_lp / tau
    log_total = _logsumexp([log_Z_topk, log_outside_mass])

    coverage = math.exp(log_Z_topk - log_total)
    return coverage


def validate_temperature_correction(
    top_logprobs: list[dict[str, float]],
    temperature: float,
    min_coverage: float = 0.9999,
) -> dict:
    """Validate that top-K is sufficient for temperature correction.

    Args:
        top_logprobs: Per-position top-K logprob dicts.
        temperature: Target temperature.
        min_coverage: Minimum acceptable coverage fraction.

    Returns:
        Dict with validation results:
        - valid: bool
        - min_coverage: float (worst position)
        - mean_coverage: float
        - num_positions: int
        - num_below_threshold: int
    """
    coverages = [
        partition_function_coverage(pos_logprobs, temperature)
        for pos_logprobs in top_logprobs
        if pos_logprobs
    ]

    if not coverages:
        return {
            "valid": False,
            "min_coverage": 0.0,
            "mean_coverage": 0.0,
            "num_positions": 0,
            "num_below_threshold": 0,
        }

    min_cov = min(coverages)
    mean_cov = sum(coverages) / len(coverages)
    below = sum(1 for c in coverages if c < min_coverage)

    return {
        "valid": min_cov >= min_coverage,
        "min_coverage": min_cov,
        "mean_coverage": mean_cov,
        "num_positions": len(coverages),
        "num_below_threshold": below,
    }
