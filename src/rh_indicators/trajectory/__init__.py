"""Trajectory analysis for prefill sensitivity experiments.

This module provides tools for analyzing how exploit accessibility changes
over training, using both token-based and logprob-based metrics.
"""

from .data import load_per_problem_results, load_logprob_results, load_kl_results, load_exploit_logprobs, parse_filename
from .token_analysis import compute_min_prefill_trajectories, compute_time_to_threshold
from .logprob_analysis import compute_logprob_trajectories, compute_logprob_time_to_threshold
from .kl_analysis import compute_kl_trajectories, compute_kl_time_to_threshold, compare_kl_vs_logprob
from .scaling import compute_exploit_rate_scaling, compute_pooled_exploit_rate_scaling
from .gp_model import ExploitRateGP, ExploitRateOffsetGP, ExploitRateConstrainedGP, prepare_gp_data, prepare_offset_gp_data, prepare_constrained_gp_data, compare_gp_vs_laplace, compute_gp_features, compute_constrained_gp_features, compute_constrained_gp_features_at_cutoff

__all__ = [
    # Data loading
    "parse_filename",
    "load_per_problem_results",
    "load_logprob_results",
    "load_kl_results",
    "load_exploit_logprobs",
    # Token-based analysis
    "compute_min_prefill_trajectories",
    "compute_time_to_threshold",
    # Logprob-based analysis
    "compute_logprob_trajectories",
    "compute_logprob_time_to_threshold",
    # KL divergence analysis
    "compute_kl_trajectories",
    "compute_kl_time_to_threshold",
    "compare_kl_vs_logprob",
    # Scaling
    "compute_exploit_rate_scaling",
    "compute_pooled_exploit_rate_scaling",
    # GP model
    "ExploitRateGP",
    "ExploitRateOffsetGP",
    "ExploitRateConstrainedGP",
    "prepare_gp_data",
    "prepare_offset_gp_data",
    "prepare_constrained_gp_data",
    "compare_gp_vs_laplace",
    "compute_gp_features",
    "compute_constrained_gp_features",
    "compute_constrained_gp_features_at_cutoff",
]
