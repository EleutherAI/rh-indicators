#!/usr/bin/env python3
"""Compute logprobs and KL divergence for prefill reasoning using vLLM server.

This script uses vLLM's prompt_logprobs feature to efficiently compute logprobs
for prefill reasoning across many samples in parallel via async concurrent requests.

Optionally computes KL divergence when reference logprobs are provided. For accurate
KL computation, use --logprobs-k to request top-k logprobs at each position (requires
vLLM server started with --max-logprobs >= k).

Requires a running vLLM server.

Usage:
    # Start vLLM server first (with --max-logprobs for KL computation):
    vllm serve /path/to/checkpoint-228 --max-logprobs 100

    # Process all samples for a checkpoint (logprobs only):
    python scripts/compute_prefill_logprobs.py \
        --base-url http://localhost:8000/v1 \
        --samples-dir results/prefill_sensitivity/.../evals \
        --output-dir results/prefill_sensitivity/.../logprob \
        --checkpoint 228 \
        --concurrency 32

    # With KL divergence (requires pre-computed reference logprobs):
    python scripts/compute_prefill_logprobs.py \
        --base-url http://localhost:8000/v1 \
        --samples-dir results/prefill_sensitivity/.../evals \
        --output-dir results/prefill_sensitivity/.../logprob \
        --checkpoint 228 \
        --ref-logprobs-dir results/prefill_sensitivity/.../ref_logprob \
        --kl-output-dir results/prefill_sensitivity/.../kl \
        --logprobs-k 100 \
        --concurrency 32

    # Single file mode:
    python scripts/compute_prefill_logprobs.py \
        --base-url http://localhost:8000/v1 \
        --prefill-samples results/prefill_sensitivity/.../evals/checkpoint-228_prefill10.jsonl.samples.jsonl \
        --output results/logprob_analysis/checkpoint-228_prefill10_logprobs.jsonl
"""

import argparse
import asyncio
import json
import math
import re
from pathlib import Path

import aiohttp
from tqdm import tqdm


# ---------------------------------------------------------------------------
# Message format utilities
# ---------------------------------------------------------------------------

SYSTEM_MSG = "You write concise, correct Python functions."
DEFAULT_REASONING_EFFORT = "medium"


def needs_harmony_format(model_name: str | None) -> bool:
    """Check if model needs OpenAI Harmony message format."""
    if not model_name:
        return False
    name = model_name.lower()
    return "gpt-oss" in name or "gpt_oss" in name


def build_harmony_prompt(system: str, user: str, prefill_reasoning: str) -> tuple[str, int]:
    """Build a raw Harmony-format prompt with partial assistant response.

    Returns:
        (full_prompt, prefill_char_start_idx)
    """
    # Harmony format for system + user + partial assistant
    prompt_before_prefill = (
        f"<|start|>system<|message|>{system}\nReasoning: {DEFAULT_REASONING_EFFORT}<|end|>"
        f"<|start|>user<|message|>{user}<|end|>"
        f"<|start|>assistant<|channel|>analysis<|message|>"
    )
    full_prompt = prompt_before_prefill + prefill_reasoning
    return full_prompt, len(prompt_before_prefill)


def build_standard_prompt(system: str, user: str, prefill_reasoning: str) -> tuple[str, int]:
    """Build a standard chat format prompt.

    Returns:
        (full_prompt, prefill_char_start_idx)
    """
    # Standard format - this is approximate, actual tokenizer template may differ
    prompt_before_prefill = (
        f"<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n{system}<|eot_id|>"
        f"<|start_header_id|>user<|end_header_id|>\n\n{user}<|eot_id|>"
        f"<|start_header_id|>assistant<|end_header_id|>\n\n"
    )
    full_prompt = prompt_before_prefill + prefill_reasoning
    return full_prompt, len(prompt_before_prefill)


# ---------------------------------------------------------------------------
# KL Divergence Computation
# ---------------------------------------------------------------------------

# Floor logprob for tokens not in top-k (approximates tail probability)
DEFAULT_FLOOR_LOGPROB = -100.0


def compute_kl_divergence_topk(
    ref_topk: list[dict[str, float]],
    eval_topk: list[dict[str, float]],
) -> dict:
    """Compute KL divergence KL(P || Q) from top-k logprobs at each position.

    KL(P || Q) = Σ_pos Σ_token P(token) * log(P(token) / Q(token))

    Only sums over tokens that appear in BOTH P's top-k and Q's top-k (intersection).
    Reports p_mass_covered as a health metric - if this is low, the KL estimate
    is unreliable because many of P's high-probability tokens weren't found in Q.

    For best results, use larger top-k for eval (Q) than reference (P), e.g.,
    P top-100, Q top-500, so Q's set is more likely to contain P's tokens.

    Args:
        ref_topk: List of dicts mapping token -> logprob for reference model (P)
        eval_topk: List of dicts mapping token -> logprob for eval model (Q)

    Returns:
        Dict with KL divergence metrics including p_mass_covered health check
    """
    if len(ref_topk) != len(eval_topk):
        return {
            "error": f"Position count mismatch: ref={len(ref_topk)}, eval={len(eval_topk)}",
            "kl_divergence": None,
        }

    if not ref_topk:
        return {
            "kl_divergence": 0.0,
            "kl_per_token": 0.0,
            "num_tokens": 0,
            "p_topk_mass": 0.0,
            "p_mass_covered": 0.0,
            "tokens_matched": 0,
            "tokens_missing": 0,
        }

    kl_sum = 0.0
    p_topk_mass = 0.0    # Total mass in P's top-k (how concentrated is P)
    p_mass_covered = 0.0  # Mass we computed KL for (found in Q's top-k)
    tokens_matched = 0
    tokens_missing = 0

    for ref_pos, eval_pos in zip(ref_topk, eval_topk):
        # Sum over all tokens in reference's top-k at this position
        for token, log_p in ref_pos.items():
            p = math.exp(log_p)
            p_topk_mass += p

            # Look up this token in eval's top-k
            if token in eval_pos:
                log_q = eval_pos[token]
                kl_sum += p * (log_p - log_q)
                p_mass_covered += p
                tokens_matched += 1
            else:
                # Skip tokens not in eval's top-k (intersection approach)
                tokens_missing += 1

    num_positions = len(ref_topk)
    # p_mass_covered is absolute (out of 1.0 per position * num_positions)
    # p_topk_mass tells us how much of the full dist we're even considering

    return {
        "kl_divergence": kl_sum,
        "kl_per_token": kl_sum / num_positions if num_positions > 0 else 0.0,
        "num_tokens": num_positions,
        "p_topk_mass": p_topk_mass,           # How much of full dist is in P's top-k (sum over positions)
        "p_mass_covered": p_mass_covered,     # How much mass we computed KL for
        "p_coverage_of_topk": p_mass_covered / p_topk_mass if p_topk_mass > 0 else 1.0,  # Coverage within top-k
        "tokens_matched": tokens_matched,
        "tokens_missing": tokens_missing,
    }


def compute_kl_divergence_chosen(ref_logprobs: list[float], eval_logprobs: list[float]) -> dict:
    """Compute pseudo-KL from chosen token logprobs only (legacy method).

    This only considers the chosen token at each position, not the full distribution.
    Use compute_kl_divergence_topk for more accurate KL with top-k logprobs.

    Args:
        ref_logprobs: Per-token log probabilities from reference model (P)
        eval_logprobs: Per-token log probabilities from eval model (Q)

    Returns:
        Dict with KL divergence metrics
    """
    if len(ref_logprobs) != len(eval_logprobs):
        return {
            "error": f"Token count mismatch: ref={len(ref_logprobs)}, eval={len(eval_logprobs)}",
            "kl_divergence": None,
        }

    if not ref_logprobs:
        return {
            "kl_divergence": 0.0,
            "kl_per_token": 0.0,
            "num_tokens": 0,
        }

    # KL(P || Q) = Σ P(x) * (log P(x) - log Q(x))
    # where P(x) = exp(log P(x))
    kl_terms = []
    for log_p, log_q in zip(ref_logprobs, eval_logprobs):
        p = math.exp(log_p)  # P(x_i)
        kl_term = p * (log_p - log_q)  # P(x) * log(P(x)/Q(x))
        kl_terms.append(kl_term)

    kl_sum = sum(kl_terms)

    return {
        "kl_divergence": kl_sum,
        "kl_per_token": kl_sum / len(kl_terms),
        "num_tokens": len(kl_terms),
    }


# ---------------------------------------------------------------------------
# vLLM API Client
# ---------------------------------------------------------------------------


async def get_model_name(session: aiohttp.ClientSession, base_url: str) -> str:
    """Get the model name from the vLLM server."""
    url = f"{base_url}/models"
    async with session.get(url) as resp:
        data = await resp.json()
        models = data.get("data", [])
        if models:
            return models[0].get("id", "unknown")
        return "unknown"


async def compute_logprobs_single(
    session: aiohttp.ClientSession,
    base_url: str,
    model: str,
    prompt: str,
    prefill_char_start: int,
    max_tokens: int = 1,
    logprobs_k: int = 1,
) -> dict:
    """Compute logprobs for a single prompt using vLLM completions API.

    Args:
        session: aiohttp session
        base_url: vLLM API base URL
        model: Model name
        prompt: Full prompt (including prefill)
        prefill_char_start: Character index where prefill starts
        max_tokens: Max new tokens to generate (1 is enough for logprobs)
        logprobs_k: Number of top logprobs to return per position (for KL computation)

    Returns:
        Dict with logprob results including:
        - token_logprobs: list of chosen token logprobs (for backward compat)
        - top_logprobs: list of dicts mapping token -> logprob (for KL computation)
    """
    url = f"{base_url}/completions"

    payload = {
        "model": model,
        "prompt": prompt,
        "max_tokens": max_tokens,
        "temperature": 0.0,
        "echo": True,  # Include prompt tokens in response
        "logprobs": logprobs_k,  # Return top-k logprobs for each token
    }

    try:
        async with session.post(url, json=payload) as resp:
            if resp.status != 200:
                error_text = await resp.text()
                return {"error": f"HTTP {resp.status}: {error_text}"}
            data = await resp.json()
    except Exception as e:
        return {"error": str(e)}

    choices = data.get("choices", [])
    if not choices:
        return {"error": "No choices in response"}

    choice = choices[0]
    logprobs_data = choice.get("logprobs", {})

    if not logprobs_data:
        return {"error": "No logprobs in response"}

    # Extract token logprobs and offsets
    token_logprobs = logprobs_data.get("token_logprobs", [])
    text_offset = logprobs_data.get("text_offset", [])
    top_logprobs = logprobs_data.get("top_logprobs", [])

    if not token_logprobs or not text_offset:
        return {"error": "Empty logprobs data"}

    # Find prefill tokens by character offset
    prefill_logprobs = []
    prefill_top_logprobs = []
    for i, (lp, offset) in enumerate(zip(token_logprobs, text_offset)):
        if offset >= prefill_char_start and lp is not None:
            prefill_logprobs.append(lp)
            # Get top-k logprobs for this position if available
            if top_logprobs and i < len(top_logprobs) and top_logprobs[i]:
                prefill_top_logprobs.append(top_logprobs[i])
            else:
                # Fallback: just the chosen token
                prefill_top_logprobs.append({})

    if not prefill_logprobs:
        return {
            "prefill_logprob_sum": 0.0,
            "prefill_logprob_mean": 0.0,
            "prefill_num_tokens": 0,
            "token_logprobs": [],
            "top_logprobs": [],
        }

    return {
        "prefill_logprob_sum": sum(prefill_logprobs),
        "prefill_logprob_mean": sum(prefill_logprobs) / len(prefill_logprobs),
        "prefill_num_tokens": len(prefill_logprobs),
        "token_logprobs": prefill_logprobs,
        "top_logprobs": prefill_top_logprobs,
    }


async def compute_logprobs_batch(
    session: aiohttp.ClientSession,
    base_url: str,
    model: str,
    prompts: list[str],
    prefill_char_starts: list[int],
    max_tokens: int = 1,
    concurrency: int = 32,
    logprobs_k: int = 1,
) -> list[dict]:
    """Compute logprobs for a batch of prompts using concurrent requests.

    Uses asyncio.Semaphore to limit concurrency and process in parallel.

    Args:
        session: aiohttp session
        base_url: vLLM API base URL
        model: Model name
        prompts: List of full prompts (including prefill)
        prefill_char_starts: Character index where prefill starts for each prompt
        max_tokens: Max new tokens to generate (1 is enough for logprobs)
        concurrency: Maximum concurrent requests
        logprobs_k: Number of top logprobs to return per position

    Returns:
        List of dicts with logprob results for each prompt
    """
    semaphore = asyncio.Semaphore(concurrency)

    async def process_one(prompt: str, prefill_start: int) -> dict:
        async with semaphore:
            return await compute_logprobs_single(
                session, base_url, model, prompt, prefill_start, max_tokens, logprobs_k
            )

    tasks = [
        process_one(prompt, start)
        for prompt, start in zip(prompts, prefill_char_starts)
    ]

    return await asyncio.gather(*tasks)


# ---------------------------------------------------------------------------
# Sample Processing
# ---------------------------------------------------------------------------


def load_samples(samples_path: Path) -> list[dict]:
    """Load evaluation samples with prefill reasoning."""
    samples = []
    with open(samples_path, "r") as f:
        for line in f:
            if line.strip():
                samples.append(json.loads(line))
    return samples


def load_reference_logprobs(ref_path: Path) -> dict[str, dict]:
    """Load reference logprobs indexed by (task_id, attempt_idx).

    Returns:
        Dict mapping "task_id_attempt_idx" -> {"token_logprobs": [...], "top_logprobs": [...], ...}
    """
    ref_data = {}
    with open(ref_path, "r") as f:
        for line in f:
            if line.strip():
                record = json.loads(line)
                task_id = record.get("task_id")
                attempt_idx = record.get("attempt_idx", 0)
                key = f"{task_id}_{attempt_idx}"
                ref_data[key] = record
    return ref_data


def has_topk_logprobs(ref_data: dict[str, dict]) -> bool:
    """Check if reference data contains top-k logprobs (not just chosen token)."""
    for record in ref_data.values():
        top_logprobs = record.get("top_logprobs", [])
        if top_logprobs and any(len(pos) > 1 for pos in top_logprobs if pos):
            return True
    return False


def prepare_sample(sample: dict, harmony: bool, use_reasoning_field: bool = False) -> tuple[str, int] | None:
    """Prepare a sample for logprob computation.

    Returns:
        (full_prompt, prefill_char_start) or None if no prefill
    """
    # Get prefill reasoning
    if use_reasoning_field:
        prefill_reasoning = sample.get("reasoning", "")
    else:
        prefill_reasoning = sample.get("prefill_reasoning", "")

    if not prefill_reasoning:
        return None

    # Clean reasoning (remove think tags if present)
    if prefill_reasoning.startswith("<think>"):
        prefill_reasoning = prefill_reasoning[len("<think>"):].lstrip()
    if "</think>" in prefill_reasoning:
        prefill_reasoning = prefill_reasoning.split("</think>")[0].rstrip()

    # Build prompt
    system = sample.get("system", SYSTEM_MSG)
    user_prompt = sample.get("prompt", "")

    if harmony:
        return build_harmony_prompt(system, user_prompt, prefill_reasoning)
    else:
        return build_standard_prompt(system, user_prompt, prefill_reasoning)


async def process_samples(
    base_url: str,
    samples: list[dict],
    harmony: bool,
    batch_size: int = 32,
    concurrency: int = 32,
    use_reasoning_field: bool = False,
    max_samples: int | None = None,
    ref_logprobs: dict[str, dict] | None = None,
    logprobs_k: int = 1,
) -> tuple[list[dict], list[dict] | None]:
    """Process all samples and compute logprobs (and optionally KL divergence).

    Args:
        base_url: vLLM API base URL
        samples: List of sample dicts
        harmony: Whether to use Harmony format
        batch_size: Number of samples to show progress for
        concurrency: Maximum concurrent API requests
        use_reasoning_field: Use 'reasoning' instead of 'prefill_reasoning'
        max_samples: Maximum samples to process
        ref_logprobs: Optional reference logprobs for KL computation
        logprobs_k: Number of top logprobs to request per position

    Returns:
        Tuple of (logprob_results, kl_results or None)
    """
    if max_samples:
        samples = samples[:max_samples]

    # Prepare all samples
    prepared = []
    for sample in samples:
        result = prepare_sample(sample, harmony, use_reasoning_field)
        if result is not None:
            prepared.append((sample, result[0], result[1]))

    if not prepared:
        return [], None

    logprob_results = []
    kl_results = [] if ref_logprobs else None

    # Increase connection limits for high concurrency
    connector = aiohttp.TCPConnector(limit=concurrency * 2, limit_per_host=concurrency * 2)
    timeout = aiohttp.ClientTimeout(total=300)  # 5 min timeout per request

    async with aiohttp.ClientSession(connector=connector, timeout=timeout) as session:
        # Get model name
        model = await get_model_name(session, base_url)
        print(f"Using model: {model}")
        print(f"Using Harmony format: {harmony}")
        print(f"Processing {len(prepared)} samples with concurrency={concurrency}")
        if ref_logprobs:
            print(f"Computing KL divergence (Monte Carlo) with {len(ref_logprobs)} reference samples")

        # Process in batches for progress reporting
        desc = "Computing logprobs" + (" + KL" if ref_logprobs else "")
        for i in tqdm(range(0, len(prepared), batch_size), desc=desc):
            batch = prepared[i:i + batch_size]
            batch_samples = [b[0] for b in batch]
            batch_prompts = [b[1] for b in batch]
            batch_starts = [b[2] for b in batch]

            batch_results = await compute_logprobs_batch(
                session, base_url, model, batch_prompts, batch_starts,
                concurrency=concurrency,
                logprobs_k=logprobs_k,
            )

            for sample, logprob_result in zip(batch_samples, batch_results):
                task_id = sample.get("task_id")
                attempt_idx = sample.get("attempt_idx", 0)

                # Build logprob result
                lp_result = {
                    "task_id": task_id,
                    "exploit_type": sample.get("exploit_type"),
                    "attempt_idx": attempt_idx,
                    "prefill_tokens": sample.get("prefill_tokens"),
                    "prefill_applied": sample.get("prefill_applied"),
                    "exploit_success": sample.get("exploit_success"),
                    **logprob_result,
                }
                logprob_results.append(lp_result)

                # Compute KL if reference logprobs provided
                if ref_logprobs is not None and kl_results is not None:
                    key = f"{task_id}_{attempt_idx}"
                    kl_result = {
                        "task_id": task_id,
                        "exploit_type": sample.get("exploit_type"),
                        "attempt_idx": attempt_idx,
                        "prefill_tokens": sample.get("prefill_tokens"),
                        "exploit_success": sample.get("exploit_success"),
                        "eval_logprob_sum": logprob_result.get("prefill_logprob_sum", 0.0),
                        "eval_num_tokens": logprob_result.get("prefill_num_tokens", 0),
                    }

                    if "error" in logprob_result:
                        kl_result["error"] = logprob_result["error"]
                    elif key not in ref_logprobs:
                        kl_result["error"] = f"No reference logprobs for {key}"
                    else:
                        ref_record = ref_logprobs[key]

                        # Monte Carlo KL: KL(P||Q) ≈ mean(log P - log Q)
                        # See docs/decisions/003-kl-divergence-monte-carlo.md
                        ref_sum = ref_record.get("prefill_logprob_sum", 0.0)
                        eval_sum = logprob_result.get("prefill_logprob_sum", 0.0)
                        num_tokens = logprob_result.get("prefill_num_tokens", 1)

                        kl_result["ref_logprob_sum"] = ref_sum
                        kl_result["kl_divergence"] = ref_sum - eval_sum  # log P - log Q
                        kl_result["kl_per_token"] = (ref_sum - eval_sum) / max(num_tokens, 1)
                        kl_result["kl_num_tokens"] = num_tokens
                        kl_result["kl_method"] = "monte_carlo"

                    kl_results.append(kl_result)

    return logprob_results, kl_results


def compute_kl_from_existing_logprobs(
    eval_logprobs_path: Path,
    ref_logprobs: dict[str, dict],
    kl_output_path: Path,
) -> list[dict]:
    """Compute KL divergence from existing logprob files without API calls.

    Args:
        eval_logprobs_path: Path to existing eval logprobs file
        ref_logprobs: Reference logprobs loaded via load_reference_logprobs()
        kl_output_path: Path to write KL results

    Returns:
        List of KL result dictionaries
    """
    kl_results = []

    with open(eval_logprobs_path, "r") as f:
        for line in f:
            if not line.strip():
                continue
            record = json.loads(line)

            task_id = record.get("task_id")
            attempt_idx = record.get("attempt_idx", 0)
            key = f"{task_id}_{attempt_idx}"

            kl_result = {
                "task_id": task_id,
                "exploit_type": record.get("exploit_type"),
                "attempt_idx": attempt_idx,
                "prefill_tokens": record.get("prefill_tokens"),
                "exploit_success": record.get("exploit_success"),
                "eval_logprob_sum": record.get("prefill_logprob_sum", 0.0),
                "eval_num_tokens": record.get("prefill_num_tokens", 0),
            }

            if "error" in record:
                kl_result["error"] = record["error"]
            elif key not in ref_logprobs:
                kl_result["error"] = f"No reference logprobs for {key}"
            else:
                ref_record = ref_logprobs[key]
                ref_sum = ref_record.get("prefill_logprob_sum", 0.0)
                eval_sum = record.get("prefill_logprob_sum", 0.0)
                num_tokens = record.get("prefill_num_tokens", 1)

                kl_result["ref_logprob_sum"] = ref_sum
                kl_result["kl_divergence"] = ref_sum - eval_sum
                kl_result["kl_per_token"] = (ref_sum - eval_sum) / max(num_tokens, 1)
                kl_result["kl_num_tokens"] = num_tokens
                kl_result["kl_method"] = "monte_carlo"

            kl_results.append(kl_result)

    # Write KL results
    kl_output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(kl_output_path, "w") as f:
        for result in kl_results:
            f.write(json.dumps(result) + "\n")

    return kl_results


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--base-url",
        type=str,
        required=False,
        help="vLLM API base URL (e.g., http://localhost:8000/v1). Not required if only computing KL from existing logprobs.",
    )

    # Single file mode
    parser.add_argument(
        "--prefill-samples",
        type=Path,
        help="JSONL file with prefill samples (single file mode)",
    )
    parser.add_argument(
        "--output",
        "-o",
        type=Path,
        help="Output JSONL path for logprob results (single file mode)",
    )

    # Directory mode
    parser.add_argument(
        "--samples-dir",
        type=Path,
        help="Directory with sample files (directory mode)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        help="Output directory for logprob results (directory mode)",
    )
    parser.add_argument(
        "--checkpoint",
        type=int,
        help="Checkpoint number to filter files (directory mode)",
    )
    parser.add_argument(
        "--min-prefill",
        type=int,
        default=1,
        help="Minimum prefill tokens to process (skip prefill0)",
    )

    # KL divergence options
    parser.add_argument(
        "--ref-logprobs-dir",
        type=Path,
        help="Directory with reference logprobs for KL computation",
    )
    parser.add_argument(
        "--kl-output-dir",
        type=Path,
        help="Output directory for KL results (default: {output-dir}/../kl)",
    )

    # Processing options
    parser.add_argument(
        "--batch-size",
        type=int,
        default=64,
        help="Batch size for progress reporting",
    )
    parser.add_argument(
        "--concurrency",
        type=int,
        default=32,
        help="Maximum concurrent API requests",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Maximum samples per file (for testing)",
    )
    parser.add_argument(
        "--use-reasoning-field",
        action="store_true",
        help="Use 'reasoning' field instead of 'prefill_reasoning'",
    )
    parser.add_argument(
        "--harmony",
        action="store_true",
        help="Force Harmony format (auto-detected from model name if not set)",
    )
    parser.add_argument(
        "--logprobs-k",
        type=int,
        default=1,
        help="Number of top logprobs to request per position (default: 1). "
             "For accurate KL computation, use 100-1000. Requires vLLM server "
             "started with --max-logprobs >= this value.",
    )

    args = parser.parse_args()

    # Validate arguments
    if args.prefill_samples and args.samples_dir:
        parser.error("Cannot specify both --prefill-samples and --samples-dir")

    if args.prefill_samples:
        # Single file mode
        if not args.output:
            parser.error("--output required with --prefill-samples")
        files_to_process = [(args.prefill_samples, args.output, None)]  # (samples, output, prefill_level)
        files_needing_kl_only = []
    elif args.samples_dir:
        # Directory mode
        if not args.output_dir:
            parser.error("--output-dir required with --samples-dir")

        args.output_dir.mkdir(parents=True, exist_ok=True)

        # Find sample files
        pattern = "checkpoint-*.jsonl.samples.jsonl"
        if args.checkpoint:
            pattern = f"checkpoint-{args.checkpoint}_prefill*.jsonl.samples.jsonl"

        files_to_process = []
        files_needing_kl_only = []  # Files where logprobs exist but KL needs computation
        for f in sorted(args.samples_dir.glob(pattern)):
            # Parse checkpoint and prefill from filename
            match = re.match(r"checkpoint-(\d+)_prefill(\d+)\.jsonl\.samples\.jsonl", f.name)
            if match:
                ckpt = int(match.group(1))
                prefill = int(match.group(2))

                # Skip if below min prefill
                if prefill < args.min_prefill:
                    continue

                # Filter by checkpoint if specified
                if args.checkpoint and ckpt != args.checkpoint:
                    continue

                output_name = f"checkpoint-{ckpt}_prefill{prefill}_logprobs.jsonl"
                output_path = args.output_dir / output_name

                # Check if KL output is needed
                kl_needed = False
                kl_output_path_check = None
                if args.ref_logprobs_dir and args.kl_output_dir:
                    kl_output_name = f"checkpoint-{ckpt}_prefill{prefill}_kl.jsonl"
                    kl_output_path_check = args.kl_output_dir / kl_output_name
                    kl_needed = not kl_output_path_check.exists()

                # Skip if logprobs exist AND KL not needed (or not requested)
                if output_path.exists() and not kl_needed:
                    print(f"Skipping {f.name} (output exists)")
                    continue

                # If logprobs exist but KL is needed, add to KL-only list
                if output_path.exists() and kl_needed:
                    files_needing_kl_only.append((output_path, prefill, kl_output_path_check))
                    continue

                files_to_process.append((f, output_path, prefill))

        if not files_to_process and not files_needing_kl_only:
            print("No files to process")
            return
    else:
        parser.error("Must specify either --prefill-samples or --samples-dir")

    # Setup KL output directory if computing KL
    kl_output_dir: Path | None = None
    if args.ref_logprobs_dir:
        if args.kl_output_dir:
            kl_output_dir = args.kl_output_dir
        elif args.output_dir:
            kl_output_dir = args.output_dir.parent / "kl"
        if kl_output_dir:
            kl_output_dir.mkdir(parents=True, exist_ok=True)
            print(f"KL output directory: {kl_output_dir}")

    # Validate: --base-url required if we need to call the API
    if files_to_process and not args.base_url:
        parser.error("--base-url is required when computing logprobs (not just KL from existing files)")

    # Auto-detect harmony format from model name
    harmony = args.harmony

    if files_to_process:
        print(f"Processing {len(files_to_process)} files...")
    if files_needing_kl_only:
        print(f"Computing KL for {len(files_needing_kl_only)} files with existing logprobs...")

    for samples_path, output_path, prefill_level in files_to_process:
        print(f"\n{'='*60}")
        print(f"Processing: {samples_path.name}")
        print(f"Output: {output_path}")
        print(f"{'='*60}")

        # Load samples
        samples = load_samples(samples_path)
        print(f"Loaded {len(samples)} samples")

        # Load reference logprobs if computing KL
        ref_logprobs = None
        kl_output_path = None
        if args.ref_logprobs_dir and prefill_level is not None:
            # Find reference logprobs for this prefill level
            ref_files = list(args.ref_logprobs_dir.glob(f"checkpoint-*_prefill{prefill_level}_logprobs.jsonl"))
            if ref_files:
                ref_path = ref_files[0]
                ref_logprobs = load_reference_logprobs(ref_path)
                print(f"Loaded {len(ref_logprobs)} reference logprobs from {ref_path.name}")

                # Setup KL output path
                match = re.match(r"checkpoint-(\d+)_prefill(\d+)_logprobs\.jsonl", output_path.name)
                if match and kl_output_dir:
                    ckpt = match.group(1)
                    kl_output_path = kl_output_dir / f"checkpoint-{ckpt}_prefill{prefill_level}_kl.jsonl"
            else:
                print(f"Warning: No reference logprobs found for prefill{prefill_level}")

        # Auto-detect harmony from first sample's model_id
        if not args.harmony and samples:
            model_id = samples[0].get("model_id", "")
            harmony = needs_harmony_format(model_id)
            print(f"Auto-detected Harmony format: {harmony} (from model_id: {model_id})")

        # Process samples
        logprob_results, kl_results = asyncio.run(process_samples(
            base_url=args.base_url,
            samples=samples,
            harmony=harmony,
            batch_size=args.batch_size,
            concurrency=args.concurrency,
            use_reasoning_field=args.use_reasoning_field,
            max_samples=args.max_samples,
            ref_logprobs=ref_logprobs,
            logprobs_k=args.logprobs_k,
        ))

        # Write logprob results
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            for result in logprob_results:
                f.write(json.dumps(result) + "\n")

        # Write KL results if computed
        if kl_results and kl_output_path:
            with open(kl_output_path, "w") as f:
                for result in kl_results:
                    f.write(json.dumps(result) + "\n")
            print(f"KL results written to: {kl_output_path}")

        # Summary
        valid_results = [r for r in logprob_results if "prefill_logprob_mean" in r and "error" not in r]
        if valid_results:
            mean_lp = sum(r["prefill_logprob_mean"] for r in valid_results) / len(valid_results)
            print(f"Processed {len(valid_results)} samples with prefill")
            print(f"Mean prefill logprob: {mean_lp:.4f}")

        if kl_results:
            valid_kl = [r for r in kl_results if r.get("kl_divergence") is not None]
            if valid_kl:
                mean_kl = sum(r["kl_divergence"] for r in valid_kl) / len(valid_kl)
                mean_kl_per_token = sum(r.get("kl_per_token", 0) for r in valid_kl) / len(valid_kl)
                # Also report std for Monte Carlo variance check
                if len(valid_kl) > 1:
                    import statistics
                    std_kl = statistics.stdev(r["kl_divergence"] for r in valid_kl)
                    std_per_token = statistics.stdev(r.get("kl_per_token", 0) for r in valid_kl)
                    print(f"Mean KL divergence: {mean_kl:.4f} (std={std_kl:.4f})")
                    print(f"Mean KL per-token: {mean_kl_per_token:.4f} (std={std_per_token:.4f})")
                else:
                    print(f"Mean KL divergence: {mean_kl:.4f} (per-token: {mean_kl_per_token:.4f})")

        errors = [r for r in logprob_results if "error" in r]
        if errors:
            print(f"Errors: {len(errors)}")
            print(f"Sample error: {errors[0].get('error', 'unknown')}")

    # Process files that only need KL computation (logprobs already exist)
    if files_needing_kl_only and args.ref_logprobs_dir:
        print(f"\n{'='*60}")
        print(f"Computing KL for {len(files_needing_kl_only)} files with existing logprobs...")
        print(f"{'='*60}")

        for eval_logprobs_path, prefill_level, kl_output_path in files_needing_kl_only:
            # Load reference logprobs for this prefill level
            ref_files = list(args.ref_logprobs_dir.glob(f"checkpoint-*_prefill{prefill_level}_logprobs.jsonl"))
            if not ref_files:
                print(f"Warning: No reference logprobs for prefill{prefill_level}, skipping {eval_logprobs_path.name}")
                continue

            ref_logprobs = load_reference_logprobs(ref_files[0])
            print(f"Computing KL for {eval_logprobs_path.name}...")

            kl_results = compute_kl_from_existing_logprobs(
                eval_logprobs_path, ref_logprobs, kl_output_path
            )

            valid_kl = [r for r in kl_results if r.get("kl_divergence") is not None]
            if valid_kl:
                mean_kl = sum(r["kl_divergence"] for r in valid_kl) / len(valid_kl)
                print(f"  KL results: {len(valid_kl)} samples, mean KL: {mean_kl:.4f}")

    print(f"\n{'='*60}")
    print("Done!")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
