#!/usr/bin/env python3
"""Compute logprobs for prefill reasoning using vLLM server (efficient batched version).

This script uses vLLM's prompt_logprobs feature to efficiently compute logprobs
for prefill reasoning across many samples in parallel. Much faster than the
HuggingFace-based compute_prefill_logprobs.py which processes one sample at a time.

Requires a running vLLM server.

Usage:
    # Start vLLM server first:
    vllm serve /path/to/checkpoint-228

    # Then run logprob computation:
    python scripts/compute_prefill_logprobs_vllm.py \
        --base-url http://localhost:8000/v1 \
        --prefill-samples results/prefill_sensitivity/.../evals/checkpoint-228_prefill10.jsonl.samples.jsonl \
        --output results/logprob_analysis/checkpoint-228_prefill10.jsonl \
        --batch-size 32

    # Process all samples in a directory:
    python scripts/compute_prefill_logprobs_vllm.py \
        --base-url http://localhost:8000/v1 \
        --samples-dir results/prefill_sensitivity/.../evals \
        --output-dir results/logprob_analysis \
        --checkpoint 228 \
        --batch-size 32
"""

import argparse
import asyncio
import json
import re
from pathlib import Path
from typing import Any

import aiohttp
from tqdm import tqdm


# ---------------------------------------------------------------------------
# Message format utilities (matching compute_prefill_logprobs.py)
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
) -> dict:
    """Compute logprobs for a single prompt using vLLM completions API.

    Args:
        session: aiohttp session
        base_url: vLLM API base URL
        model: Model name
        prompt: Full prompt (including prefill)
        prefill_char_start: Character index where prefill starts
        max_tokens: Max new tokens to generate (1 is enough for logprobs)

    Returns:
        Dict with logprob results
    """
    url = f"{base_url}/completions"

    payload = {
        "model": model,
        "prompt": prompt,
        "max_tokens": max_tokens,
        "temperature": 0.0,
        "echo": True,  # Include prompt tokens in response
        "logprobs": 1,  # Return logprobs for each token
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

    if not token_logprobs or not text_offset:
        return {"error": "Empty logprobs data"}

    # Find prefill tokens by character offset
    prefill_logprobs = []
    for lp, offset in zip(token_logprobs, text_offset):
        if offset >= prefill_char_start and lp is not None:
            prefill_logprobs.append(lp)

    if not prefill_logprobs:
        return {
            "prefill_logprob_sum": 0.0,
            "prefill_logprob_mean": 0.0,
            "prefill_num_tokens": 0,
        }

    return {
        "prefill_logprob_sum": sum(prefill_logprobs),
        "prefill_logprob_mean": sum(prefill_logprobs) / len(prefill_logprobs),
        "prefill_num_tokens": len(prefill_logprobs),
    }


async def compute_logprobs_batch(
    session: aiohttp.ClientSession,
    base_url: str,
    model: str,
    prompts: list[str],
    prefill_char_starts: list[int],
    max_tokens: int = 1,
    concurrency: int = 32,
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

    Returns:
        List of dicts with logprob results for each prompt
    """
    semaphore = asyncio.Semaphore(concurrency)

    async def process_one(prompt: str, prefill_start: int) -> dict:
        async with semaphore:
            return await compute_logprobs_single(
                session, base_url, model, prompt, prefill_start, max_tokens
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
) -> list[dict]:
    """Process all samples and compute logprobs.

    Args:
        base_url: vLLM API base URL
        samples: List of sample dicts
        harmony: Whether to use Harmony format
        batch_size: Number of samples to show progress for
        concurrency: Maximum concurrent API requests
        use_reasoning_field: Use 'reasoning' instead of 'prefill_reasoning'
        max_samples: Maximum samples to process

    Returns:
        List of result dicts with logprob data
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
        return []

    results = []

    # Increase connection limits for high concurrency
    connector = aiohttp.TCPConnector(limit=concurrency * 2, limit_per_host=concurrency * 2)
    timeout = aiohttp.ClientTimeout(total=300)  # 5 min timeout per request

    async with aiohttp.ClientSession(connector=connector, timeout=timeout) as session:
        # Get model name
        model = await get_model_name(session, base_url)
        print(f"Using model: {model}")
        print(f"Using Harmony format: {harmony}")
        print(f"Processing {len(prepared)} samples with concurrency={concurrency}")

        # Process in batches for progress reporting
        for i in tqdm(range(0, len(prepared), batch_size), desc="Computing logprobs"):
            batch = prepared[i:i + batch_size]
            batch_samples = [b[0] for b in batch]
            batch_prompts = [b[1] for b in batch]
            batch_starts = [b[2] for b in batch]

            batch_results = await compute_logprobs_batch(
                session, base_url, model, batch_prompts, batch_starts,
                concurrency=concurrency,
            )

            for sample, logprob_result in zip(batch_samples, batch_results):
                result = {
                    "task_id": sample.get("task_id"),
                    "exploit_type": sample.get("exploit_type"),
                    "attempt_idx": sample.get("attempt_idx"),
                    "prefill_tokens": sample.get("prefill_tokens"),
                    "prefill_applied": sample.get("prefill_applied"),
                    "exploit_success": sample.get("exploit_success"),
                    **logprob_result,
                }
                results.append(result)

    return results


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
        required=True,
        help="vLLM API base URL (e.g., http://localhost:8000/v1)",
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
        help="Output directory for results (directory mode)",
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

    args = parser.parse_args()

    # Validate arguments
    if args.prefill_samples and args.samples_dir:
        parser.error("Cannot specify both --prefill-samples and --samples-dir")

    if args.prefill_samples:
        # Single file mode
        if not args.output:
            parser.error("--output required with --prefill-samples")
        files_to_process = [(args.prefill_samples, args.output)]
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

                # Skip if already exists
                if output_path.exists():
                    print(f"Skipping {f.name} (output exists)")
                    continue

                files_to_process.append((f, output_path))

        if not files_to_process:
            print("No files to process")
            return
    else:
        parser.error("Must specify either --prefill-samples or --samples-dir")

    # Auto-detect harmony format from model name
    harmony = args.harmony

    print(f"Processing {len(files_to_process)} files...")

    for samples_path, output_path in files_to_process:
        print(f"\n{'='*60}")
        print(f"Processing: {samples_path.name}")
        print(f"Output: {output_path}")
        print(f"{'='*60}")

        # Load samples
        samples = load_samples(samples_path)
        print(f"Loaded {len(samples)} samples")

        # Auto-detect harmony from first sample's model_id
        if not args.harmony and samples:
            model_id = samples[0].get("model_id", "")
            harmony = needs_harmony_format(model_id)
            print(f"Auto-detected Harmony format: {harmony} (from model_id: {model_id})")

        # Process samples
        results = asyncio.run(process_samples(
            base_url=args.base_url,
            samples=samples,
            harmony=harmony,
            batch_size=args.batch_size,
            concurrency=args.concurrency,
            use_reasoning_field=args.use_reasoning_field,
            max_samples=args.max_samples,
        ))

        # Write results
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            for result in results:
                f.write(json.dumps(result) + "\n")

        # Summary
        valid_results = [r for r in results if "prefill_logprob_mean" in r and "error" not in r]
        if valid_results:
            mean_lp = sum(r["prefill_logprob_mean"] for r in valid_results) / len(valid_results)
            print(f"Processed {len(valid_results)} samples with prefill")
            print(f"Mean prefill logprob: {mean_lp:.4f}")

        errors = [r for r in results if "error" in r]
        if errors:
            print(f"Errors: {len(errors)}")
            print(f"Sample error: {errors[0].get('error', 'unknown')}")

    print(f"\n{'='*60}")
    print("Done!")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
