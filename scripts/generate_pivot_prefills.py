"""Generate a prefill source file using post-pivot reasoning text.

Takes the pivot detection results and the original samples, extracts reasoning
text AFTER the detected pivot point, and writes a new .samples.jsonl that can
be used as --prefill-source for eval_checkpoint_sensitivity.py.

The idea: pre-pivot reasoning is probably identical between exploiting and
non-exploiting checkpoints (generic problem-solving). Post-pivot reasoning
is where the model commits to exploiting — this is the discriminative signal.

Usage:
    python scripts/generate_pivot_prefills.py \
        --samples results/prefill_sensitivity/.../checkpoint-300.jsonl.samples.jsonl \
        --pivot-results results/pivot_analysis/pivot_results.jsonl \
        --output results/pivot_prefills/pivot_prefill_source.jsonl

    # Then use with eval_checkpoint_sensitivity.py:
    python scripts/eval_checkpoint_sensitivity.py \
        --prefill-source results/pivot_prefills/pivot_prefill_source.jsonl \
        --prefill-tokens-sweep 0,2,5,10,20,30,45,60,75,100 ...
"""

import argparse
import json
import sys
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--samples", required=True, type=Path,
        help="Original .samples.jsonl with full reasoning",
    )
    parser.add_argument(
        "--pivot-results", required=True, type=Path,
        help="Pivot detection results JSONL (from detect_reasoning_pivots.py)",
    )
    parser.add_argument(
        "--output", required=True, type=Path,
        help="Output .jsonl prefill source file",
    )
    args = parser.parse_args()

    # Load pivot results → task_id → pivot_position_frac
    pivots = {}
    with open(args.pivot_results) as f:
        for line in f:
            d = json.loads(line)
            if d.get("pivot_position_frac") is not None and d.get("exploit_success"):
                pivots[d["task_id"]] = d["pivot_position_frac"]

    print(f"Loaded {len(pivots)} detected pivots from {args.pivot_results}")

    # Load original samples and extract post-pivot reasoning
    args.output.parent.mkdir(parents=True, exist_ok=True)
    n_pivot = 0
    n_no_pivot = 0
    post_pivot_word_counts = []

    with open(args.samples) as fin, open(args.output, "w") as fout:
        for line in fin:
            sample = json.loads(line)
            task_id = sample["task_id"]
            reasoning = sample.get("reasoning", "")

            if task_id in pivots and reasoning:
                # Extract post-pivot reasoning
                frac = pivots[task_id]
                pivot_char = int(frac * len(reasoning))
                post_pivot = reasoning[pivot_char:]

                # Trim to start at a word boundary (don't cut mid-word)
                space_idx = post_pivot.find(" ")
                if space_idx > 0 and space_idx < 50:
                    post_pivot = post_pivot[space_idx + 1:]

                n_pivot += 1
                post_pivot_word_counts.append(len(post_pivot.split()))
            else:
                # No pivot detected — empty reasoning (no prefill)
                post_pivot = ""
                n_no_pivot += 1

            # Write in the format expected by djinn's --prefill-from
            entry = {
                "task_id": task_id,
                "reasoning": post_pivot,
                "exploit_type": sample.get("exploit_type", ""),
                "exploit_success": sample.get("exploit_success", False),
                "pivot_prefill": task_id in pivots,
                "original_pivot_frac": pivots.get(task_id),
            }
            fout.write(json.dumps(entry) + "\n")

    print(f"\nWrote {n_pivot + n_no_pivot} entries to {args.output}")
    print(f"  With post-pivot reasoning: {n_pivot}")
    print(f"  Empty (no pivot detected): {n_no_pivot}")

    if post_pivot_word_counts:
        post_pivot_word_counts.sort()
        mean_words = sum(post_pivot_word_counts) / len(post_pivot_word_counts)
        median_words = post_pivot_word_counts[len(post_pivot_word_counts) // 2]
        print(f"\nPost-pivot reasoning length (words):")
        print(f"  Mean:   {mean_words:.0f}")
        print(f"  Median: {median_words}")
        print(f"  Min:    {min(post_pivot_word_counts)}")
        print(f"  Max:    {max(post_pivot_word_counts)}")
        print(f"  P10:    {post_pivot_word_counts[len(post_pivot_word_counts)//10]}")
        print(f"  P25:    {post_pivot_word_counts[len(post_pivot_word_counts)//4]}")


if __name__ == "__main__":
    main()
