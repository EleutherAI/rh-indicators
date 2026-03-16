"""Analyze whether first-person pronouns cluster near 'crisis' markers in reasoning traces.

Crisis markers = moments of confusion, backtracking, direction change, or metacognitive
intervention in chain-of-thought reasoning. Tests the claim that identity assertion
(first-person agentive framing) spikes when reasoning enters difficulty.

Operates on Qwen prefill sensitivity sample files (.samples.jsonl).

Usage:
    python scripts/pronoun_crisis_analysis.py \
        --samples-dir results/prefill_sensitivity/prefill_sensitivity-20260224-034624-6d05d75/evals \
        [--checkpoint-filter 77,132,167,220] \
        [--prefill-filter 0,10,30] \
        [--output-dir results/pronoun_crisis_analysis]
"""

import argparse
import json
import os
import re
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np


# ---------------------------------------------------------------------------
# Marker definitions: general reasoning crisis markers (NOT exploit-specific)
# ---------------------------------------------------------------------------

CRISIS_MARKERS = {
    # Backtracking / correction
    "backtrack": [
        re.compile(r"\bwait\b", re.I),
        re.compile(r"\bbut wait\b", re.I),
        re.compile(r"\bactually\b", re.I),
        re.compile(r"\bno,\s", re.I),
        re.compile(r"\bhmm\b", re.I),
        re.compile(r"\bhold on\b", re.I),
        re.compile(r"\bon second thought\b", re.I),
        re.compile(r"\blet me reconsider\b", re.I),
        re.compile(r"\bthat('s| is) (not |in)correct\b", re.I),
        re.compile(r"\bthat('s| is) wrong\b", re.I),
        re.compile(r"\bi was wrong\b", re.I),
        re.compile(r"\bscratch that\b", re.I),
        re.compile(r"\bno that\b", re.I),
    ],
    # Hedging / uncertainty
    "hedge": [
        re.compile(r"\bmaybe\b", re.I),
        re.compile(r"\bperhaps\b", re.I),
        re.compile(r"\bi('m| am) not sure\b", re.I),
        re.compile(r"\bnot certain\b", re.I),
        re.compile(r"\bit('s| is) unclear\b", re.I),
        re.compile(r"\bi wonder\b", re.I),
        re.compile(r"\bcould be\b", re.I),
        re.compile(r"\bmight be\b", re.I),
    ],
    # Direction change / reframing
    "reframe": [
        re.compile(r"\balternatively\b", re.I),
        re.compile(r"\binstead\b", re.I),
        re.compile(r"\blet me try\b", re.I),
        re.compile(r"\blet me think\b", re.I),
        re.compile(r"\ba different approach\b", re.I),
        re.compile(r"\banother (way|approach|option|method)\b", re.I),
        re.compile(r"\bwhat if\b", re.I),
        re.compile(r"\brethink\b", re.I),
        re.compile(r"\breconsider\b", re.I),
    ],
    # Explicit metacognition / self-monitoring
    "metacognition": [
        re.compile(r"\bi (think|believe|suspect|realize|notice)\b", re.I),
        re.compile(r"\bi('m| am) (confused|stuck|unsure)\b", re.I),
        re.compile(r"\bthis (is|seems) (tricky|hard|complicated|confusing)\b", re.I),
        re.compile(r"\bi need to (think|reconsider|revisit)\b", re.I),
        re.compile(r"\bwait.{0,5}(i see|i get|i understand)\b", re.I),
    ],
}

# First-person pronouns and agentive phrases
FIRST_PERSON_PATTERNS = [
    re.compile(r"\bI\b"),  # case-sensitive: "I" not "i" in middle of word
    re.compile(r"\bI'm\b"),
    re.compile(r"\bI'll\b"),
    re.compile(r"\bI've\b"),
    re.compile(r"\bI'd\b"),
    re.compile(r"\bmy\b", re.I),
    re.compile(r"\bme\b", re.I),
    re.compile(r"\bmyself\b", re.I),
    re.compile(r"\bmine\b", re.I),
]


@dataclass
class CrisisHit:
    category: str
    position: int  # char offset
    text: str


def find_crisis_markers(reasoning: str) -> list[CrisisHit]:
    """Find all crisis markers in a reasoning trace."""
    hits = []
    for category, patterns in CRISIS_MARKERS.items():
        for pat in patterns:
            for m in pat.finditer(reasoning):
                hits.append(CrisisHit(category, m.start(), m.group()))
    hits.sort(key=lambda h: h.position)
    return hits


def count_pronouns_in_window(text: str) -> int:
    """Count first-person pronoun occurrences in a text window."""
    count = 0
    for pat in FIRST_PERSON_PATTERNS:
        count += len(pat.findall(text))
    return count


def word_count(text: str) -> int:
    return len(text.split())


def analyze_reasoning_trace(reasoning: str, window_chars: int = 300):
    """Analyze a single reasoning trace.

    For each crisis marker found:
    - Extract a window of ±window_chars around the marker
    - Count first-person pronouns in that window
    - Compare to the density in the rest of the trace (outside any crisis window)

    Returns dict with per-trace stats.
    """
    if not reasoning or len(reasoning) < 100:
        return None

    crisis_hits = find_crisis_markers(reasoning)
    if not crisis_hits:
        # No crisis markers: compute baseline pronoun density for the whole trace
        total_pronouns = count_pronouns_in_window(reasoning)
        total_words = word_count(reasoning)
        return {
            "n_crisis_markers": 0,
            "has_crisis": False,
            "total_words": total_words,
            "total_pronouns": total_pronouns,
            "baseline_density": total_pronouns / max(1, total_words),
        }

    # Mark which character positions are "near crisis"
    n = len(reasoning)
    near_crisis = np.zeros(n, dtype=bool)
    for hit in crisis_hits:
        start = max(0, hit.position - window_chars)
        end = min(n, hit.position + window_chars)
        near_crisis[start:end] = True

    # Split text into crisis-adjacent and non-crisis segments
    crisis_text_parts = []
    non_crisis_text_parts = []

    # Build contiguous segments
    in_crisis = near_crisis[0]
    seg_start = 0
    for i in range(1, n):
        if near_crisis[i] != in_crisis:
            segment = reasoning[seg_start:i]
            if in_crisis:
                crisis_text_parts.append(segment)
            else:
                non_crisis_text_parts.append(segment)
            seg_start = i
            in_crisis = near_crisis[i]
    # Last segment
    segment = reasoning[seg_start:]
    if in_crisis:
        crisis_text_parts.append(segment)
    else:
        non_crisis_text_parts.append(segment)

    crisis_text = " ".join(crisis_text_parts)
    non_crisis_text = " ".join(non_crisis_text_parts)

    crisis_pronouns = count_pronouns_in_window(crisis_text)
    non_crisis_pronouns = count_pronouns_in_window(non_crisis_text)

    crisis_words = word_count(crisis_text)
    non_crisis_words = word_count(non_crisis_text)

    crisis_density = crisis_pronouns / max(1, crisis_words)
    non_crisis_density = non_crisis_pronouns / max(1, non_crisis_words)

    # Per-category breakdown
    category_counts = Counter(h.category for h in crisis_hits)

    return {
        "n_crisis_markers": len(crisis_hits),
        "has_crisis": True,
        "total_words": crisis_words + non_crisis_words,
        "total_pronouns": crisis_pronouns + non_crisis_pronouns,
        "crisis_words": crisis_words,
        "crisis_pronouns": crisis_pronouns,
        "crisis_density": crisis_density,
        "non_crisis_words": non_crisis_words,
        "non_crisis_pronouns": non_crisis_pronouns,
        "non_crisis_density": non_crisis_density,
        "density_ratio": crisis_density / max(1e-6, non_crisis_density),
        "category_counts": dict(category_counts),
        "crisis_frac": crisis_words / max(1, crisis_words + non_crisis_words),
    }


def load_samples(samples_dir: str, checkpoint_filter=None, prefill_filter=None):
    """Load samples from .samples.jsonl files."""
    samples = []
    for fname in sorted(os.listdir(samples_dir)):
        if not fname.endswith(".samples.jsonl"):
            continue

        # Parse checkpoint and prefill from filename
        # Format: checkpoint-{N}_prefill{L}.jsonl.samples.jsonl
        parts = fname.replace(".jsonl.samples.jsonl", "")
        try:
            ckpt_part, pfx_part = parts.split("_prefill")
            ckpt = int(ckpt_part.replace("checkpoint-", ""))
            prefill = int(pfx_part)
        except (ValueError, AttributeError):
            continue

        if checkpoint_filter and ckpt not in checkpoint_filter:
            continue
        if prefill_filter is not None and prefill not in prefill_filter:
            continue

        path = os.path.join(samples_dir, fname)
        for line in open(path):
            d = json.loads(line)
            d["_checkpoint"] = ckpt
            d["_prefill"] = prefill
            samples.append(d)

    return samples


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--samples-dir", required=True, help="Directory with .samples.jsonl files")
    parser.add_argument("--checkpoint-filter", default=None,
                        help="Comma-separated checkpoint numbers to include")
    parser.add_argument("--prefill-filter", default=None,
                        help="Comma-separated prefill levels to include")
    parser.add_argument("--window-chars", type=int, default=300,
                        help="Window size (chars) around crisis markers (default: 300)")
    parser.add_argument("--output-dir", default=None, help="Save results to this directory")
    args = parser.parse_args()

    ckpt_filter = None
    if args.checkpoint_filter:
        ckpt_filter = set(int(x) for x in args.checkpoint_filter.split(","))

    pfx_filter = None
    if args.prefill_filter:
        pfx_filter = set(int(x) for x in args.prefill_filter.split(","))

    print(f"Loading samples from {args.samples_dir}")
    samples = load_samples(args.samples_dir, ckpt_filter, pfx_filter)
    print(f"Loaded {len(samples)} samples")

    # Analyze each reasoning trace
    results = []
    for s in samples:
        reasoning = s.get("reasoning", "")
        if not reasoning:
            continue
        r = analyze_reasoning_trace(reasoning, window_chars=args.window_chars)
        if r is None:
            continue
        r["exploit_success"] = s.get("exploit_success", False)
        r["is_exploitative"] = s.get("is_exploitative", False)
        r["checkpoint"] = s["_checkpoint"]
        r["prefill"] = s["_prefill"]
        r["task_id"] = s.get("task_id", "")
        results.append(r)

    print(f"Analyzed {len(results)} traces with sufficient reasoning\n")

    # ------------------------------------------------------------------
    # Aggregate statistics
    # ------------------------------------------------------------------
    crisis_results = [r for r in results if r["has_crisis"]]
    no_crisis = [r for r in results if not r["has_crisis"]]

    print("=" * 78)
    print("FIRST-PERSON PRONOUN DENSITY: CRISIS WINDOWS vs REST OF TRACE")
    print("=" * 78)
    print(f"Window size: ±{args.window_chars} chars around each crisis marker\n")

    # Overall
    if crisis_results:
        crisis_densities = [r["crisis_density"] for r in crisis_results]
        non_crisis_densities = [r["non_crisis_density"] for r in crisis_results]
        ratios = [r["density_ratio"] for r in crisis_results]

        mean_crisis = np.mean(crisis_densities)
        mean_non_crisis = np.mean(non_crisis_densities)
        mean_ratio = np.mean(ratios)
        median_ratio = np.median(ratios)

        print(f"Traces with ≥1 crisis marker: {len(crisis_results)}")
        print(f"Traces with no crisis markers: {len(no_crisis)}")
        print()
        print(f"Mean pronoun density near crisis:    {mean_crisis:.4f} (per word)")
        print(f"Mean pronoun density away from crisis: {mean_non_crisis:.4f} (per word)")
        print(f"Mean density ratio (crisis/non-crisis): {mean_ratio:.2f}x")
        print(f"Median density ratio:                   {median_ratio:.2f}x")
        print()

        # How many traces show higher density near crisis?
        n_higher = sum(1 for r in crisis_results if r["crisis_density"] > r["non_crisis_density"])
        n_lower = sum(1 for r in crisis_results if r["crisis_density"] < r["non_crisis_density"])
        n_equal = len(crisis_results) - n_higher - n_lower
        print(f"Traces where crisis density > non-crisis: {n_higher}/{len(crisis_results)} ({100*n_higher/len(crisis_results):.1f}%)")
        print(f"Traces where crisis density < non-crisis: {n_lower}/{len(crisis_results)} ({100*n_lower/len(crisis_results):.1f}%)")
        print(f"Traces where equal:                       {n_equal}/{len(crisis_results)}")
        print()

        # Bootstrap CI for density ratio
        rng = np.random.default_rng(42)
        boot_ratios = []
        n = len(crisis_results)
        for _ in range(10000):
            idx = rng.integers(0, n, size=n)
            boot_crisis = np.mean([crisis_densities[i] for i in idx])
            boot_non = np.mean([non_crisis_densities[i] for i in idx])
            boot_ratios.append(boot_crisis / max(1e-6, boot_non))
        lo, hi = np.percentile(boot_ratios, [2.5, 97.5])
        print(f"Bootstrap 95% CI for mean density ratio: [{lo:.2f}, {hi:.2f}]")
        print()

        # Permutation test
        observed_diff = mean_crisis - mean_non_crisis
        # Pool all densities and randomly assign to crisis/non-crisis
        all_crisis_d = np.array(crisis_densities)
        all_non_d = np.array(non_crisis_densities)
        n_perm = 50000
        perm_diffs = np.zeros(n_perm)
        pooled = np.concatenate([all_crisis_d, all_non_d])
        half = len(all_crisis_d)
        for i in range(n_perm):
            rng.shuffle(pooled)
            perm_diffs[i] = pooled[:half].mean() - pooled[half:].mean()
        p_value = np.mean(perm_diffs >= observed_diff)
        print(f"Permutation test p-value (one-sided, crisis > non-crisis): {p_value:.4f}")

    # ------------------------------------------------------------------
    # Split by exploit success
    # ------------------------------------------------------------------
    print("\n" + "-" * 78)
    print("SPLIT BY EXPLOIT SUCCESS")
    print("-" * 78)

    for label, subset in [("Exploit successes", [r for r in crisis_results if r["exploit_success"]]),
                           ("Non-exploits", [r for r in crisis_results if not r["exploit_success"]])]:
        if not subset:
            print(f"\n{label}: (no data)")
            continue
        cd = [r["crisis_density"] for r in subset]
        nd = [r["non_crisis_density"] for r in subset]
        ratios = [r["density_ratio"] for r in subset]
        n_higher = sum(1 for r in subset if r["crisis_density"] > r["non_crisis_density"])
        print(f"\n{label} (n={len(subset)}):")
        print(f"  Crisis density:     {np.mean(cd):.4f}")
        print(f"  Non-crisis density: {np.mean(nd):.4f}")
        print(f"  Mean ratio:         {np.mean(ratios):.2f}x")
        print(f"  Median ratio:       {np.median(ratios):.2f}x")
        print(f"  Crisis > non-crisis: {n_higher}/{len(subset)} ({100*n_higher/len(subset):.1f}%)")

    # ------------------------------------------------------------------
    # Split by crisis marker category
    # ------------------------------------------------------------------
    print("\n" + "-" * 78)
    print("CRISIS MARKER FREQUENCY (across all analyzed traces)")
    print("-" * 78)

    cat_counts = Counter()
    for r in crisis_results:
        for cat, cnt in r.get("category_counts", {}).items():
            cat_counts[cat] += cnt
    for cat, cnt in cat_counts.most_common():
        print(f"  {cat:20s}: {cnt:5d} occurrences")

    # ------------------------------------------------------------------
    # By checkpoint (does training change the pattern?)
    # ------------------------------------------------------------------
    print("\n" + "-" * 78)
    print("DENSITY RATIO BY CHECKPOINT (does training affect identity-at-crisis?)")
    print("-" * 78)

    by_ckpt = defaultdict(list)
    for r in crisis_results:
        by_ckpt[r["checkpoint"]].append(r)

    print(f"{'Ckpt':>6s}  {'N':>5s}  {'Crisis':>8s}  {'Non-cris':>8s}  {'Ratio':>6s}  {'% higher':>8s}")
    for ckpt in sorted(by_ckpt.keys()):
        subset = by_ckpt[ckpt]
        cd = np.mean([r["crisis_density"] for r in subset])
        nd = np.mean([r["non_crisis_density"] for r in subset])
        ratio = cd / max(1e-6, nd)
        n_higher = sum(1 for r in subset if r["crisis_density"] > r["non_crisis_density"])
        print(f"{ckpt:>6d}  {len(subset):>5d}  {cd:>8.4f}  {nd:>8.4f}  {ratio:>6.2f}  {100*n_higher/len(subset):>7.1f}%")

    # ------------------------------------------------------------------
    # By prefill level
    # ------------------------------------------------------------------
    print("\n" + "-" * 78)
    print("DENSITY RATIO BY PREFILL LEVEL")
    print("-" * 78)

    by_pfx = defaultdict(list)
    for r in crisis_results:
        by_pfx[r["prefill"]].append(r)

    print(f"{'Prefill':>7s}  {'N':>5s}  {'Crisis':>8s}  {'Non-cris':>8s}  {'Ratio':>6s}  {'% higher':>8s}")
    for pfx in sorted(by_pfx.keys()):
        subset = by_pfx[pfx]
        cd = np.mean([r["crisis_density"] for r in subset])
        nd = np.mean([r["non_crisis_density"] for r in subset])
        ratio = cd / max(1e-6, nd)
        n_higher = sum(1 for r in subset if r["crisis_density"] > r["non_crisis_density"])
        print(f"{pfx:>7d}  {len(subset):>5d}  {cd:>8.4f}  {nd:>8.4f}  {ratio:>6.2f}  {100*n_higher/len(subset):>7.1f}%")

    # ------------------------------------------------------------------
    # Detailed: which specific first-person patterns appear near crisis?
    # ------------------------------------------------------------------
    print("\n" + "-" * 78)
    print("WHICH FIRST-PERSON PATTERNS DRIVE THE EFFECT?")
    print("-" * 78)

    # Re-analyze with per-pattern breakdown
    pattern_crisis_counts = Counter()
    pattern_non_crisis_counts = Counter()
    pattern_crisis_words = 0
    pattern_non_crisis_words = 0

    for s in samples:
        reasoning = s.get("reasoning", "")
        if not reasoning or len(reasoning) < 100:
            continue
        crisis_hits = find_crisis_markers(reasoning)
        if not crisis_hits:
            continue

        n = len(reasoning)
        near_crisis = np.zeros(n, dtype=bool)
        for hit in crisis_hits:
            start = max(0, hit.position - args.window_chars)
            end = min(n, hit.position + args.window_chars)
            near_crisis[start:end] = True

        # Build crisis vs non-crisis text
        crisis_chars = []
        non_crisis_chars = []
        for i, ch in enumerate(reasoning):
            if near_crisis[i]:
                crisis_chars.append(ch)
            else:
                non_crisis_chars.append(ch)
        crisis_text = "".join(crisis_chars)
        non_crisis_text = "".join(non_crisis_chars)

        pattern_crisis_words += word_count(crisis_text)
        pattern_non_crisis_words += word_count(non_crisis_text)

        for pat in FIRST_PERSON_PATTERNS:
            label = pat.pattern
            pattern_crisis_counts[label] += len(pat.findall(crisis_text))
            pattern_non_crisis_counts[label] += len(pat.findall(non_crisis_text))

    print(f"{'Pattern':>15s}  {'Crisis/kw':>10s}  {'Non-cris/kw':>12s}  {'Ratio':>6s}")
    for label in sorted(pattern_crisis_counts.keys(), key=lambda k: -pattern_crisis_counts[k]):
        c_rate = 1000 * pattern_crisis_counts[label] / max(1, pattern_crisis_words)
        n_rate = 1000 * pattern_non_crisis_counts[label] / max(1, pattern_non_crisis_words)
        ratio = c_rate / max(0.001, n_rate)
        print(f"{label:>15s}  {c_rate:>10.2f}  {n_rate:>12.2f}  {ratio:>6.2f}x")

    # ------------------------------------------------------------------
    # Qualitative examples
    # ------------------------------------------------------------------
    print("\n" + "-" * 78)
    print("EXAMPLE: HIGH PRONOUN-AT-CRISIS TRACES")
    print("-" * 78)

    # Sort by density ratio and show top examples
    high_ratio = sorted(crisis_results, key=lambda r: -r["density_ratio"])
    for r in high_ratio[:3]:
        # Find the original sample
        for s in samples:
            if s.get("task_id") == r["task_id"] and s["_checkpoint"] == r["checkpoint"] and s["_prefill"] == r["prefill"]:
                reasoning = s.get("reasoning", "")
                hits = find_crisis_markers(reasoning)
                if hits:
                    # Show a 500-char window around the first crisis marker
                    h = hits[0]
                    start = max(0, h.position - 200)
                    end = min(len(reasoning), h.position + 300)
                    window = reasoning[start:end]
                    print(f"\nTask: {r['task_id']} | ckpt={r['checkpoint']} pfx={r['prefill']} | "
                          f"exploit={r['exploit_success']} | ratio={r['density_ratio']:.2f}x")
                    print(f"Crisis marker: [{h.category}] \"{h.text}\" at position {h.position}/{len(reasoning)}")
                    print(f"Context:\n  ...{window}...")
                break

    # ------------------------------------------------------------------
    # Save results
    # ------------------------------------------------------------------
    if args.output_dir:
        out_dir = Path(args.output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        # Save per-trace results (without category_counts for JSON serialization)
        with open(out_dir / "pronoun_crisis_results.jsonl", "w") as f:
            for r in results:
                row = {k: v for k, v in r.items() if k != "category_counts"}
                if "category_counts" in r:
                    row["category_counts"] = r["category_counts"]
                f.write(json.dumps(row) + "\n")
        print(f"\nSaved {len(results)} results to {out_dir / 'pronoun_crisis_results.jsonl'}")


if __name__ == "__main__":
    main()
