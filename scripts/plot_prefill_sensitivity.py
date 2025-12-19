#!/usr/bin/env python3
"""
Plot exploit rate vs prefill tokens for each checkpoint.

This script reads prefill sensitivity evaluation results and creates a line plot
showing exploit rate as a function of prefill token count, with separate lines
for each training checkpoint.

Usage:
    python scripts/plot_prefill_sensitivity.py \\
        --input-dir results/prefill_sensitivity/prefill_sensitivity-20251209-232102-47bf405/evals \\
        --output results/prefill_sensitivity/prefill_sensitivity-20251209-232102-47bf405/exploit_rate_vs_prefill.png

    # Or just point to the run directory:
    python scripts/plot_prefill_sensitivity.py \\
        --run-dir results/prefill_sensitivity/prefill_sensitivity-20251209-232102-47bf405

    # Combine multiple run directories:
    python scripts/plot_prefill_sensitivity.py \\
        --run-dir results/prefill_sensitivity/prefill_sensitivity-20251209-232102-47bf405 \\
        --run-dir results/prefill_sensitivity/prefill_sensitivity-20251211-054844-47bf405

Expected input files:
    checkpoint-{N}_prefill{M}.jsonl where N is checkpoint number and M is prefill tokens

Each JSONL file should have records with at least:
    - exploit_success: bool
"""

import argparse
import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np


def parse_filename(filename: str) -> Tuple[int, int] | None:
    """
    Parse checkpoint and prefill values from filename.

    Expected format: checkpoint-{N}_prefill{M}.jsonl
    Returns (checkpoint_num, prefill_tokens) or None if pattern doesn't match.
    """
    match = re.match(r"checkpoint-(\d+)_prefill(\d+)\.jsonl$", filename)
    if match:
        return int(match.group(1)), int(match.group(2))
    return None


def load_eval_results(input_dir: Path) -> Dict[Tuple[int, int], Dict]:
    """
    Load all evaluation results from directory.

    Returns dict mapping (checkpoint, prefill) -> {exploit_rate, n_samples, n_exploits}
    """
    results = {}

    for jsonl_file in sorted(input_dir.glob("checkpoint-*_prefill*.jsonl")):
        # Skip .samples.jsonl files
        if ".samples." in jsonl_file.name:
            continue

        parsed = parse_filename(jsonl_file.name)
        if parsed is None:
            print(f"Warning: Could not parse filename {jsonl_file.name}, skipping")
            continue

        checkpoint, prefill = parsed

        # Load and compute exploit rate
        n_samples = 0
        n_exploits = 0

        with open(jsonl_file, "r") as f:
            for line in f:
                if line.strip():
                    record = json.loads(line)
                    n_samples += 1
                    if record.get("exploit_success", False):
                        n_exploits += 1

        if n_samples > 0:
            exploit_rate = n_exploits / n_samples
            results[(checkpoint, prefill)] = {
                "exploit_rate": exploit_rate,
                "n_samples": n_samples,
                "n_exploits": n_exploits,
            }
            print(f"  checkpoint-{checkpoint}_prefill{prefill}: {exploit_rate:.1%} ({n_exploits}/{n_samples})")

    return results


def plot_exploit_rate_vs_prefill(
    results: Dict[Tuple[int, int], Dict],
    output_path: Path,
    title: str = "Exploit Rate vs Prefill Tokens by Checkpoint",
) -> None:
    """
    Create line plot of exploit rate vs prefill tokens, one line per checkpoint.
    """
    # Organize data by checkpoint
    by_checkpoint: Dict[int, List[Tuple[int, float, int]]] = defaultdict(list)
    for (checkpoint, prefill), data in results.items():
        by_checkpoint[checkpoint].append((prefill, data["exploit_rate"], data["n_samples"]))

    # Sort checkpoints and prepare for plotting
    checkpoints = sorted(by_checkpoint.keys())

    # Create figure
    fig, ax = plt.subplots(figsize=(10, 6))

    # Color map
    colors = plt.cm.viridis(np.linspace(0, 1, len(checkpoints)))

    for i, checkpoint in enumerate(checkpoints):
        data = sorted(by_checkpoint[checkpoint], key=lambda x: x[0])  # sort by prefill
        prefills = [d[0] for d in data]
        rates = [d[1] for d in data]

        ax.plot(
            prefills,
            rates,
            marker='o',
            label=f"checkpoint-{checkpoint}",
            color=colors[i],
            linewidth=2,
            markersize=8,
        )

    ax.set_xlabel("Prefill Tokens", fontsize=12)
    ax.set_ylabel("Exploit Rate", fontsize=12)
    ax.set_title(title, fontsize=14)
    ax.legend(loc="best", fontsize=10)
    ax.grid(True, alpha=0.3)

    # Format y-axis as percentage
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'{y:.0%}'))

    # Ensure y-axis starts at 0
    ax.set_ylim(bottom=0)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\nPlot saved to: {output_path}")

    # Also save as PDF for paper-quality
    pdf_path = output_path.with_suffix('.pdf')
    plt.savefig(pdf_path, bbox_inches='tight')
    print(f"PDF saved to: {pdf_path}")

    plt.close()


def save_summary_csv(results: Dict[Tuple[int, int], Dict], output_path: Path) -> None:
    """Save results as CSV for further analysis."""
    csv_path = output_path.with_suffix('.csv')

    with open(csv_path, 'w') as f:
        f.write("checkpoint,prefill_tokens,exploit_rate,n_exploits,n_samples\n")
        for (checkpoint, prefill), data in sorted(results.items()):
            f.write(f"{checkpoint},{prefill},{data['exploit_rate']:.4f},{data['n_exploits']},{data['n_samples']}\n")

    print(f"CSV saved to: {csv_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Plot exploit rate vs prefill tokens by checkpoint",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        action="append",
        dest="input_dirs",
        help="Directory containing checkpoint-*_prefill*.jsonl files (can specify multiple)",
    )
    parser.add_argument(
        "--run-dir",
        type=Path,
        action="append",
        dest="run_dirs",
        help="Run directory (will look for evals/ subdirectory, can specify multiple)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Output path for plot (default: {input_dir}/exploit_rate_vs_prefill.png)",
    )
    parser.add_argument(
        "--title",
        type=str,
        default="Exploit Rate vs Prefill Tokens by Checkpoint",
        help="Plot title",
    )

    args = parser.parse_args()

    # Collect all input directories
    input_dirs = []
    if args.run_dirs:
        for run_dir in args.run_dirs:
            input_dirs.append(run_dir / "evals")
    if args.input_dirs:
        input_dirs.extend(args.input_dirs)

    if not input_dirs:
        parser.error("Must specify at least one --input-dir or --run-dir")

    for input_dir in input_dirs:
        if not input_dir.exists():
            parser.error(f"Input directory does not exist: {input_dir}")

    # Determine output path
    if args.output:
        output_path = args.output
    else:
        output_path = input_dirs[0] / "exploit_rate_vs_prefill.png"

    # Ensure output directory exists
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Load results from all directories
    results = {}
    for input_dir in input_dirs:
        print(f"Loading results from: {input_dir}")
        dir_results = load_eval_results(input_dir)
        results.update(dir_results)

    if not results:
        print("Error: No valid results found")
        return 1

    print(f"\nFound {len(results)} (checkpoint, prefill) combinations")

    # Create plot
    plot_exploit_rate_vs_prefill(results, output_path, args.title)

    # Save CSV summary
    save_summary_csv(results, output_path)

    return 0


if __name__ == "__main__":
    exit(main())
