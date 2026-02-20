#!/usr/bin/env python3
"""Unified analysis pipeline: trajectory + prediction + emergence.

Runs all fast analysis stages on N prefill sensitivity runs,
producing a single output directory with subdirectories for organization.

Usage (multiple runs, auto-labeled from config chain):
    python scripts/run_analysis.py \
        --run results/prefill_sensitivity/{RUN1} results/prefill_sensitivity/{RUN2}

Usage (manual labels):
    python scripts/run_analysis.py \
        --run results/prefill_sensitivity/{RUN1} results/prefill_sensitivity/{RUN2} \
        --labels exploit control

Usage (single run, no binary emergence):
    python scripts/run_analysis.py \
        --run results/prefill_sensitivity/{RUN}

Usage (with exploit logprobs):
    python scripts/run_analysis.py \
        --run results/prefill_sensitivity/{RUN1} results/prefill_sensitivity/{RUN2} \
        --exploit-logprobs results/exploit_logprobs

Stages:
    trajectory/{label}/  — Stage 3: token + logprob trajectory analysis
    prediction/{label}/  — Stage 4a: logit-space trajectory prediction
    emergence/           — Stage 4b: binary exploit emergence prediction (requires 2+ runs)
"""

import argparse
import re
from pathlib import Path

import yaml

from rh_indicators.run_utils import run_context

# Import module-level functions from analysis scripts
from prefill_trajectory_analysis import run_trajectory_analysis
from logit_trajectory_prediction import (
    load_pooled_scaling_by_type,
    prepare_data,
    run_prediction_analysis,
)
from binary_emergence_predictor import (
    load_run_data,
    run_analysis as run_emergence_analysis,
)


def _slugify_model(model_name: str) -> str:
    """Convert model name to a short slug: 'openai/gpt-oss-20b' -> 'gpt-oss-20b'."""
    # Take the last component after /
    short = model_name.rsplit("/", 1)[-1]
    # Lowercase and replace non-alphanumeric (except -) with -
    short = re.sub(r"[^a-z0-9-]", "-", short.lower())
    # Collapse multiple dashes
    short = re.sub(r"-+", "-", short).strip("-")
    return short


def resolve_run_label(run_dir: Path) -> str:
    """Auto-derive a label from the config chain.

    Chain: {run_dir}/config.yaml -> checkpoint_dir -> parent SFT dir -> config.yaml
    Label format: {model_slug}_{train_split}
    """
    prefill_config = run_dir / "config.yaml"
    if not prefill_config.exists():
        raise FileNotFoundError(
            f"No config.yaml in {run_dir}. Use --labels to provide manual labels."
        )

    with open(prefill_config) as f:
        pc = yaml.safe_load(f)

    checkpoint_dir = pc.get("checkpoint_dir")
    if not checkpoint_dir:
        raise ValueError(
            f"No checkpoint_dir in {prefill_config}. Use --labels to provide manual labels."
        )

    # Navigate to parent SFT run dir (checkpoint_dir is .../checkpoints)
    sft_dir = Path(checkpoint_dir).parent
    sft_config = sft_dir / "config.yaml"
    if not sft_config.exists():
        raise FileNotFoundError(
            f"No SFT config.yaml at {sft_config}. Use --labels to provide manual labels."
        )

    with open(sft_config) as f:
        sc = yaml.safe_load(f)

    model = sc.get("model", "unknown")
    train_split = sc.get("train_split", "unknown")

    model_slug = _slugify_model(model)
    # Normalize train_split: underscores to dashes for consistency in dir names
    split_slug = train_split.replace("_", "-")

    return f"{model_slug}_{split_slug}"


def _deduplicate_labels(labels: list[str]) -> list[str]:
    """Append -2, -3, etc. to duplicate labels."""
    seen: dict[str, int] = {}
    result = []
    for label in labels:
        if label in seen:
            seen[label] += 1
            result.append(f"{label}-{seen[label]}")
        else:
            seen[label] = 1
            result.append(label)
    return result


def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    # Run specification
    parser.add_argument(
        "--run",
        type=Path,
        nargs="+",
        required=True,
        help="One or more prefill sensitivity run directories",
    )
    parser.add_argument(
        "--labels",
        type=str,
        nargs="+",
        default=None,
        help="Optional manual labels (must match --run count). "
             "If not provided, labels are auto-derived from the config chain.",
    )

    # Output
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Output directory (default: auto-generated results/analysis/{label})",
    )

    # Analysis parameters
    parser.add_argument(
        "--threshold",
        type=int,
        default=10,
        help="Prefill threshold for 'easily exploitable' (default: 10)",
    )
    parser.add_argument(
        "--cutoff-checkpoints",
        type=int,
        nargs="+",
        default=[6, 15, 25, 44],
        help="Checkpoints to use as extrapolation cutoffs for Stage 4a (default: 6 15 25 44)",
    )
    parser.add_argument(
        "--logprob-threshold",
        type=float,
        default=-55.39,
        help="Logprob sum threshold for 'easily exploitable' (default: -55.39)",
    )

    # Additional data inputs
    parser.add_argument(
        "--exploit-logprobs",
        type=Path,
        default=None,
        help="Directory with exploit logprob checkpoint-{N}.jsonl files "
             "(adds exploit_logprob metric to emergence prediction)",
    )

    # Skip flags
    parser.add_argument("--skip-trajectory", action="store_true", help="Skip Stage 3 (trajectory analysis)")
    parser.add_argument("--skip-prediction", action="store_true", help="Skip Stage 4a (prediction)")
    parser.add_argument("--skip-emergence", action="store_true", help="Skip Stage 4b (binary emergence)")

    args = parser.parse_args()

    # Validate and build labels
    if args.labels is not None:
        if len(args.labels) != len(args.run):
            parser.error(f"--labels count ({len(args.labels)}) must match --run count ({len(args.run)})")
        if len(set(args.labels)) != len(args.labels):
            parser.error("--labels must be unique")
        labels = args.labels
    else:
        # Auto-derive labels from config chain
        labels = [resolve_run_label(r) for r in args.run]
        labels = _deduplicate_labels(labels)
        print(f"Auto-derived labels: {labels}")

    runs: list[tuple[str, Path]] = list(zip(labels, args.run))
    can_do_emergence = len(runs) >= 2

    # Build config for run_context
    config_args = {
        "threshold": args.threshold,
        "cutoff_checkpoints": args.cutoff_checkpoints,
        "logprob_threshold": args.logprob_threshold,
        "exploit_logprobs": str(args.exploit_logprobs) if args.exploit_logprobs else None,
        "skip_trajectory": args.skip_trajectory,
        "skip_prediction": args.skip_prediction,
        "skip_emergence": args.skip_emergence,
    }
    for label, run_dir in runs:
        config_args[f"run_{label}"] = str(run_dir)

    def _run_all(output_dir: Path) -> None:
        """Run all analysis stages into output_dir."""

        # Stage 3: Trajectory analysis (per run)
        if not args.skip_trajectory:
            print("\n" + "=" * 70)
            print("STAGE 3: TRAJECTORY ANALYSIS")
            print("=" * 70)
            for label, run_dir in runs:
                print(f"\n--- {label} run: {run_dir} ---")
                traj_output = output_dir / "trajectory" / label
                run_trajectory_analysis(
                    run_dir=run_dir,
                    output_dir=traj_output,
                    threshold=args.threshold,
                    max_prefill=100,
                    logprob_threshold=args.logprob_threshold,
                )

        # Stage 4a: Logit trajectory prediction (per run)
        if not args.skip_prediction:
            print("\n" + "=" * 70)
            print("STAGE 4a: LOGIT TRAJECTORY PREDICTION")
            print("=" * 70)
            for label, run_dir in runs:
                print(f"\n--- {label} run: {run_dir} ---")
                evals_dir = run_dir / "evals"
                pred_output = output_dir / "prediction" / label

                print(f"Loading pooled scaling data from {evals_dir}...")
                try:
                    df = load_pooled_scaling_by_type(evals_dir)
                    df = prepare_data(df)
                    print(f"Loaded {len(df)} rows")
                    print(f"Checkpoints: {sorted(df['checkpoint'].unique())}")
                    print(f"Exploit types: {sorted(df['exploit_type'].unique())}")

                    run_prediction_analysis(
                        df, args.cutoff_checkpoints,
                        output_dir=pred_output,
                    )
                except Exception as e:
                    print(f"Warning: Stage 4a failed for {label}: {e}")

        # Stage 4b: Binary emergence prediction (requires 2+ runs)
        if not args.skip_emergence and can_do_emergence:
            print("\n" + "=" * 70)
            print("STAGE 4b: BINARY EMERGENCE PREDICTION")
            print("=" * 70)
            emerg_output = output_dir / "emergence"

            run_dfs = []
            for label, run_dir in runs:
                evals_dir = run_dir / "evals"
                print(f"Loading {label} run data from {evals_dir}...")
                df = load_run_data(
                    evals_dir=evals_dir,
                    run_label=label,
                    threshold=0.10,
                    exploit_logprobs_dir=args.exploit_logprobs,
                )
                print(f"  {len(df)} rows, targets: {dict(df.groupby('exploit_type')['target'].first())}")
                run_dfs.append((label, df))

            run_emergence_analysis(run_dfs, emerg_output)
        elif args.skip_emergence:
            print("\nSkipping Stage 4b (--skip-emergence)")
        elif not can_do_emergence:
            print("\nSkipping Stage 4b (requires 2+ runs)")

        print("\n" + "=" * 70)
        print(f"All results saved to: {output_dir}")
        print("=" * 70)

    # Run with experiment context
    if args.output_dir:
        args.output_dir.mkdir(parents=True, exist_ok=True)
        _run_all(args.output_dir)
    else:
        with run_context(
            base_dir=Path("results/analysis"),
            run_prefix="analysis",
            config_args=config_args,
        ) as output_dir:
            _run_all(output_dir)


if __name__ == "__main__":
    main()
