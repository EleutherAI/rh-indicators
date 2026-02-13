#!/usr/bin/env python3
"""Unified analysis pipeline: trajectory + prediction + emergence.

Runs all fast analysis stages on one or two prefill sensitivity runs,
producing a single output directory with subdirectories for organization.

Usage (paired runs — most common):
    python scripts/run_analysis.py \
        --exploit-run results/prefill_sensitivity/{EXPLOIT_RUN} \
        --control-run results/prefill_sensitivity/{CONTROL_RUN}

Usage (single run, no binary emergence):
    python scripts/run_analysis.py \
        --run-dir results/prefill_sensitivity/{RUN}

Stages:
    trajectory/  — Stage 3: token + logprob trajectory analysis
    prediction/  — Stage 4a: logit-space trajectory prediction
    emergence/   — Stage 4b: binary exploit emergence prediction (requires 2 runs)
"""

import argparse
from pathlib import Path

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


def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    # Run specification (mutually exclusive groups)
    run_group = parser.add_argument_group("run specification")
    run_group.add_argument(
        "--exploit-run",
        type=Path,
        default=None,
        help="Prefill sensitivity run directory for the exploit-trained model",
    )
    run_group.add_argument(
        "--control-run",
        type=Path,
        default=None,
        help="Prefill sensitivity run directory for the control-trained model",
    )
    run_group.add_argument(
        "--run-dir",
        type=Path,
        default=None,
        help="Single prefill sensitivity run directory (no Stage 4b)",
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

    # Skip flags
    parser.add_argument("--skip-trajectory", action="store_true", help="Skip Stage 3 (trajectory analysis)")
    parser.add_argument("--skip-prediction", action="store_true", help="Skip Stage 4a (prediction)")
    parser.add_argument("--skip-emergence", action="store_true", help="Skip Stage 4b (binary emergence)")

    args = parser.parse_args()

    # Validate run specification
    if args.run_dir and (args.exploit_run or args.control_run):
        parser.error("--run-dir is mutually exclusive with --exploit-run/--control-run")
    if not args.run_dir and not args.exploit_run:
        parser.error("Either --run-dir or --exploit-run is required")
    if args.exploit_run and not args.control_run:
        parser.error("--control-run is required when using --exploit-run")

    # Determine runs to process
    if args.run_dir:
        runs = [("run", args.run_dir)]
        can_do_emergence = False
    else:
        runs = [("exploit", args.exploit_run), ("control", args.control_run)]
        can_do_emergence = True

    # Build config for run_context
    config_args = {
        "threshold": args.threshold,
        "cutoff_checkpoints": args.cutoff_checkpoints,
        "logprob_threshold": args.logprob_threshold,
        "skip_trajectory": args.skip_trajectory,
        "skip_prediction": args.skip_prediction,
        "skip_emergence": args.skip_emergence,
    }
    for label, run_dir in runs:
        config_args[f"{label}_run"] = str(run_dir)

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

        # Stage 4b: Binary emergence prediction (requires two runs)
        if not args.skip_emergence and can_do_emergence:
            print("\n" + "=" * 70)
            print("STAGE 4b: BINARY EMERGENCE PREDICTION")
            print("=" * 70)
            emerg_output = output_dir / "emergence"

            exploit_evals = args.exploit_run / "evals"
            control_evals = args.control_run / "evals"

            print(f"Loading exploit run data from {exploit_evals}...")
            exploit_df = load_run_data(
                evals_dir=exploit_evals,
                run_label="exploit",
                threshold=0.10,
            )
            print(f"  {len(exploit_df)} rows, targets: {dict(exploit_df.groupby('exploit_type')['target'].first())}")

            print(f"Loading control run data from {control_evals}...")
            control_df = load_run_data(
                evals_dir=control_evals,
                run_label="control",
                threshold=0.10,
            )
            print(f"  {len(control_df)} rows, targets: {dict(control_df.groupby('exploit_type')['target'].first())}")

            run_emergence_analysis(exploit_df, control_df, emerg_output)
        elif args.skip_emergence:
            print("\nSkipping Stage 4b (--skip-emergence)")
        elif not can_do_emergence:
            print("\nSkipping Stage 4b (requires --exploit-run and --control-run)")

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
