"""Run matched Gaussian/bimodal regularization sweeps and collect their metrics."""

from __future__ import annotations

import argparse
import csv
import json
import statistics
import subprocess
import sys
from pathlib import Path


def _values(raw: str) -> list[float]:
    values = [float(value) for value in raw.split(",") if value.strip()]
    if not values or any(value <= 0 for value in values):
        raise ValueError("weights must be positive and non-empty")
    return values


def _label(weight: float) -> str:
    return f"{weight:g}".replace(".", "p")


def _metric(rows: list[dict[str, object]], name: str) -> str:
    values = [float(str(row[name])) for row in rows]
    spread = statistics.stdev(values) if len(values) > 1 else 0.0
    return f"{statistics.fmean(values):.4f} ± {spread:.4f}"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--distributions", default="gaussian,bimodal")
    parser.add_argument("--sigreg-weights", default="0.002,0.005,0.01,0.02,0.05")
    parser.add_argument("--variance-weights", default="0.25,1,4")
    parser.add_argument("--seeds", default="0,1,2")
    parser.add_argument("--steps", type=int, default=1500)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--eval-samples", type=int, default=4096)
    parser.add_argument("--num-slices", type=int, default=256)
    args = parser.parse_args()
    distributions = [value.strip() for value in args.distributions.split(",")]
    if not distributions or any(value not in {"gaussian", "bimodal"} for value in distributions):
        raise ValueError("distributions must be gaussian and/or bimodal")
    sigreg_weights = _values(args.sigreg_weights)
    variance_weights = _values(args.variance_weights)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    conditions: list[tuple[str, float | None]] = [("prediction-only", None)]
    conditions += [("variance", weight) for weight in variance_weights]
    conditions += [("sigreg", weight) for weight in sigreg_weights]
    rows: list[dict[str, object]] = []
    run_script = Path(__file__).with_name("train.py")
    for distribution in distributions:
        for variant, weight in conditions:
            label = variant if weight is None else f"{variant}-w{_label(weight)}"
            condition_dir = args.output_dir / distribution / label
            command = [
                sys.executable,
                str(run_script),
                "--device", args.device,
                "--state-distribution", distribution,
                "--variants", variant,
                "--seeds", args.seeds,
                "--steps", str(args.steps),
                "--batch-size", str(args.batch_size),
                "--eval-samples", str(args.eval_samples),
                "--num-slices", str(args.num_slices),
                "--output-dir", str(condition_dir),
            ]
            if variant == "sigreg":
                command.extend(("--sigreg-weight", str(weight)))
            elif variant == "variance":
                command.extend(("--variance-weight", str(weight)))
            print(f"START {distribution}/{label}", flush=True)
            subprocess.run(command, check=True, stdout=subprocess.DEVNULL)
            payload = json.loads((condition_dir / "summary.json").read_text())
            for run in payload["runs"]:
                rows.append(
                    {
                        "distribution": distribution,
                        "variant": variant,
                        "weight": weight if weight is not None else 0.0,
                        **run,
                    }
                )
            print(f"DONE {distribution}/{label}", flush=True)

    csv_path = args.output_dir / "metrics.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    manifest = {
        "configuration": vars(args) | {"output_dir": str(args.output_dir)},
        "runs": rows,
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    lines = [
        "# SIGReg follow-up sweep",
        "",
        "Each cell reports the mean across seeds; ± is sample standard deviation.",
        "Bimodal state coordinates are independent standardized sign mixtures with 0.15 Gaussian jitter.",
        "",
        "| latent distribution | objective | weight | probe R² | predicted next-state R² | effective rank / 8 | covariance RMSE | train seconds |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for distribution in distributions:
        for variant, weight in conditions:
            selected = [
                row for row in rows
                if row["distribution"] == distribution
                and row["variant"] == variant
                and row["weight"] == (weight if weight is not None else 0.0)
            ]

            lines.append(
                f"| {distribution} | {variant} | {weight if weight is not None else '—'} | "
                f"{_metric(selected, 'linear_probe_r2')} | "
                f"{_metric(selected, 'predicted_next_state_r2')} | "
                f"{_metric(selected, 'effective_rank')} | "
                f"{_metric(selected, 'covariance_identity_rmse')} | "
                f"{_metric(selected, 'train_seconds')} |"
            )
    (args.output_dir / "REPORT.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )
    print(f"WROTE {args.output_dir}", flush=True)


if __name__ == "__main__":
    main()
