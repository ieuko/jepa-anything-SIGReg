"""Plot the 32-epoch CIFAR-10 SIGReg weight sweep from saved JSON results."""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path

import matplotlib.pyplot as plt


def _runs(path: Path, variant: str) -> list[dict[str, float]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    return [item for item in payload["results"] if item["variant"] == variant]


def _mean_and_sd(runs: list[dict[str, float]], key: str) -> tuple[float, float]:
    values = [float(item[key]) for item in runs]
    return statistics.fmean(values), statistics.stdev(values)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--result-dir", type=Path, required=True)
    args = parser.parse_args()
    root = args.result_dir
    files = {
        0.001: root / "sigreg-w0p001-32epochs/results.json",
        0.005: root / "long-32epochs/results.json",
        0.02: root / "sigreg-w0p02-32epochs/results.json",
        0.05: root / "sigreg-w0p05-32epochs/results.json",
        0.1: root / "sigreg-w0p1-32epochs/results.json",
    }
    weights = list(files)
    runs = [_runs(files[weight], "sigreg") for weight in weights]
    accuracies = [_mean_and_sd(item, "linear_probe_accuracy") for item in runs]
    ranks = [_mean_and_sd(item, "effective_rank") for item in runs]
    random_accuracy, _ = _mean_and_sd(_runs(root / "results.json", "random"), "linear_probe_accuracy")
    variance_runs = _runs(root / "long-32epochs/results.json", "variance")
    variance_accuracy, _ = _mean_and_sd(variance_runs, "linear_probe_accuracy")
    variance_rank, _ = _mean_and_sd(variance_runs, "effective_rank")

    figure, axes = plt.subplots(2, 1, figsize=(7, 6), sharex=True, constrained_layout=True)
    axes[0].errorbar(
        weights, [value[0] * 100 for value in accuracies],
        yerr=[value[1] * 100 for value in accuracies], marker="o", capsize=3,
        label="SIGReg (3 seeds)",
    )
    axes[0].axhline(random_accuracy * 100, color="gray", linestyle="--", label="Random encoder")
    axes[0].axhline(variance_accuracy * 100, color="tab:orange", linestyle=":", label="Variance floor")
    axes[0].set_ylabel("Linear-probe accuracy (%)")
    axes[0].legend(loc="lower right")
    axes[0].grid(alpha=0.25)

    axes[1].errorbar(
        weights, [value[0] for value in ranks],
        yerr=[value[1] for value in ranks], marker="o", capsize=3,
        color="tab:blue",
    )
    axes[1].axhline(variance_rank, color="tab:orange", linestyle=":")
    axes[1].set_ylabel("Effective rank (out of 64)")
    axes[1].set_xlabel("SIGReg weight (log scale)")
    axes[1].set_xscale("log")
    axes[1].set_xticks(weights, [f"{value:g}" for value in weights])
    axes[1].grid(alpha=0.25)
    figure.suptitle("CIFAR-10 exploratory weight sweep, 32 epochs")
    figure.savefig(root / "weight_sweep.png", dpi=180)
    plt.close(figure)


if __name__ == "__main__":
    main()
