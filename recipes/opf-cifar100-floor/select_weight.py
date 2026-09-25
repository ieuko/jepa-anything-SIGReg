"""Select one predeclared SIGReg weight using CIFAR-100 validation only."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

WEIGHTS = (0.0001, 0.0005, 0.001, 0.005)
DIRS = ("w0p0001", "w0p0005", "w0p001", "w0p005")
SEEDS = (0, 1)
TIE_BAND = 0.002


def read_cell(path: Path, expected_weight: float) -> dict[str, Any]:
    payload: dict[str, Any] = json.loads(path.read_text(encoding="utf-8"))
    config = payload["configuration"]
    if (
        config["dataset"] != "cifar100"
        or not config["validation_only"]
        or config["epochs"] != 16
        or config["seeds"] != "0,1"
        or config["sigreg_weight"] != expected_weight
    ):
        raise ValueError(f"unexpected screen configuration: {path}")
    results = payload["results"]
    if any("test_probe_accuracy" in row for row in results):
        raise ValueError(f"test score found in validation-only screen: {path}")
    plus = [row for row in results if row["variant"] == "plus"]
    if tuple(row["seed"] for row in plus) != SEEDS:
        raise ValueError(f"missing plus seeds: {path}")
    return {
        "weight": expected_weight,
        "validation_accuracies": [row["validation_probe_accuracy"] for row in plus],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--screen-root", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    cells = [
        read_cell(args.screen_root / dirname / "results.json", weight)
        for weight, dirname in zip(WEIGHTS, DIRS)
    ]
    for cell in cells:
        cell["mean_validation_accuracy"] = sum(cell["validation_accuracies"]) / 2
    maximum = max(cell["mean_validation_accuracy"] for cell in cells)
    eligible = [
        cell for cell in cells if cell["mean_validation_accuracy"] >= maximum - TIE_BAND
    ]
    selected = min(eligible, key=lambda cell: cell["weight"])
    first = json.loads(
        (args.screen_root / DIRS[0] / "results.json").read_text(encoding="utf-8")
    )
    original = [row for row in first["results"] if row["variant"] == "original"]
    if tuple(row["seed"] for row in original) != SEEDS:
        raise ValueError("missing original seeds")
    original_mean = sum(row["validation_probe_accuracy"] for row in original) / 2
    result = {
        "protocol": "recipes/opf-cifar100-floor/LOW_WEIGHT_TRANSFER_PROTOCOL.md",
        "criterion": "highest mean CIFAR-100 validation accuracy; smallest weight within 0.002",
        "weights": cells,
        "original_validation_accuracies": [
            row["validation_probe_accuracy"] for row in original
        ],
        "original_mean_validation_accuracy": original_mean,
        "selected_weight": selected["weight"],
        "selected_minus_original_validation_accuracy": (
            selected["mean_validation_accuracy"] - original_mean
        ),
        "svhn_scores_viewed_before_selection": False,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
