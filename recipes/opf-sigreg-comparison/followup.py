"""Pre-registered validation, active-floor, and external-stream OPF study."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import platform
import statistics
from pathlib import Path
from typing import Any

import torch
from train import RECIPE, evaluate, make_data, train_one

SCREEN_SEEDS = (0, 1)
FINAL_SEEDS = (0, 1, 2, 3, 4)
STRESS_SEEDS = (0, 1, 2)
SIGREG_WEIGHTS = (0.001, 0.005)
GRAM_WEIGHTS = (0.05, 0.5, 5.0)
PRIOR_CELL = (0.005, 0.05)


def _arguments(
    output: Path,
    variant: str,
    sigreg_weight: float,
    gram_weight: float,
    *,
    floor_min_std: float | None = None,
) -> argparse.Namespace:
    return argparse.Namespace(
        batch_size=256,
        device=torch.device("cpu"),
        ema_momentum=0.996,
        encoder_min_std=floor_min_std,
        evaluation_split="validation",
        factor_min_std=floor_min_std,
        hidden=444,
        init_output_scale=1.0,
        learning_rate=0.001,
        log_every=1000,
        num_slices=256,
        orthogonality_weight=gram_weight,
        output_dir=output,
        probe_ridge=0.001,
        sigreg_weight=sigreg_weight,
        steps=1000,
        variants=variant,
    )


def _save(output: Path, payload: dict[str, Any]) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(".part")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    temporary.replace(output)


def _run(
    name: str,
    seed: int,
    args: argparse.Namespace,
    data: dict[str, torch.Tensor],
    recipe: dict[str, Any],
    *,
    external: dict[str, dict[str, torch.Tensor]] | None = None,
) -> dict[str, Any]:
    metrics, history, model = train_one(args.variants, seed, data, recipe, args)
    row: dict[str, Any] = {
        "cell": name,
        "seed": seed,
        "variant": args.variants,
        "sigreg_weight": args.sigreg_weight if args.variants != "original" else 0.0,
        "gram_weight": args.orthogonality_weight,
        "floor_min_std": args.factor_min_std,
        "validation": metrics,
        "history": history,
    }
    if external is not None:
        row["external"] = {
            key: evaluate(model, data, recipe, args.probe_ridge, "test", evaluation)
            for key, evaluation in external.items()
        }
    return row


def _mean(rows: list[dict[str, Any]], cell: str, metric: str) -> float:
    return statistics.mean(
        row["validation"][metric] for row in rows if row["cell"] == cell
    )


def _select(screen: list[dict[str, Any]]) -> dict[str, Any]:
    baseline_h1 = _mean(screen, "original", "rollout_r2_h1")
    baseline_h8 = _mean(screen, "original", "rollout_r2_h8")
    candidates: list[dict[str, Any]] = [
        {
            "cell": f"sigreg-{sigreg:g}-gram-{gram:g}",
            "sigreg_weight": sigreg,
            "gram_weight": gram,
        }
        for sigreg in SIGREG_WEIGHTS
        for gram in GRAM_WEIGHTS
    ]
    for item in candidates:
        name = str(item["cell"])
        item["validation_h1"] = _mean(screen, name, "rollout_r2_h1")
        item["validation_h8"] = _mean(screen, name, "rollout_r2_h8")
        item["validation_gram_rmse"] = _mean(screen, name, "basis_gram_rmse")
    h1_ok = lambda item: float(item["validation_h1"]) >= baseline_h1 - 0.02
    eligible = [
        item
        for item in candidates
        if float(item["validation_gram_rmse"]) <= 0.03 and h1_ok(item)
    ]
    if eligible:
        chosen = min(
            eligible,
            key=lambda item: (
                -float(item["validation_h8"]),
                float(item["validation_gram_rmse"]),
                float(item["sigreg_weight"]),
            ),
        )
        rule = "geometry_guard_met"
    else:
        fallbacks = [
            item
            for item in candidates
            if float(item["validation_h8"]) >= baseline_h8 and h1_ok(item)
        ]
        chosen = min(
            fallbacks or candidates,
            key=lambda item: (
                float(item["validation_gram_rmse"]),
                -float(item["validation_h8"]),
                float(item["sigreg_weight"]),
            ),
        )
        rule = "fallback_geometry_guard_not_met"
    return {
        "baseline_validation_h1": baseline_h1,
        "baseline_validation_h8": baseline_h8,
        "candidates": candidates,
        "chosen": chosen,
        "rule": rule,
    }


def _external_recipes(recipe: dict[str, Any]) -> dict[str, dict[str, Any]]:
    iid = copy.deepcopy(recipe)
    iid["provenance"]["seed"] = 18181
    shifted = copy.deepcopy(recipe)
    shifted["provenance"]["seed"] = 19191
    shifted["system"]["transition"]["state_matrix"][0][1] = 0.18
    shifted["system"]["transition"]["state_matrix"][2][3] = 0.23
    shifted["system"]["transition"]["process_noise_std"] = 0.015
    shifted["system"]["observation"]["noise_std"] = 0.03
    return {"iid_new_seed": iid, "shifted_dynamics": shifted}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--recipe", type=Path, default=RECIPE)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    recipe_bytes = args.recipe.read_bytes()
    recipe: dict[str, Any] = json.loads(recipe_bytes)
    data, stream_hash = make_data(recipe, torch.device("cpu"))
    output = args.output_dir / "results.json"
    payload: dict[str, Any] = {
        "protocol": "recipes/opf-sigreg-comparison/FOLLOWUP_PROTOCOL.md",
        "recipe_sha256": hashlib.sha256(recipe_bytes).hexdigest(),
        "train_stream_sha256": stream_hash,
        "environment": {
            "device": "CPU",
            "python": platform.python_version(),
            "torch": torch.__version__,
        },
        "screen": [],
        "selection": None,
        "stress": [],
        "external_stream_sha256": {},
        "external": [],
    }

    print("STAGE 1: validation-only screen", flush=True)
    for seed in SCREEN_SEEDS:
        row = _run(
            "original",
            seed,
            _arguments(args.output_dir, "original", 0, 0.05),
            data,
            recipe,
        )
        payload["screen"].append(row)
        _save(output, payload)
    for sigreg in SIGREG_WEIGHTS:
        for gram in GRAM_WEIGHTS:
            name = f"sigreg-{sigreg:g}-gram-{gram:g}"
            for seed in SCREEN_SEEDS:
                row = _run(
                    name,
                    seed,
                    _arguments(
                        args.output_dir,
                        "original-plus-sigreg",
                        sigreg,
                        gram,
                    ),
                    data,
                    recipe,
                )
                payload["screen"].append(row)
                _save(output, payload)
    payload["selection"] = _select(payload["screen"])
    _save(output, payload)
    print(
        "SELECTED",
        json.dumps(payload["selection"]["chosen"], sort_keys=True),
        flush=True,
    )

    print("STAGE 2: active-floor stress", flush=True)
    for variant in ("original", "original-plus-sigreg", "sigreg-replaces-floors"):
        for seed in STRESS_SEEDS:
            row = _run(
                variant,
                seed,
                _arguments(
                    args.output_dir,
                    variant,
                    0.005,
                    0.05,
                    floor_min_std=1.0,
                ),
                data,
                recipe,
            )
            payload["stress"].append(row)
            _save(output, payload)

    print("STAGE 3: locked external streams", flush=True)
    external: dict[str, dict[str, torch.Tensor]] = {}
    normalizer = (data["observation_mean"], data["observation_std"])
    for name, external_recipe in _external_recipes(recipe).items():
        external[name], stream_hash = make_data(
            external_recipe,
            torch.device("cpu"),
            normalizer,
        )
        payload["external_stream_sha256"][name] = stream_hash
    _save(output, payload)
    selected = payload["selection"]["chosen"]
    final_cells: list[tuple[str, str, float, float]] = [
        ("original", "original", 0, 0.05),
        (
            str(selected["cell"]),
            "original-plus-sigreg",
            float(selected["sigreg_weight"]),
            float(selected["gram_weight"]),
        ),
    ]
    if (float(selected["sigreg_weight"]), float(selected["gram_weight"])) != PRIOR_CELL:
        final_cells.append(("prior-sigreg", "original-plus-sigreg", *PRIOR_CELL))
    for cell, variant, sigreg, gram in final_cells:
        for seed in FINAL_SEEDS:
            row = _run(
                cell,
                seed,
                _arguments(
                    args.output_dir,
                    variant,
                    sigreg,
                    gram,
                ),
                data,
                recipe,
                external=external,
            )
            payload["external"].append(row)
            _save(output, payload)
    print(f"WROTE {output}", flush=True)


if __name__ == "__main__":
    main()
