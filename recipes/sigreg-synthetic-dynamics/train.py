#!/usr/bin/env python3
"""A controlled collapse-prevention ablation for JEPA Anything + SIGReg."""

from __future__ import annotations

import argparse
import csv
import json
import math
import platform
import statistics
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch
from jepa_anything_core import (
    OrthogonalFactorProjection,
    SIGReg,
    factor_prediction_loss,
    jepa_anything_objective,
)
from torch import Tensor, nn

VARIANTS = ("prediction-only", "variance", "sigreg")


@dataclass(frozen=True)
class RunMetrics:
    variant: str
    seed: int
    prediction_mse: float
    coordinate_std_mean: float
    coordinate_std_min: float
    collapsed_coordinate_fraction: float
    covariance_identity_rmse: float
    covariance_condition_number: float
    effective_rank: float
    linear_probe_r2: float
    sigreg_statistic: float
    train_seconds: float


class SyntheticWorld:
    """Stationary latent dynamics observed through a fixed nonlinear mixing."""

    def __init__(self, state_dim: int, observation_dim: int, device: torch.device) -> None:
        generator = torch.Generator(device=device).manual_seed(20260925)
        blocks: list[Tensor] = []
        for index in range(state_dim // 2):
            angle = 0.18 + 0.09 * index
            radius = 0.96 - 0.035 * index
            blocks.append(
                radius
                * torch.tensor(
                    [
                        [math.cos(angle), -math.sin(angle)],
                        [math.sin(angle), math.cos(angle)],
                    ],
                    device=device,
                )
            )
        self.dynamics = torch.block_diag(*blocks)
        linear = torch.randn(
            state_dim,
            observation_dim,
            generator=generator,
            device=device,
        )
        nonlinear = torch.randn(
            state_dim,
            observation_dim,
            generator=generator,
            device=device,
        )
        self.linear = linear / linear.norm(dim=0, keepdim=True).clamp_min(1e-8)
        self.nonlinear = nonlinear / nonlinear.norm(dim=0, keepdim=True).clamp_min(1e-8)

    def observe(self, state: Tensor, generator: torch.Generator) -> Tensor:
        signal = state @ self.linear + 0.2 * (state.square() - 1.0) @ self.nonlinear
        noise = 0.02 * torch.randn(
            signal.shape,
            generator=generator,
            device=signal.device,
            dtype=signal.dtype,
        )
        return torch.tanh(signal + noise)

    def sample(
        self,
        count: int,
        generator: torch.Generator,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        state = torch.randn(
            count,
            self.dynamics.shape[0],
            generator=generator,
            device=self.dynamics.device,
        )
        transition_noise = 0.08 * torch.randn(
            state.shape,
            generator=generator,
            device=state.device,
        )
        next_state = state @ self.dynamics.T + transition_noise
        return (
            self.observe(state, generator),
            self.observe(next_state, generator),
            state,
            next_state,
        )


class TinyJEPA(nn.Module):
    def __init__(self, observation_dim: int, state_dim: int, hidden_dim: int) -> None:
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(observation_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, state_dim),
        )
        self.predictor = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, state_dim),
        )

    def forward(self, context: Tensor, target: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        context_state = self.encoder(context)
        target_state = self.encoder(target)
        predicted_state = self.predictor(context_state)
        return predicted_state, target_state, context_state


def _choose_device(requested: str) -> torch.device:
    if requested != "auto":
        return torch.device(requested)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _covariance_metrics(embeddings: Tensor) -> dict[str, float]:
    work = embeddings.detach().double().cpu()
    centered = work - work.mean(dim=0, keepdim=True)
    covariance = centered.T @ centered / centered.shape[0]
    eigenvalues = torch.linalg.eigvalsh(covariance).clamp_min(0.0)
    probabilities = eigenvalues / eigenvalues.sum().clamp_min(1e-12)
    entropy = -(probabilities * probabilities.clamp_min(1e-12).log()).sum()
    standard_deviations = covariance.diag().clamp_min(0.0).sqrt()
    condition = eigenvalues[-1] / eigenvalues[0].clamp_min(1e-12)
    identity = torch.eye(covariance.shape[0], dtype=covariance.dtype)
    return {
        "coordinate_std_mean": float(standard_deviations.mean()),
        "coordinate_std_min": float(standard_deviations.min()),
        "collapsed_coordinate_fraction": float((standard_deviations < 0.1).double().mean()),
        "covariance_identity_rmse": float((covariance - identity).square().mean().sqrt()),
        "covariance_condition_number": float(condition),
        "effective_rank": float(entropy.exp()),
    }


def _linear_probe_r2(
    train_embeddings: Tensor,
    train_state: Tensor,
    test_embeddings: Tensor,
    test_state: Tensor,
) -> float:
    train_x = train_embeddings.detach().double().cpu()
    test_x = test_embeddings.detach().double().cpu()
    train_y = train_state.detach().double().cpu()
    test_y = test_state.detach().double().cpu()
    train_design = torch.cat((train_x, torch.ones(train_x.shape[0], 1)), dim=1)
    test_design = torch.cat((test_x, torch.ones(test_x.shape[0], 1)), dim=1)
    coefficients = torch.linalg.lstsq(train_design, train_y).solution
    prediction = test_design @ coefficients
    residual = (test_y - prediction).square().sum()
    total = (test_y - test_y.mean(dim=0, keepdim=True)).square().sum().clamp_min(1e-12)
    return float(1.0 - residual / total)


def _train_one(
    *,
    variant: str,
    seed: int,
    args: argparse.Namespace,
    device: torch.device,
    world: SyntheticWorld,
) -> tuple[RunMetrics, list[dict[str, float]]]:
    torch.manual_seed(seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)
    model = TinyJEPA(args.observation_dim, args.state_dim, args.hidden_dim).to(device)
    projection = OrthogonalFactorProjection(
        args.state_dim,
        args.num_factors,
        args.state_dim // args.num_factors,
        learnable=False,
    ).to(device)
    regularizer = SIGReg(num_slices=args.num_slices, seed=seed).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)
    data_generator = torch.Generator(device=device).manual_seed(10000 + seed)
    history: list[dict[str, float]] = []
    started = time.perf_counter()

    model.train()
    for step in range(1, args.steps + 1):
        context, target, _, _ = world.sample(args.batch_size, data_generator)
        predicted_state, target_state, context_state = model(context, target)
        predicted_factors = projection.decompose(predicted_state)
        target_factors = projection.decompose(target_state)

        if variant == "prediction-only":
            total = factor_prediction_loss(predicted_factors, target_factors)
            prediction = total
            regularization = total.detach() * 0.0
        else:
            use_sigreg = variant == "sigreg"
            objective = jepa_anything_objective(
                predicted_factors,
                target_factors,
                projection.analysis_basis(),
                context_state,
                orthogonality_weight=0.0,
                factor_activity_weight=0.0 if use_sigreg else args.variance_weight,
                encoder_variance_weight=0.0 if use_sigreg else args.variance_weight,
                sigreg_weight=args.sigreg_weight if use_sigreg else 0.0,
                sigreg=regularizer,
                sigreg_embeddings=torch.cat((context_state, target_state), dim=0),
                factor_min_std=1.0,
                encoder_min_std=1.0,
            )
            total = objective.total
            prediction = objective.prediction
            regularization = (
                args.sigreg_weight * objective.sigreg
                if use_sigreg
                else args.variance_weight
                * (objective.factor_activity + objective.encoder_variance)
            )

        optimizer.zero_grad(set_to_none=True)
        total.backward()
        optimizer.step()

        if step == 1 or step % args.log_every == 0 or step == args.steps:
            history.append(
                {
                    "step": float(step),
                    "total": float(total.detach()),
                    "prediction": float(prediction.detach()),
                    "regularization": float(regularization.detach()),
                }
            )

    if device.type == "cuda":
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - started

    model.eval()
    with torch.inference_mode():
        probe_generator = torch.Generator(device=device).manual_seed(70000 + seed)
        train_context, _, train_state, _ = world.sample(args.eval_samples, probe_generator)
        test_context, test_target, test_state, _ = world.sample(
            args.eval_samples,
            probe_generator,
        )
        train_embeddings = model.encoder(train_context)
        test_embeddings = model.encoder(test_context)
        target_embeddings = model.encoder(test_target)
        predicted_embeddings = model.predictor(test_embeddings)
        prediction_mse = float((predicted_embeddings - target_embeddings).square().mean())
        covariance = _covariance_metrics(test_embeddings)
        probe_r2 = _linear_probe_r2(
            train_embeddings,
            train_state,
            test_embeddings,
            test_state,
        )
        evaluation_sigreg = SIGReg(
            num_slices=max(args.num_slices, 512),
            seed=314159,
        ).to(device)
        sigreg_statistic = float(evaluation_sigreg(test_embeddings))

    return (
        RunMetrics(
            variant=variant,
            seed=seed,
            prediction_mse=prediction_mse,
            linear_probe_r2=probe_r2,
            sigreg_statistic=sigreg_statistic,
            train_seconds=elapsed,
            **covariance,
        ),
        history,
    )


def _summarize(runs: list[RunMetrics]) -> dict[str, dict[str, dict[str, float]]]:
    summary: dict[str, dict[str, dict[str, float]]] = {}
    metric_names = [
        name
        for name in RunMetrics.__dataclass_fields__
        if name not in {"variant", "seed"}
    ]
    for variant in VARIANTS:
        selected = [run for run in runs if run.variant == variant]
        if not selected:
            continue
        summary[variant] = {}
        for metric in metric_names:
            values = [float(getattr(run, metric)) for run in selected]
            summary[variant][metric] = {
                "mean": statistics.fmean(values),
                "std": statistics.stdev(values) if len(values) > 1 else 0.0,
            }
    return summary


def _write_report(payload: dict[str, Any], path: Path) -> None:
    summary = payload["summary"]
    lines = [
        "# JEPA Anything + SIGReg synthetic ablation",
        "",
        (
            "The experiment jointly trains one encoder and a next-state predictor. "
            "Prediction-only training admits the trivial constant solution; the variance "
            "baseline uses the existing coordinate activity floors; SIGReg replaces those "
            "floors with sliced Epps--Pulley regularization toward an isotropic Gaussian."
        ),
        "",
        "| variant | pred. MSE ↓ | min std ↑ | cov. RMSE ↓ | eff. rank ↑ | probe R² ↑ | SIGReg ↓ |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for variant in VARIANTS:
        if variant not in summary:
            continue
        values = summary[variant]
        lines.append(
            f"| {variant} | {values['prediction_mse']['mean']:.5f} | "
            f"{values['coordinate_std_min']['mean']:.4f} | "
            f"{values['covariance_identity_rmse']['mean']:.4f} | "
            f"{values['effective_rank']['mean']:.3f} | "
            f"{values['linear_probe_r2']['mean']:.4f} | "
            f"{values['sigreg_statistic']['mean']:.3f} |"
        )
    lines.extend(
        [
            "",
            (
                f"Device: `{payload['environment']['device_name']}`; PyTorch "
                f"`{payload['environment']['torch_version']}`; "
                f"{len(payload['runs'])} total runs."
            ),
            "",
            (
                "These are controlled synthetic results, not a domain benchmark. The linear "
                "probe uses ground-truth simulator state only for evaluation."
            ),
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path("runs/sigreg-synthetic"))
    parser.add_argument("--device", default="auto")
    parser.add_argument("--variants", default=",".join(VARIANTS))
    parser.add_argument("--seeds", default="0,1,2")
    parser.add_argument("--steps", type=int, default=1500)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--eval-samples", type=int, default=4096)
    parser.add_argument("--state-dim", type=int, default=8)
    parser.add_argument("--observation-dim", type=int, default=32)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--num-factors", type=int, default=4)
    parser.add_argument("--num-slices", type=int, default=256)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--variance-weight", type=float, default=1.0)
    parser.add_argument("--sigreg-weight", type=float, default=0.02)
    parser.add_argument("--log-every", type=int, default=100)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    variants = tuple(item.strip() for item in args.variants.split(",") if item.strip())
    invalid_variants = sorted(set(variants) - set(VARIANTS))
    if invalid_variants:
        raise ValueError(f"unknown variants: {invalid_variants}")
    seeds = tuple(int(item) for item in args.seeds.split(",") if item.strip())
    if args.state_dim % 2 != 0 or args.state_dim % args.num_factors != 0:
        raise ValueError("state_dim must be even and divisible by num_factors")
    if not seeds or args.steps <= 0 or args.batch_size <= 0 or args.eval_samples <= 0:
        raise ValueError("seeds, steps, batch_size, and eval_samples must be non-empty/positive")

    device = _choose_device(args.device)
    world = SyntheticWorld(args.state_dim, args.observation_dim, device)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    runs: list[RunMetrics] = []
    histories: dict[str, list[dict[str, float]]] = {}
    for variant in variants:
        for seed in seeds:
            metrics, history = _train_one(
                variant=variant,
                seed=seed,
                args=args,
                device=device,
                world=world,
            )
            runs.append(metrics)
            histories[f"{variant}/seed-{seed}"] = history
            print(json.dumps(asdict(metrics), sort_keys=True), flush=True)

    device_name = (
        torch.cuda.get_device_name(device)
        if device.type == "cuda"
        else platform.processor() or "CPU"
    )
    payload: dict[str, Any] = {
        "schema_version": 1,
        "configuration": {
            key: str(value) if isinstance(value, Path) else value
            for key, value in vars(args).items()
        },
        "environment": {
            "device": str(device),
            "device_name": device_name,
            "torch_version": torch.__version__,
            "cuda_version": torch.version.cuda,
            "python_version": platform.python_version(),
        },
        "runs": [asdict(run) for run in runs],
        "summary": _summarize(runs),
        "history": histories,
    }
    summary_path = args.output_dir / "summary.json"
    summary_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    with (args.output_dir / "metrics.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(RunMetrics.__dataclass_fields__))
        writer.writeheader()
        writer.writerows(asdict(run) for run in runs)
    _write_report(payload, args.output_dir / "REPORT.md")
    print(f"wrote {summary_path}")


if __name__ == "__main__":
    main()
