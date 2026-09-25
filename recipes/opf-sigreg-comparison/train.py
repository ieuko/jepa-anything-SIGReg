"""Compare the public JEPA-Anything OPF objective with SIGReg additions."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import platform
import random
import struct
import time
from pathlib import Path
from typing import Any

import torch
from jepa_anything_core import (
    OrthogonalFactorProjection,
    SIGReg,
    ema_update,
    jepa_anything_objective,
    parameter_count,
)
from torch import Tensor, nn

VARIANTS = ("original", "original-plus-sigreg", "sigreg-replaces-floors")
HORIZONS = (1, 2, 4, 8)
RECIPE = Path("recipes/synthetic-linear-dynamics/recipe.json")


def _matvec(matrix: list[list[float]], vector: list[float]) -> list[float]:
    return [sum(a * b for a, b in zip(row, vector)) for row in matrix]


def make_data(
    recipe: dict[str, Any], device: torch.device,
) -> tuple[dict[str, Tensor], str]:
    """Generate the declared simulator, preserving trajectory-level splits."""

    system = recipe["system"]
    transition = system["transition"]
    observation = system["observation"]
    n_trajectories = int(recipe["data"]["trajectory_count"])
    steps = int(recipe["data"]["steps_per_trajectory"])
    state_dim = int(system["state_dim"])
    rng = random.Random(int(recipe["provenance"]["seed"]))
    fingerprint = hashlib.sha256()
    fingerprint.update(b"jepa-anything.synthetic-linear-dynamics/v1\0")
    observations: list[list[list[float]]] = []
    states: list[list[list[float]]] = []
    controls: list[list[list[float]]] = []
    for trajectory_id in range(n_trajectories):
        phase = rng.uniform(-math.pi, math.pi)
        state = [rng.gauss(0.0, 0.6) for _ in range(state_dim)]
        trajectory_obs: list[list[float]] = []
        trajectory_states: list[list[float]] = []
        trajectory_controls: list[list[float]] = []
        for step in range(steps):
            control = 0.7 * math.sin(0.11 * step + phase) + 0.2 * math.cos(0.037 * step)
            observed = [
                value + rng.gauss(0.0, float(observation["noise_std"]))
                for value in _matvec(observation["matrix"], state)
            ]
            fingerprint.update(struct.pack(">II", trajectory_id, step))
            for value in (*state, control, *observed):
                fingerprint.update(struct.pack(">d", float(value)))
            trajectory_obs.append(observed)
            trajectory_states.append(state)
            trajectory_controls.append([control])
            next_state = _matvec(transition["state_matrix"], state)
            state = [
                next_state[index]
                + float(transition["control_matrix"][index][0]) * control
                + rng.gauss(0.0, float(transition["process_noise_std"]))
                for index in range(state_dim)
            ]
        observations.append(trajectory_obs)
        states.append(trajectory_states)
        controls.append(trajectory_controls)

    obs = torch.tensor(observations, dtype=torch.float32, device=device)
    state_tensor = torch.tensor(states, dtype=torch.float32, device=device)
    control_tensor = torch.tensor(controls, dtype=torch.float32, device=device)
    first, last = recipe["data"]["split"]["train_ids"]
    train_obs = obs[first : last + 1].reshape(-1, obs.shape[-1])
    obs_mean = train_obs.mean(0)
    obs_std = train_obs.std(0).clamp_min(1e-6)
    return {
        "observations": (obs - obs_mean) / obs_std,
        "states": state_tensor,
        "controls": control_tensor,
        "observation_mean": obs_mean,
        "observation_std": obs_std,
    }, fingerprint.hexdigest()


class OPFModel(nn.Module):
    """Online/EMA encoders, learned OPF basis, and factor-specific predictors."""

    def __init__(self, observation_dim: int, state_dim: int, num_factors: int, hidden: int) -> None:
        super().__init__()
        factor_dim = state_dim // num_factors
        self.online_encoder = nn.Sequential(
            nn.Linear(observation_dim, hidden), nn.GELU(),
            nn.Linear(hidden, state_dim),
        )
        self.target_encoder = copy.deepcopy(self.online_encoder)
        self.target_encoder.requires_grad_(False)
        self.projection = OrthogonalFactorProjection(
            state_dim, num_factors, factor_dim, learnable=True,
        )
        self.factor_predictors = nn.ModuleList([
            nn.Sequential(
                nn.Linear(state_dim + 1, hidden), nn.GELU(),
                nn.Linear(hidden, factor_dim),
            )
            for _ in range(num_factors)
        ])

    def train(self, mode: bool = True) -> OPFModel:
        super().train(mode)
        self.target_encoder.eval()
        return self

    def predict_factors(self, context_state: Tensor, control: Tensor) -> Tensor:
        inputs = torch.cat((context_state, control), dim=-1)
        return torch.stack([head(inputs) for head in self.factor_predictors], dim=-2)

    def predict_state(self, context_state: Tensor, control: Tensor) -> Tensor:
        return self.projection.compose(self.predict_factors(context_state, control))


def _r2(predicted: Tensor, actual: Tensor) -> float:
    residual = (predicted - actual).double().square().sum()
    centered = actual.double() - actual.double().mean(dim=0, keepdim=True)
    total = centered.square().sum().clamp_min(1e-12)
    return float(1 - residual / total)


def _geometry(features: Tensor, projection: OrthogonalFactorProjection) -> dict[str, float]:
    centered = features.double() - features.double().mean(0)
    covariance = centered.T @ centered / len(features)
    eigenvalues = torch.linalg.eigvalsh(covariance).clamp_min(0)
    probabilities = eigenvalues / eigenvalues.sum().clamp_min(1e-12)
    rank = (-(probabilities * probabilities.clamp_min(1e-12).log()).sum()).exp()
    basis = projection.analysis_basis().detach().double().reshape(features.shape[-1], -1)
    gram = basis @ basis.T
    identity = torch.eye(len(gram), device=gram.device, dtype=gram.dtype)
    reconstructed = projection.compose(projection.decompose(features))
    return {
        "effective_rank": float(rank),
        "min_coordinate_std": float(covariance.diag().clamp_min(0).sqrt().min()),
        "basis_gram_rmse": float((gram - identity).square().mean().sqrt()),
        "state_roundtrip_rmse": float((reconstructed - features).square().mean().sqrt()),
    }


def evaluate(
    model: OPFModel, data: dict[str, Tensor], recipe: dict[str, Any], ridge: float,
) -> dict[str, float]:
    model.eval()
    obs, states, controls = data["observations"], data["states"], data["controls"]
    train_first, train_last = recipe["data"]["split"]["train_ids"]
    test_first, test_last = recipe["data"]["split"]["test_ids"]
    with torch.inference_mode():
        train_features = model.online_encoder(
            obs[train_first : train_last + 1].reshape(-1, obs.shape[-1])
        ).double()
        train_states = states[train_first : train_last + 1].reshape(-1, states.shape[-1]).double()
        feature_mean = train_features.mean(0)
        feature_std = train_features.std(0).clamp_min(1e-4)
        state_mean = train_states.mean(0)
        design = (train_features - feature_mean) / feature_std
        coefficients = torch.linalg.solve(
            design.T @ design + ridge * len(design) * torch.eye(
                design.shape[-1], device=design.device, dtype=design.dtype,
            ),
            design.T @ (train_states - state_mean),
        )

        def decode(features: Tensor) -> Tensor:
            return ((features.double() - feature_mean) / feature_std) @ coefficients + state_mean

        test_obs = obs[test_first : test_last + 1]
        test_states = states[test_first : test_last + 1]
        test_controls = controls[test_first : test_last + 1]
        all_features = model.online_encoder(test_obs.reshape(-1, obs.shape[-1]))
        metrics = {
            "current_state_probe_r2": _r2(
                decode(all_features), test_states.reshape(-1, states.shape[-1])
            ),
            **_geometry(all_features, model.projection),
        }
        start_count = obs.shape[1] - max(HORIZONS)
        predicted_state = model.online_encoder(
            test_obs[:, :start_count].reshape(-1, obs.shape[-1])
        )
        for horizon in range(1, max(HORIZONS) + 1):
            action = test_controls[:, horizon - 1 : horizon - 1 + start_count].reshape(-1, 1)
            predicted_state = model.predict_state(predicted_state, action)
            if horizon in HORIZONS:
                actual = test_states[:, horizon : horizon + start_count].reshape(
                    -1, states.shape[-1]
                )
                metrics[f"rollout_r2_h{horizon}"] = _r2(decode(predicted_state), actual)
    return metrics


def train_one(
    variant: str, seed: int, data: dict[str, Tensor], recipe: dict[str, Any],
    args: argparse.Namespace,
) -> tuple[dict[str, float | str | int], list[dict[str, float]]]:
    torch.manual_seed(seed)
    if args.device.type == "cuda":
        torch.cuda.manual_seed_all(seed)
    state_dim = int(recipe["representation"]["d"])
    num_factors = int(recipe["representation"]["K"])
    model = OPFModel(int(recipe["system"]["observation_dim"]), state_dim, num_factors, args.hidden).to(args.device)
    capacity = recipe["representation"]["capacity"]
    planned_parameters = int(capacity["full_model_trainable_parameters"])
    tolerance = float(capacity["parameter_tolerance"])
    if abs(parameter_count(model) - planned_parameters) / planned_parameters > tolerance:
        raise RuntimeError("OPF trainable parameter count violates the source recipe")
    optimizer = torch.optim.AdamW(
        (parameter for parameter in model.parameters() if parameter.requires_grad),
        lr=args.learning_rate,
    )
    regularizer = SIGReg(num_slices=args.num_slices, seed=seed).to(args.device)
    train_first, train_last = recipe["data"]["split"]["train_ids"]
    obs = data["observations"][train_first : train_last + 1]
    controls = data["controls"][train_first : train_last + 1]
    context = obs[:, :-1].reshape(-1, obs.shape[-1])
    target = obs[:, 1:].reshape(-1, obs.shape[-1])
    action = controls[:, :-1].reshape(-1, 1)
    sampling = torch.Generator(device=args.device).manual_seed(10000 + seed)
    losses = recipe["losses"]
    floor_weight = 0.0 if variant == "sigreg-replaces-floors" else 1.0
    sigreg_weight = 0.0 if variant == "original" else args.sigreg_weight
    history: list[dict[str, float]] = []
    started = time.perf_counter()
    model.train()
    for step in range(1, args.steps + 1):
        index = torch.randint(len(context), (args.batch_size,), generator=sampling, device=args.device)
        context_state = model.online_encoder(context[index])
        with torch.no_grad():
            target_state = model.target_encoder(target[index])
        predicted_factors = model.predict_factors(context_state, action[index])
        target_factors = model.projection.decompose(target_state)
        objective = jepa_anything_objective(
            predicted_factors, target_factors, model.projection.analysis_basis(), context_state,
            orthogonality_weight=float(losses["projector_orthogonality"]["weight"]),
            factor_activity_weight=floor_weight * float(losses["factor_activity"]["weight"]),
            encoder_variance_weight=floor_weight * float(losses["online_encoder_activity"]["weight"]),
            sigreg_weight=sigreg_weight,
            sigreg=regularizer,
            sigreg_embeddings=context_state,
            factor_min_std=float(losses["factor_activity"]["min_std"]),
            encoder_min_std=float(losses["online_encoder_activity"]["min_std"]),
        )
        optimizer.zero_grad(set_to_none=True)
        objective.total.backward()
        optimizer.step()
        model.projection.after_optimizer_step(step)
        ema_update(model.target_encoder, model.online_encoder, args.ema_momentum)
        if step == 1 or step % args.log_every == 0 or step == args.steps:
            record = {
                "step": float(step),
                "loss": float(objective.total.detach()),
                "prediction": float(objective.prediction.detach()),
                "orthogonality": float(objective.orthogonality.detach()),
                "factor_activity": float(objective.factor_activity.detach()),
                "encoder_variance": float(objective.encoder_variance.detach()),
                "sigreg": float(objective.sigreg.detach()),
            }
            history.append(record)
            print(f"{variant} seed={seed} step={step} loss={record['loss']:.6f}", flush=True)
    if args.device.type == "cuda":
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - started
    metrics: dict[str, float | str | int] = {
        "variant": variant,
        "seed": seed,
        "train_seconds": elapsed,
        "trainable_parameters": parameter_count(model),
        **evaluate(model, data, recipe, args.probe_ridge),
    }
    return metrics, history


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--recipe", type=Path, default=RECIPE)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--variants", default=",".join(VARIANTS))
    parser.add_argument("--seeds", default="0,1,2")
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--batch-size", type=int, default=256)
    # 27 * 444 + 24 = 12,012 trainable parameters: within the fixture's 0.3% plan.
    parser.add_argument("--hidden", type=int, default=444)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--ema-momentum", type=float, default=0.996)
    parser.add_argument("--sigreg-weight", type=float, default=0.005)
    parser.add_argument("--num-slices", type=int, default=256)
    parser.add_argument("--probe-ridge", type=float, default=1e-3)
    parser.add_argument("--log-every", type=int, default=100)
    args = parser.parse_args()
    variants = tuple(value.strip() for value in args.variants.split(",") if value.strip())
    seeds = tuple(int(value) for value in args.seeds.split(",") if value.strip())
    if not variants or len(set(variants)) != len(variants) or set(variants) - set(VARIANTS):
        raise ValueError("variants must be unique members of the protocol")
    if not seeds or len(set(seeds)) != len(seeds) or any(seed < 0 for seed in seeds):
        raise ValueError("seeds must be unique non-negative integers")
    if args.steps <= 0 or args.batch_size < 2 or args.hidden <= 0 or args.log_every <= 0:
        raise ValueError("steps, batch size, hidden width, and log interval must be positive")
    if not 0 <= args.ema_momentum < 1 or args.sigreg_weight < 0 or args.probe_ridge <= 0:
        raise ValueError("invalid EMA momentum, SIGReg weight, or probe ridge")
    args.device = torch.device(
        ("cuda" if torch.cuda.is_available() else "cpu") if args.device == "auto" else args.device
    )
    recipe_bytes = args.recipe.read_bytes()
    recipe: dict[str, Any] = json.loads(recipe_bytes)
    data, stream_sha256 = make_data(recipe, args.device)
    output = args.output_dir / "results.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    results: list[dict[str, float | str | int]] = []
    histories: dict[str, list[dict[str, float]]] = {}
    payload = {
        "configuration": {
            key: str(value) if isinstance(value, (Path, torch.device)) else value
            for key, value in vars(args).items()
        },
        "recipe_sha256": hashlib.sha256(recipe_bytes).hexdigest(),
        "generated_stream_sha256": stream_sha256,
        "split": recipe["data"]["split"],
        "environment": {
            "device_name": torch.cuda.get_device_name(args.device) if args.device.type == "cuda" else "CPU",
            "torch_version": torch.__version__,
            "python_version": platform.python_version(),
        },
        "results": results,
        "history": histories,
    }
    for variant in variants:
        for seed in seeds:
            metrics, history = train_one(variant, seed, data, recipe, args)
            results.append(metrics)
            histories[f"{variant}/seed-{seed}"] = history
            temporary = output.with_suffix(".part")
            temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
            temporary.replace(output)
            print(json.dumps(metrics, sort_keys=True), flush=True)
    print(f"WROTE {output}", flush=True)


if __name__ == "__main__":
    main()
