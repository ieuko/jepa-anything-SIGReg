"""Fixed OPF/SIGReg comparison on UCI HAR inertial-sensor sequences."""

from __future__ import annotations

import argparse
import copy
import hashlib
import io
import json
import platform
import shutil
import time
import urllib.request
import zipfile
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from jepa_anything_core import (
    OrthogonalFactorProjection,
    SIGReg,
    ema_update,
    jepa_anything_objective,
    parameter_count,
)
from torch import Tensor, nn

URL = (
    "https://archive.ics.uci.edu/static/public/240/"
    "human%2Bactivity%2Brecognition%2Busing%2Bsmartphones.zip"
)
CHANNELS = (
    "body_acc_x",
    "body_acc_y",
    "body_acc_z",
    "body_gyro_x",
    "body_gyro_y",
    "body_gyro_z",
    "total_acc_x",
    "total_acc_y",
    "total_acc_z",
)
VARIANTS = (
    "random",
    "original",
    "plus",
    "replace",
    "plus-strong-gram",
    "replace-strong-gram",
)
CHUNKS = 16
CHUNK_WIDTH = 8 * len(CHANNELS)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _archive(data_dir: Path, download_url: str) -> Path:
    data_dir.mkdir(parents=True, exist_ok=True)
    archive = data_dir / "uci-har-240.zip"
    if archive.exists():
        return archive
    temporary = archive.with_suffix(".part")
    request = urllib.request.Request(
        download_url, headers={"User-Agent": "jepa-anything-sigreg/1"}
    )
    with (
        urllib.request.urlopen(request, timeout=120) as response,
        temporary.open("wb") as target,
    ):
        shutil.copyfileobj(response, target)
    with zipfile.ZipFile(temporary) as bundle:
        if bundle.testzip() is not None:
            raise RuntimeError("UCI HAR archive CRC check failed")
    temporary.replace(archive)
    return archive


def _read_array(
    bundle: zipfile.ZipFile, path: str, dtype: type[np.float32 | np.int64]
) -> np.ndarray:
    return np.loadtxt(io.BytesIO(bundle.read(f"UCI HAR Dataset/{path}")), dtype=dtype)


def _read_split(bundle: zipfile.ZipFile, split: str) -> tuple[Tensor, Tensor, Tensor]:
    components = [
        _read_array(
            bundle, f"{split}/Inertial Signals/{channel}_{split}.txt", np.float32
        )
        for channel in CHANNELS
    ]
    signals = np.stack(components, axis=-1)
    if signals.ndim != 3 or signals.shape[1:] != (128, len(CHANNELS)):
        raise RuntimeError("unexpected UCI HAR signal shape")
    labels = _read_array(bundle, f"{split}/y_{split}.txt", np.int64) - 1
    subjects = _read_array(bundle, f"{split}/subject_{split}.txt", np.int64)
    if not (len(signals) == len(labels) == len(subjects)):
        raise RuntimeError("UCI HAR signal, label, and subject counts differ")
    if np.min(labels) < 0 or np.max(labels) > 5:
        raise RuntimeError("unexpected UCI HAR activity label")
    return (
        torch.from_numpy(signals),
        torch.from_numpy(labels),
        torch.from_numpy(subjects),
    )


def _chunks(signals: Tensor, mean: Tensor, std: Tensor, device: torch.device) -> Tensor:
    normalized = (signals.to(device) - mean) / std
    return normalized.reshape(len(signals), CHUNKS, CHUNK_WIDTH)


def load_data(
    data_dir: Path,
    download_url: str,
    device: torch.device,
) -> tuple[dict[str, dict[str, Tensor]], dict[str, Any]]:
    archive = _archive(data_dir, download_url)
    # UCI distributes an outer archive containing the original dataset ZIP.
    with zipfile.ZipFile(archive) as outer:
        inner_bytes = outer.read("UCI HAR Dataset.zip")
    with zipfile.ZipFile(io.BytesIO(inner_bytes)) as bundle:
        train_x, train_y, train_subjects = _read_split(bundle, "train")
        test_x, test_y, test_subjects = _read_split(bundle, "test")
    train_subject_ids = sorted(int(value) for value in train_subjects.unique())
    val_ids = train_subject_ids[-3:]
    train_mask = ~torch.isin(train_subjects, torch.tensor(val_ids))
    mean = train_x[train_mask].mean((0, 1)).to(device)
    std = train_x[train_mask].std((0, 1)).clamp_min(1e-6).to(device)
    partitions = {
        "train": {
            "chunks": _chunks(train_x[train_mask], mean, std, device),
            "labels": train_y[train_mask],
        },
        "validation": {
            "chunks": _chunks(train_x[~train_mask], mean, std, device),
            "labels": train_y[~train_mask],
        },
        "test": {
            "chunks": _chunks(test_x, mean, std, device),
            "labels": test_y,
        },
    }
    if not all(len(partition["chunks"]) for partition in partitions.values()):
        raise RuntimeError("empty UCI HAR partition")
    metadata: dict[str, Any] = {
        "archive_sha256": _sha256(archive),
        "inner_archive_sha256": hashlib.sha256(inner_bytes).hexdigest(),
        "archive_bytes": archive.stat().st_size,
        "download_url": download_url,
        "train_subject_ids": [
            value for value in train_subject_ids if value not in val_ids
        ],
        "validation_subject_ids": val_ids,
        "test_subject_ids": sorted(int(value) for value in test_subjects.unique()),
        "partition_sizes": {
            key: len(value["chunks"]) for key, value in partitions.items()
        },
        "channel_mean_train_only": mean.tolist(),
        "channel_std_train_only": std.tolist(),
    }
    if set(metadata["train_subject_ids"]) & set(val_ids):
        raise RuntimeError("subject overlap between train and validation")
    if set(train_subject_ids) & set(metadata["test_subject_ids"]):
        raise RuntimeError("subject overlap between official train and test")
    return partitions, metadata


def smoke_data(
    device: torch.device,
) -> tuple[dict[str, dict[str, Tensor]], dict[str, Any]]:
    generator = torch.Generator().manual_seed(1729)
    partitions = {}
    for name, count in (("train", 48), ("validation", 18), ("test", 18)):
        signal = torch.randn(count, CHUNKS, CHUNK_WIDTH, generator=generator).to(device)
        labels = torch.arange(count) % 6
        partitions[name] = {"chunks": signal, "labels": labels}
    return partitions, {
        "smoke_data": True,
        "partition_sizes": {
            key: len(value["chunks"]) for key, value in partitions.items()
        },
    }


class Model(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.online_encoder = nn.Sequential(
            nn.Linear(CHUNK_WIDTH, 128),
            nn.GELU(),
            nn.Linear(128, 64),
        )
        self.target_encoder = copy.deepcopy(self.online_encoder)
        self.target_encoder.requires_grad_(False)
        self.projection = OrthogonalFactorProjection(64, 8, 8, learnable=True)
        self.factor_predictors = nn.ModuleList(
            [
                nn.Sequential(nn.Linear(64, 128), nn.GELU(), nn.Linear(128, 8))
                for _ in range(8)
            ]
        )

    def train(self, mode: bool = True) -> Model:
        super().train(mode)
        self.target_encoder.eval()
        return self

    def predict_factors(self, context: Tensor) -> Tensor:
        return torch.stack([head(context) for head in self.factor_predictors], dim=-2)


def _r2(predicted: Tensor, actual: Tensor) -> float:
    residual = (predicted.double() - actual.double()).square().sum()
    centered = actual.double() - actual.double().mean(0)
    total = centered.square().sum().clamp_min(1e-12)
    return float(1 - residual / total)


def _ridge(features: Tensor, targets: Tensor) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    x = features.double().cpu()
    y = targets.double().cpu()
    mean = x.mean(0)
    scale = x.std(0).clamp_min(1e-4)
    target_mean = y.mean(0)
    design = (x - mean) / scale
    coefficients = torch.linalg.solve(
        design.T @ design + 0.001 * len(x) * torch.eye(x.shape[1], dtype=torch.float64),
        design.T @ (y - target_mean),
    )
    return mean, scale, target_mean, coefficients


def _decode(features: Tensor, probe: tuple[Tensor, Tensor, Tensor, Tensor]) -> Tensor:
    mean, scale, target_mean, coefficients = probe
    return ((features.double().cpu() - mean) / scale) @ coefficients + target_mean


def _encode(model: Model, chunks: Tensor, batch_size: int = 2048) -> Tensor:
    flattened = chunks.reshape(-1, CHUNK_WIDTH)
    with torch.inference_mode():
        return torch.cat(
            [model.online_encoder(batch).cpu() for batch in flattened.split(batch_size)]
        ).reshape(len(chunks), CHUNKS, 64)


def _geometry(
    features: Tensor, projection: OrthogonalFactorProjection
) -> dict[str, float]:
    flat = features.reshape(-1, 64).double()
    centered = flat - flat.mean(0)
    covariance = centered.T @ centered / len(flat)
    std = covariance.diag().clamp_min(0).sqrt()
    eigenvalues = torch.linalg.eigvalsh(covariance).clamp_min(0)
    probabilities = eigenvalues / eigenvalues.sum().clamp_min(1e-12)
    rank = (-(probabilities * probabilities.clamp_min(1e-12).log()).sum()).exp()
    basis = projection.analysis_basis().detach().cpu().double().reshape(64, 64)
    gram = basis @ basis.T
    return {
        "effective_rank": float(rank),
        "mean_coordinate_std": float(std.mean()),
        "min_coordinate_std": float(std.min()),
        "gram_rmse": float((gram - torch.eye(64)).square().mean().sqrt()),
    }


def _forecast(
    model: Model,
    chunks: Tensor,
    decoder: tuple[Tensor, Tensor, Tensor, Tensor],
) -> dict[str, float]:
    starts = CHUNKS - 4
    basis = model.projection.analysis_basis().detach().reshape(64, 64)
    synthesis = torch.linalg.pinv(basis).T
    initial = chunks[:, :starts].reshape(-1, CHUNK_WIDTH)
    target = chunks
    predictions: dict[int, list[Tensor]] = {1: [], 4: []}
    with torch.inference_mode():
        for batch in initial.split(2048):
            latent = model.online_encoder(batch)
            for horizon in range(1, 5):
                latent = model.predict_factors(latent).reshape(-1, 64) @ synthesis
                if horizon in predictions:
                    predictions[horizon].append(latent.cpu())
    return {
        f"sensor_rollout_r2_h{horizon}": _r2(
            _decode(torch.cat(predictions[horizon]), decoder),
            target[:, horizon : horizon + starts].reshape(-1, CHUNK_WIDTH).cpu(),
        )
        for horizon in (1, 4)
    }


def evaluate(
    model: Model,
    partitions: dict[str, dict[str, Tensor]],
    random_reference: bool,
) -> dict[str, float]:
    model.eval()
    embeddings = {
        key: _encode(model, value["chunks"]) for key, value in partitions.items()
    }
    train = partitions["train"]
    activity_probe = _ridge(
        embeddings["train"].mean(1), F.one_hot(train["labels"], 6).float()
    )
    metrics: dict[str, float] = {}
    for split in ("validation", "test"):
        scores = _decode(embeddings[split].mean(1), activity_probe)
        metrics[f"{split}_activity_accuracy"] = float(
            (scores.argmax(1) == partitions[split]["labels"]).double().mean()
        )
        metrics.update(
            {
                f"{split}_{key}": value
                for key, value in _geometry(embeddings[split], model.projection).items()
            }
        )
    if not random_reference:
        decoder = _ridge(
            embeddings["train"].reshape(-1, 64),
            train["chunks"].reshape(-1, CHUNK_WIDTH),
        )
        for split in ("validation", "test"):
            metrics.update(
                {
                    f"{split}_{key}": value
                    for key, value in _forecast(
                        model, partitions[split]["chunks"], decoder
                    ).items()
                }
            )
    return metrics


def run_one(
    variant: str,
    seed: int,
    partitions: dict[str, dict[str, Tensor]],
    args: argparse.Namespace,
) -> tuple[dict[str, Any], list[dict[str, float]]]:
    torch.manual_seed(seed)
    if args.device.type == "cuda":
        torch.cuda.manual_seed_all(seed)
    model = Model().to(args.device)
    gram_weight = 5.0 if "strong-gram" in variant else 0.05
    sigreg_weight = 0.0 if variant in ("random", "original") else 0.001
    floor_weight = 0.0 if variant.startswith("replace") else 0.1
    regularizer = SIGReg(num_slices=128, seed=seed).to(args.device)
    optimizer = torch.optim.AdamW(
        (parameter for parameter in model.parameters() if parameter.requires_grad),
        lr=0.001,
    )
    train_chunks = partitions["train"]["chunks"]
    context = train_chunks[:, :-1].reshape(-1, CHUNK_WIDTH)
    target = train_chunks[:, 1:].reshape(-1, CHUNK_WIDTH)
    sampler = torch.Generator(device=args.device).manual_seed(10000 + seed)
    factor_active = torch.zeros((), device=args.device)
    encoder_active = torch.zeros((), device=args.device)
    factor_raw_sum = torch.zeros((), device=args.device)
    encoder_raw_sum = torch.zeros((), device=args.device)
    history: list[dict[str, float]] = []
    started = time.perf_counter()
    if variant != "random":
        model.train()
        for step in range(1, args.steps + 1):
            indices = torch.randint(
                len(context), (256,), generator=sampler, device=args.device
            )
            context_state = model.online_encoder(context[indices])
            with torch.no_grad():
                target_state = model.target_encoder(target[indices])
            predicted = model.predict_factors(context_state)
            target_factors = model.projection.decompose(target_state)
            objective = jepa_anything_objective(
                predicted,
                target_factors,
                model.projection.analysis_basis(),
                context_state,
                orthogonality_weight=gram_weight,
                factor_activity_weight=floor_weight,
                encoder_variance_weight=floor_weight,
                sigreg_weight=sigreg_weight,
                sigreg=regularizer,
                sigreg_embeddings=context_state,
                factor_min_std=0.01,
                encoder_min_std=0.01,
            )
            factor_raw = objective.factor_activity.detach()
            encoder_raw = objective.encoder_variance.detach()
            factor_active += (factor_raw > 0).float()
            encoder_active += (encoder_raw > 0).float()
            factor_raw_sum += factor_raw
            encoder_raw_sum += encoder_raw
            optimizer.zero_grad(set_to_none=True)
            objective.total.backward()
            optimizer.step()
            model.projection.after_optimizer_step(step)
            ema_update(model.target_encoder, model.online_encoder, 0.996)
            if step == 1 or step % args.log_every == 0 or step == args.steps:
                row = {
                    "step": float(step),
                    "total": float(objective.total.detach()),
                    "prediction": float(objective.prediction.detach()),
                    "orthogonality": float(objective.orthogonality.detach()),
                    "factor_activity": float(factor_raw),
                    "encoder_variance": float(encoder_raw),
                    "sigreg": float(objective.sigreg.detach()),
                }
                history.append(row)
                print(
                    f"{variant} seed={seed} step={step} loss={row['total']:.5f}",
                    flush=True,
                )
    if args.device.type == "cuda":
        torch.cuda.synchronize()
    train_seconds = time.perf_counter() - started
    metrics = {
        "variant": variant,
        "seed": seed,
        "trainable_parameters": parameter_count(model),
        "steps": 0 if variant == "random" else args.steps,
        "gram_weight": gram_weight,
        "sigreg_weight": sigreg_weight,
        "floor_weight": floor_weight,
        "factor_floor_active_fraction": float(factor_active / args.steps),
        "encoder_floor_active_fraction": float(encoder_active / args.steps),
        "factor_floor_raw_mean": float(factor_raw_sum / args.steps),
        "encoder_floor_raw_mean": float(encoder_raw_sum / args.steps),
        "train_seconds": train_seconds,
        **evaluate(model, partitions, variant == "random"),
    }
    return metrics, history


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-dir", type=Path, default=Path("/workspace/datasets/uci-har")
    )
    parser.add_argument("--download-url", default=URL)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--variants", default=",".join(VARIANTS))
    parser.add_argument("--seeds", default="0,1,2")
    parser.add_argument("--steps", type=int, default=2000)
    parser.add_argument("--log-every", type=int, default=250)
    parser.add_argument("--smoke-data", action="store_true")
    args = parser.parse_args()
    variants = tuple(value.strip() for value in args.variants.split(","))
    seeds = tuple(int(value) for value in args.seeds.split(","))
    if (
        not variants
        or set(variants) - set(VARIANTS)
        or len(set(variants)) != len(variants)
    ):
        raise ValueError("invalid variants")
    if not seeds or min(seeds) < 0 or len(set(seeds)) != len(seeds):
        raise ValueError("invalid seeds")
    if args.steps < 1 or args.log_every < 1:
        raise ValueError("steps and log interval must be positive")
    args.device = torch.device(
        ("cuda" if torch.cuda.is_available() else "cpu")
        if args.device == "auto"
        else args.device
    )
    partitions, metadata = (
        smoke_data(args.device)
        if args.smoke_data
        else load_data(args.data_dir, args.download_url, args.device)
    )
    results: list[dict[str, Any]] = []
    histories: dict[str, list[dict[str, float]]] = {}
    payload: dict[str, Any] = {
        "protocol": "recipes/opf-uci-har/PROTOCOL.md",
        "configuration": {
            key: str(value) if isinstance(value, (Path, torch.device)) else value
            for key, value in vars(args).items()
        },
        "dataset": metadata,
        "environment": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "numpy": np.__version__,
            "device": torch.cuda.get_device_name(args.device)
            if args.device.type == "cuda"
            else "CPU",
        },
        "results": results,
        "history": histories,
    }
    output = args.output_dir / "results.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    for variant in variants:
        for seed in seeds:
            metrics, history = run_one(variant, seed, partitions, args)
            results.append(metrics)
            histories[f"{variant}/seed-{seed}"] = history
            temporary = output.with_suffix(".part")
            temporary.write_text(
                json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
            )
            temporary.replace(output)
            print(json.dumps(metrics, sort_keys=True), flush=True)
    print(f"WROTE {output}", flush=True)


if __name__ == "__main__":
    main()
