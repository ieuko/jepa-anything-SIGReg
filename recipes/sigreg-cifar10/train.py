"""JEPA-style CIFAR-10/100 test of SIGReg and coordinate variance floors."""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import tarfile
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from jepa_anything_core import (
    OrthogonalFactorProjection,
    SIGReg,
    factor_prediction_loss,
    jepa_anything_objective,
)
from torch import Tensor, nn


@dataclass(frozen=True)
class DatasetSpec:
    url: str
    md5: str
    archive_bytes: int
    archive_name: str
    prefix: str
    train_batches: tuple[tuple[str, int], ...]
    test_batches: tuple[tuple[str, int], ...]
    label_index: int
    num_classes: int


DATASETS = {
    "cifar10": DatasetSpec(
        "https://www.cs.toronto.edu/~kriz/cifar-10-binary.tar.gz",
        "c32a1d4ab5d03f1284b67883e8d87530", 170052171,
        "cifar-10-binary.tar.gz", "cifar-10-batches-bin",
        tuple((f"data_batch_{i}.bin", 10000) for i in range(1, 6)),
        (("test_batch.bin", 10000),), 0, 10,
    ),
    "cifar100": DatasetSpec(
        "https://www.cs.toronto.edu/~kriz/cifar-100-binary.tar.gz",
        "03b5dce01913d631647c71ecec9e9cb8", 168513733,
        "cifar-100-binary.tar.gz", "cifar-100-binary",
        (("train.bin", 50000),), (("test.bin", 10000),), 1, 100,
    ),
}
VARIANTS = ("random", "prediction-only", "variance", "sigreg")


class Model(nn.Module):
    def __init__(self, width: int) -> None:
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 64, 3, 2, 1), nn.GELU(),
            nn.Conv2d(64, 128, 3, 2, 1), nn.GELU(),
            nn.Conv2d(128, 256, 3, 2, 1), nn.GELU(),
            nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(256, width),
        )
        self.predictor = nn.Sequential(
            nn.Linear(width, 256), nn.GELU(), nn.Linear(256, width)
        )


def file_md5(path: Path) -> str:
    digest = hashlib.md5()  # Publisher-provided integrity checksum, not cryptographic trust.
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_batch(raw: bytes, count: int, spec: DatasetSpec) -> tuple[Tensor, Tensor]:
    record_bytes = 3072 + spec.label_index + 1
    if len(raw) != count * record_bytes:
        raise RuntimeError(f"invalid CIFAR batch: expected {count * record_bytes} bytes")
    records = np.frombuffer(raw, dtype=np.uint8).reshape(count, record_bytes)
    labels = torch.from_numpy(records[:, spec.label_index].copy()).long()
    images = torch.from_numpy(records[:, -3072:].copy().reshape(count, 3, 32, 32))
    if (labels >= spec.num_classes).any():
        raise RuntimeError("CIFAR label out of range")
    return images, labels


def load_cifar(
    data_dir: Path, download_url: str, spec: DatasetSpec,
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    data_dir.mkdir(parents=True, exist_ok=True)
    archive = data_dir / spec.archive_name
    if not archive.exists() or file_md5(archive) != spec.md5:
        temporary = archive.with_suffix(".part")
        for attempt in range(12):
            offset = temporary.stat().st_size if temporary.exists() else 0
            if offset == spec.archive_bytes:
                break
            headers = {"Range": f"bytes={offset}-"} if offset else {}
            request = urllib.request.Request(download_url, headers=headers)
            try:
                with urllib.request.urlopen(request, timeout=45) as response:
                    if offset and response.status != 206:
                        offset = 0
                    mode = "ab" if offset else "wb"
                    with temporary.open(mode) as dest:
                        for chunk in iter(lambda: response.read(1 << 20), b""):
                            dest.write(chunk)
            except (OSError, urllib.error.URLError) as error:
                print(f"download retry {attempt + 1}: {error}", flush=True)
                time.sleep(2)
        if not temporary.exists() or temporary.stat().st_size != spec.archive_bytes:
            raise RuntimeError("CIFAR download did not complete after retries")
        if file_md5(temporary) != spec.md5:
            raise RuntimeError("CIFAR archive checksum mismatch")
        temporary.replace(archive)

    def split(
        bundle: tarfile.TarFile, batches: tuple[tuple[str, int], ...],
    ) -> tuple[Tensor, Tensor]:
        images: list[Tensor] = []
        labels: list[Tensor] = []
        for name, count in batches:
            stream = bundle.extractfile(f"{spec.prefix}/{name}")
            if stream is None:
                raise RuntimeError(f"missing CIFAR batch {name}")
            batch_images, batch_labels = parse_batch(stream.read(), count, spec)
            images.append(batch_images)
            labels.append(batch_labels)
        return torch.cat(images), torch.cat(labels)

    with tarfile.open(archive, "r:gz") as bundle:
        train_x, train_y = split(bundle, spec.train_batches)
        test_x, test_y = split(bundle, spec.test_batches)
    return train_x, train_y, test_x, test_y


def augment(images: Tensor, generator: torch.Generator) -> Tensor:
    batch = images.shape[0]
    padded = F.pad(images.float().div_(255), (4, 4, 4, 4), mode="reflect")
    top = torch.randint(0, 9, (batch,), device=images.device, generator=generator)
    left = torch.randint(0, 9, (batch,), device=images.device, generator=generator)
    offset = torch.arange(32, device=images.device)
    positions = (
        (top[:, None, None] + offset[None, :, None]) * 40
        + left[:, None, None] + offset[None, None, :]
    ).reshape(batch, 1, 1024)
    cropped = padded.flatten(2).gather(2, positions.expand(-1, 3, -1))
    cropped = cropped.reshape(batch, 3, 32, 32)
    flipped = torch.rand(batch, device=images.device, generator=generator) < 0.5
    cropped = torch.where(flipped[:, None, None, None], cropped.flip(-1), cropped)
    brightness = 0.9 + 0.2 * torch.rand(
        batch, 1, 1, 1, device=images.device, generator=generator
    )
    contrast = 0.9 + 0.2 * torch.rand(
        batch, 1, 1, 1, device=images.device, generator=generator
    )
    mean = cropped.mean(dim=(2, 3), keepdim=True)
    return ((cropped - mean) * contrast + mean).mul_(brightness).clamp_(0, 1)


def features(encoder: nn.Module, images: Tensor, batch_size: int) -> Tensor:
    encoder.eval()
    with torch.inference_mode():
        return torch.cat([
            encoder(batch.float().div(255)).cpu() for batch in images.split(batch_size)
        ])


def probe_accuracy(
    train: Tensor, train_y: Tensor, test: Tensor, test_y: Tensor, num_classes: int,
) -> float:
    train = train.double()
    mean = train.mean(dim=0)
    scale = train.std(dim=0).clamp_min(1e-4)
    x = (train - mean) / scale
    z = (test.double() - mean) / scale
    labels = F.one_hot(train_y, num_classes=num_classes).double()
    prior = labels.mean(dim=0)
    ridge = 1e-3 * len(train_y)
    coefficients = torch.linalg.solve(
        x.T @ x + ridge * torch.eye(x.shape[1], dtype=x.dtype),
        x.T @ (labels - prior),
    )
    return float(((z @ coefficients + prior).argmax(dim=1) == test_y).double().mean())


def geometry(embeddings: Tensor) -> dict[str, float]:
    centered = embeddings.double() - embeddings.double().mean(dim=0)
    covariance = centered.T @ centered / len(centered)
    std = covariance.diag().clamp_min(0).sqrt()
    eigenvalues = torch.linalg.eigvalsh(covariance).clamp_min(0)
    probabilities = eigenvalues / eigenvalues.sum().clamp_min(1e-12)
    entropy = -(probabilities * probabilities.clamp_min(1e-12).log()).sum()
    identity = torch.eye(covariance.shape[0], dtype=covariance.dtype)
    return {
        "mean_coordinate_std": float(std.mean()),
        "min_coordinate_std": float(std.min()),
        "collapsed_coordinate_fraction": float((std < 0.1).double().mean()),
        "effective_rank": float(entropy.exp()),
        "covariance_identity_rmse": float((covariance - identity).square().mean().sqrt()),
    }


def run(
    variant: str, seed: int, train_x: Tensor, train_y: Tensor,
    test_x: Tensor, test_y: Tensor, args: argparse.Namespace,
) -> tuple[dict[str, float | str | int], list[dict[str, float]]]:
    torch.manual_seed(seed)
    if train_x.is_cuda:
        torch.cuda.manual_seed_all(seed)
    generator = torch.Generator(device=train_x.device).manual_seed(10000 + seed)
    model = Model(args.state_dim).to(train_x.device)
    projection = OrthogonalFactorProjection(
        args.state_dim, args.num_factors, args.state_dim // args.num_factors,
        learnable=False,
    ).to(train_x.device)
    regularizer = SIGReg(num_slices=args.num_slices, seed=seed).to(train_x.device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)
    history: list[dict[str, float]] = []
    started = time.perf_counter()
    if variant != "random":
        for epoch in range(args.epochs):
            model.train()
            order = torch.randperm(len(train_x), generator=generator, device=train_x.device)
            loss_sum = torch.zeros((), device=train_x.device)
            batches = 0
            for indices in order.split(args.batch_size):
                if len(indices) < 2:
                    continue
                images = train_x[indices]
                context = model.encoder(augment(images, generator))
                target = model.encoder(augment(images, generator))
                prediction = model.predictor(context)
                predicted_factors = projection.decompose(prediction)
                target_factors = projection.decompose(target)
                if variant == "prediction-only":
                    loss = factor_prediction_loss(predicted_factors, target_factors)
                else:
                    sigreg = variant == "sigreg"
                    objective = jepa_anything_objective(
                        predicted_factors, target_factors, projection.analysis_basis(), context,
                        orthogonality_weight=0.0,
                        factor_activity_weight=0.0 if sigreg else args.variance_weight,
                        encoder_variance_weight=0.0 if sigreg else args.variance_weight,
                        sigreg_weight=args.sigreg_weight if sigreg else 0.0,
                        sigreg=regularizer,
                        sigreg_embeddings=torch.cat((context, target), dim=0),
                        factor_min_std=1.0, encoder_min_std=1.0,
                    )
                    loss = objective.total
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
                loss_sum += loss.detach()
                batches += 1
            history.append({"epoch": float(epoch + 1), "loss": float(loss_sum / batches)})
            print(f"{variant} seed={seed} epoch={epoch+1} loss={history[-1]['loss']:.5f}", flush=True)
    if train_x.is_cuda:
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - started
    train_features = features(model.encoder, train_x, args.eval_batch_size)
    test_features = features(model.encoder, test_x, args.eval_batch_size)
    result: dict[str, float | str | int] = {
        "variant": variant, "seed": seed,
        "linear_probe_accuracy": probe_accuracy(
            train_features, train_y.cpu(), test_features, test_y.cpu(), args.num_classes
        ),
        "train_seconds": elapsed,
        **geometry(test_features),
    }
    return result, history


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=tuple(DATASETS), default="cifar10")
    parser.add_argument("--data-dir", type=Path, default=Path("/workspace/datasets"))
    parser.add_argument("--download-url")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--variants", default=",".join(VARIANTS))
    parser.add_argument("--seeds", default="0,1,2")
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--eval-batch-size", type=int, default=512)
    parser.add_argument("--state-dim", type=int, default=64)
    parser.add_argument("--num-factors", type=int, default=8)
    parser.add_argument("--num-slices", type=int, default=128)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--variance-weight", type=float, default=4.0)
    parser.add_argument("--sigreg-weight", type=float, default=0.005)
    parser.add_argument("--smoke-data", action="store_true")
    args = parser.parse_args()
    spec = DATASETS[args.dataset]
    args.num_classes = spec.num_classes
    if args.download_url is None:
        args.download_url = spec.url
    variants = [value.strip() for value in args.variants.split(",")]
    seeds = [int(value) for value in args.seeds.split(",")]
    if not variants or any(value not in VARIANTS for value in variants):
        raise ValueError("invalid variant")
    if not seeds or args.epochs <= 0 or args.batch_size < 2:
        raise ValueError("invalid seeds, epochs, or batch size")
    if args.state_dim % args.num_factors:
        raise ValueError("state_dim must be divisible by num_factors")
    device = torch.device(
        ("cuda" if torch.cuda.is_available() else "cpu") if args.device == "auto"
        else args.device
    )
    if args.smoke_data:
        generator = torch.Generator().manual_seed(123)
        train_x = torch.randint(0, 256, (64, 3, 32, 32), generator=generator, dtype=torch.uint8)
        train_y = torch.randint(0, spec.num_classes, (64,), generator=generator)
        test_x = torch.randint(0, 256, (32, 3, 32, 32), generator=generator, dtype=torch.uint8)
        test_y = torch.randint(0, spec.num_classes, (32,), generator=generator)
        checksum = "synthetic-smoke"
    else:
        train_x, train_y, test_x, test_y = load_cifar(args.data_dir, args.download_url, spec)
        checksum = spec.md5
    train_x = train_x.to(device)
    test_x = test_x.to(device)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    results: list[dict[str, float | str | int]] = []
    histories: dict[str, list[dict[str, float]]] = {}
    payload = {
        "configuration": {
            key: str(value) if isinstance(value, Path) else value
            for key, value in vars(args).items()
        },
        "dataset": {
            "url": args.download_url if not args.smoke_data else None,
            "archive_md5": checksum,
            "train_count": len(train_x),
            "test_count": len(test_x),
            "labels_used_for_pretraining": False,
        },
        "environment": {
            "device": str(device),
            "device_name": torch.cuda.get_device_name(device) if device.type == "cuda" else "CPU",
            "torch_version": torch.__version__,
            "cuda_version": torch.version.cuda,
            "python_version": platform.python_version(),
        },
        "results": results,
        "history": histories,
    }
    output = args.output_dir / "results.json"
    for variant in variants:
        for seed in seeds:
            result, history = run(variant, seed, train_x, train_y, test_x, test_y, args)
            results.append(result)
            histories[f"{variant}/seed-{seed}"] = history
            temporary = output.with_suffix(".part")
            temporary.write_text(
                json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
            )
            temporary.replace(output)
            print(json.dumps(result, sort_keys=True), flush=True)
    print(f"WROTE {output}", flush=True)


if __name__ == "__main__":
    main()
