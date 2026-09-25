"""JEPA-style CIFAR-10 test of SIGReg and coordinate variance floors."""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import tarfile
import time
import urllib.request
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

URL = "https://www.cs.toronto.edu/~kriz/cifar-10-binary.tar.gz"
MD5 = "c32a1d4ab5d03f1284b67883e8d87530"
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


def load_cifar(data_dir: Path, download_url: str) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    data_dir.mkdir(parents=True, exist_ok=True)
    archive = data_dir / "cifar-10-binary.tar.gz"
    if not archive.exists() or file_md5(archive) != MD5:
        temporary = data_dir / "cifar-10-binary.part"
        with urllib.request.urlopen(download_url, timeout=120) as response, temporary.open("wb") as dest:
            for chunk in iter(lambda: response.read(1 << 20), b""):
                dest.write(chunk)
        if file_md5(temporary) != MD5:
            raise RuntimeError("CIFAR-10 archive checksum mismatch")
        temporary.replace(archive)

    def split(bundle: tarfile.TarFile, names: list[str]) -> tuple[Tensor, Tensor]:
        images: list[np.ndarray] = []
        labels: list[np.ndarray] = []
        for name in names:
            stream = bundle.extractfile(f"cifar-10-batches-bin/{name}")
            if stream is None:
                raise RuntimeError(f"missing CIFAR-10 batch {name}")
            raw = stream.read()
            if len(raw) != 10000 * 3073:
                raise RuntimeError(f"invalid CIFAR-10 batch {name}")
            records = np.frombuffer(raw, dtype=np.uint8).reshape(10000, 3073)
            labels.append(records[:, 0].copy())
            images.append(records[:, 1:].copy().reshape(10000, 3, 32, 32))
        return torch.from_numpy(np.concatenate(images)), torch.from_numpy(np.concatenate(labels)).long()

    with tarfile.open(archive, "r:gz") as bundle:
        train_x, train_y = split(bundle, [f"data_batch_{i}.bin" for i in range(1, 6)])
        test_x, test_y = split(bundle, ["test_batch.bin"])
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


def probe_accuracy(train: Tensor, train_y: Tensor, test: Tensor, test_y: Tensor) -> float:
    train = train.double()
    mean = train.mean(dim=0)
    scale = train.std(dim=0).clamp_min(1e-4)
    x = (train - mean) / scale
    z = (test.double() - mean) / scale
    labels = F.one_hot(train_y, num_classes=10).double()
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
            train_features, train_y.cpu(), test_features, test_y.cpu()
        ),
        "train_seconds": elapsed,
        **geometry(test_features),
    }
    return result, history


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("/workspace/datasets"))
    parser.add_argument("--download-url", default=URL)
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
        train_y = torch.randint(0, 10, (64,), generator=generator)
        test_x = torch.randint(0, 256, (32, 3, 32, 32), generator=generator, dtype=torch.uint8)
        test_y = torch.randint(0, 10, (32,), generator=generator)
        checksum = "synthetic-smoke"
    else:
        train_x, train_y, test_x, test_y = load_cifar(args.data_dir, args.download_url)
        checksum = MD5
    train_x = train_x.to(device)
    test_x = test_x.to(device)
    results = []
    histories = {}
    for variant in variants:
        for seed in seeds:
            result, history = run(variant, seed, train_x, train_y, test_x, test_y, args)
            results.append(result)
            histories[f"{variant}/seed-{seed}"] = history
            print(json.dumps(result, sort_keys=True), flush=True)
    args.output_dir.mkdir(parents=True, exist_ok=True)
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
    output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"WROTE {output}", flush=True)


if __name__ == "__main__":
    main()
