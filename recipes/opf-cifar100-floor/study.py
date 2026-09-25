"""Fixed real-image OPF test of naturally active 0.01 variance floors."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import platform
import sys
import time
import urllib.error
import urllib.request
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

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "sigreg-cifar10"))
from train import (
    DATASETS,
    augment,
    features,
    geometry,
    load_cifar,
    probe_accuracy,
)
from train import (
    Model as ReferenceModel,
)

VARIANTS = (
    "random",
    "original",
    "plus",
    "replace",
    "plus-strong-gram",
    "replace-strong-gram",
)
MIRROR = "https://data.brainchip.com/dataset-mirror/cifar100/cifar-100-binary.tar.gz"
SVHN_MIRROR = "https://cseweb.ucsd.edu/~weijian/static/datasets/svhn"
SVHN_FILES = {
    "train": ("train_32x32.mat", "e26dedcc434d2e4c54c9b2d4a06d8373"),
    "test": ("test_32x32.mat", "eb5a983be6a315427106f1b164d9cef3"),
}


def _file_md5(path: Path) -> str:
    digest = hashlib.md5()  # Dataset integrity checksum from torchvision.
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_svhn(data_dir: Path, mirror: str) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    import scipy.io  # type: ignore[import-untyped]

    data_dir.mkdir(parents=True, exist_ok=True)
    splits: list[tuple[Tensor, Tensor]] = []
    for name, (filename, expected_md5) in SVHN_FILES.items():
        path = data_dir / filename
        if not path.exists() or _file_md5(path) != expected_md5:
            temporary = path.with_suffix(".part")
            for attempt in range(12):
                offset = temporary.stat().st_size if temporary.exists() else 0
                headers = {"Range": f"bytes={offset}-"} if offset else {}
                request = urllib.request.Request(
                    f"{mirror.rstrip('/')}/{filename}", headers=headers
                )
                try:
                    with urllib.request.urlopen(request, timeout=45) as response:
                        if offset and response.status != 206:
                            offset = 0
                        with temporary.open("ab" if offset else "wb") as dest:
                            for chunk in iter(lambda: response.read(1 << 20), b""):
                                dest.write(chunk)
                except (OSError, urllib.error.URLError) as error:
                    print(f"{name} download retry {attempt + 1}: {error}", flush=True)
                    time.sleep(2)
                    continue
                break
            if not temporary.exists() or _file_md5(temporary) != expected_md5:
                raise RuntimeError(f"SVHN {name} file missing or checksum mismatch")
            temporary.replace(path)
        raw = scipy.io.loadmat(path)
        images = np.transpose(raw["X"], (3, 2, 0, 1)).copy()
        labels = raw["y"].astype(np.int64).reshape(-1) % 10
        if images.shape != (73257 if name == "train" else 26032, 3, 32, 32):
            raise RuntimeError(f"invalid SVHN {name} shape: {images.shape}")
        splits.append((torch.from_numpy(images), torch.from_numpy(labels)))
    return splits[0][0], splits[0][1], splits[1][0], splits[1][1]


def augment_svhn(images: Tensor, generator: torch.Generator) -> Tensor:
    """CIFAR-like crop/jitter without label-destroying horizontal flips."""
    batch = images.shape[0]
    padded = F.pad(images.float().div_(255), (4, 4, 4, 4), mode="reflect")
    top = torch.randint(0, 9, (batch,), device=images.device, generator=generator)
    left = torch.randint(0, 9, (batch,), device=images.device, generator=generator)
    offset = torch.arange(32, device=images.device)
    positions = (
        (top[:, None, None] + offset[None, :, None]) * 40
        + left[:, None, None]
        + offset[None, None, :]
    ).reshape(batch, 1, 1024)
    cropped = padded.flatten(2).gather(2, positions.expand(-1, 3, -1))
    cropped = cropped.reshape(batch, 3, 32, 32)
    brightness = 0.9 + 0.2 * torch.rand(
        batch, 1, 1, 1, device=images.device, generator=generator
    )
    contrast = 0.9 + 0.2 * torch.rand(
        batch, 1, 1, 1, device=images.device, generator=generator
    )
    mean = cropped.mean(dim=(2, 3), keepdim=True)
    return ((cropped - mean) * contrast + mean).mul_(brightness).clamp_(0, 1)


class OPFImageModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.online_encoder = ReferenceModel(64).encoder
        self.target_encoder = copy.deepcopy(self.online_encoder)
        self.target_encoder.requires_grad_(False)
        self.projection = OrthogonalFactorProjection(64, 8, 8, learnable=True)
        self.factor_predictors = nn.ModuleList(
            [
                nn.Sequential(nn.Linear(64, 256), nn.GELU(), nn.Linear(256, 8))
                for _ in range(8)
            ]
        )

    def train(self, mode: bool = True) -> OPFImageModel:
        super().train(mode)
        self.target_encoder.eval()
        return self

    def predict_factors(self, state: Tensor) -> Tensor:
        return torch.stack([head(state) for head in self.factor_predictors], dim=-2)


def _basis_rmse(projection: OrthogonalFactorProjection) -> float:
    basis = projection.analysis_basis().detach().double().cpu().reshape(64, 64)
    gram = basis @ basis.T
    return float((gram - torch.eye(64, dtype=torch.float64)).square().mean().sqrt())


def run_one(
    variant: str,
    seed: int,
    train_x: Tensor,
    train_y: Tensor,
    val_x: Tensor,
    val_y: Tensor,
    test_x: Tensor,
    test_y: Tensor,
    args: argparse.Namespace,
) -> tuple[dict[str, Any], list[dict[str, float]]]:
    torch.manual_seed(seed)
    if args.device.type == "cuda":
        torch.cuda.manual_seed_all(seed)
    model = OPFImageModel().to(args.device)
    sampler = torch.Generator(device=args.device).manual_seed(10000 + seed)
    regularizer = SIGReg(num_slices=128, seed=seed).to(args.device)
    optimizer = torch.optim.AdamW(
        (parameter for parameter in model.parameters() if parameter.requires_grad),
        lr=0.001,
    )
    gram_weight = 5.0 if "strong-gram" in variant else 0.05
    sigreg_weight = 0.0 if variant in ("random", "original") else args.sigreg_weight
    floor_weight = 0.0 if variant.startswith("replace") else 0.1
    factor_active = torch.zeros((), device=args.device)
    encoder_active = torch.zeros((), device=args.device)
    factor_raw_sum = torch.zeros((), device=args.device)
    encoder_raw_sum = torch.zeros((), device=args.device)
    updates = 0
    history: list[dict[str, float]] = []
    started = time.perf_counter()
    if variant != "random":
        for epoch in range(1, args.epochs + 1):
            model.train()
            order = torch.randperm(len(train_x), generator=sampler, device=args.device)
            epoch_loss = torch.zeros((), device=args.device)
            batches = 0
            for indices in order.split(256):
                if len(indices) < 2:
                    continue
                images = train_x[indices]
                augmentation = augment_svhn if args.dataset == "svhn" else augment
                context = model.online_encoder(augmentation(images, sampler))
                with torch.no_grad():
                    target = model.target_encoder(augmentation(images, sampler))
                predicted = model.predict_factors(context)
                target_factors = model.projection.decompose(target)
                objective = jepa_anything_objective(
                    predicted,
                    target_factors,
                    model.projection.analysis_basis(),
                    context,
                    orthogonality_weight=gram_weight,
                    factor_activity_weight=floor_weight,
                    encoder_variance_weight=floor_weight,
                    sigreg_weight=sigreg_weight,
                    sigreg=regularizer,
                    sigreg_embeddings=context,
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
                model.projection.after_optimizer_step(updates + 1)
                ema_update(model.target_encoder, model.online_encoder, 0.996)
                updates += 1
                batches += 1
                epoch_loss += objective.total.detach()
            row = {"epoch": float(epoch), "mean_loss": float(epoch_loss / batches)}
            history.append(row)
            print(
                f"{variant} seed={seed} epoch={epoch} loss={row['mean_loss']:.5f}",
                flush=True,
            )
    if args.device.type == "cuda":
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - started
    train_z = features(model.online_encoder, train_x, 512)
    val_z = features(model.online_encoder, val_x, 512)
    test_z = (
        None if args.validation_only else features(model.online_encoder, test_x, 512)
    )
    metrics: dict[str, Any] = {
        "variant": variant,
        "seed": seed,
        "updates": updates,
        "trainable_parameters": parameter_count(model),
        "sigreg_weight": sigreg_weight,
        "gram_weight": gram_weight,
        "floor_weight": floor_weight,
        "factor_floor_active_fraction": float(factor_active / max(updates, 1)),
        "encoder_floor_active_fraction": float(encoder_active / max(updates, 1)),
        "factor_floor_raw_mean": float(factor_raw_sum / max(updates, 1)),
        "encoder_floor_raw_mean": float(encoder_raw_sum / max(updates, 1)),
        "validation_probe_accuracy": probe_accuracy(
            train_z, train_y.cpu(), val_z, val_y.cpu(), args.num_classes
        ),
        "gram_rmse": _basis_rmse(model.projection),
        "train_seconds": elapsed,
    }
    if test_z is not None:
        metrics["test_probe_accuracy"] = probe_accuracy(
            train_z, train_y.cpu(), test_z, test_y.cpu(), args.num_classes
        )
        metrics.update(
            {f"test_{key}": value for key, value in geometry(test_z).items()}
        )
    metrics.update(
        {f"validation_{key}": value for key, value in geometry(val_z).items()}
    )
    return metrics, history


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("/workspace/datasets"))
    parser.add_argument("--download-url")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--dataset", choices=("cifar100", "svhn"), default="cifar100")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--epochs", type=int, default=16)
    parser.add_argument("--variants", default=",".join(VARIANTS))
    parser.add_argument("--seeds", default="0,1,2")
    parser.add_argument("--sigreg-weight", type=float, default=0.05)
    parser.add_argument("--validation-only", action="store_true")
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
    if not seeds or min(seeds) < 0 or len(set(seeds)) != len(seeds) or args.epochs < 1:
        raise ValueError("invalid seeds or epochs")
    if args.sigreg_weight < 0:
        raise ValueError("sigreg weight must be nonnegative")
    args.num_classes = 10 if args.dataset == "svhn" else 100
    if args.download_url is None:
        args.download_url = SVHN_MIRROR if args.dataset == "svhn" else MIRROR
    args.device = torch.device(
        ("cuda" if torch.cuda.is_available() else "cpu")
        if args.device == "auto"
        else args.device
    )
    if args.smoke_data:
        generator = torch.Generator().manual_seed(1729)
        all_x = torch.randint(
            0, 256, (96, 3, 32, 32), generator=generator, dtype=torch.uint8
        )
        all_y = torch.arange(96) % args.num_classes
        train_x, val_x, test_x = all_x[:64], all_x[64:80], all_x[80:]
        train_y, val_y, test_y = all_y[:64], all_y[64:80], all_y[80:]
        archive_md5: str | dict[str, str] = "synthetic-smoke"
    else:
        if args.dataset == "svhn":
            all_x, all_y, test_x, test_y = load_svhn(args.data_dir, args.download_url)
            order = torch.randperm(
                len(all_x), generator=torch.Generator().manual_seed(20260926)
            )
            train_indices, validation_indices = order[:65000], order[65000:]
            train_x, val_x = all_x[train_indices], all_x[validation_indices]
            train_y, val_y = all_y[train_indices], all_y[validation_indices]
            archive_md5 = {key: value[1] for key, value in SVHN_FILES.items()}
        else:
            all_x, all_y, test_x, test_y = load_cifar(
                args.data_dir, args.download_url, DATASETS["cifar100"]
            )
            train_x, val_x = all_x[:45000], all_x[45000:]
            train_y, val_y = all_y[:45000], all_y[45000:]
            archive_md5 = DATASETS["cifar100"].md5
    train_x = train_x.to(args.device)
    val_x = val_x.to(args.device)
    test_x = test_x.to(args.device)
    output = args.output_dir / "results.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    results: list[dict[str, Any]] = []
    histories: dict[str, list[dict[str, float]]] = {}
    payload: dict[str, Any] = {
        "protocol": "recipes/opf-cifar100-floor/PROTOCOL.md",
        "configuration": {
            key: str(value) if isinstance(value, (Path, torch.device)) else value
            for key, value in vars(args).items()
        },
        "dataset": {
            "archive_md5": archive_md5,
            "download_url": None if args.smoke_data else args.download_url,
            "train_count": len(train_x),
            "validation_count": len(val_x),
            "test_count": len(test_x),
            "labels_used_for_ssl": False,
            "test_probe_evaluated": not args.validation_only,
            "validation_split_seed": 20260926 if args.dataset == "svhn" else None,
        },
        "environment": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "device": torch.cuda.get_device_name(args.device)
            if args.device.type == "cuda"
            else "CPU",
        },
        "results": results,
        "history": histories,
    }
    for variant in variants:
        for seed in seeds:
            metrics, history = run_one(
                variant,
                seed,
                train_x,
                train_y,
                val_x,
                val_y,
                test_x,
                test_y,
                args,
            )
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
