"""Sketched Isotropic Gaussian Regularization (SIGReg).

The implementation follows the LeJEPA construction: random unit-vector slices
reduce a multivariate embedding distribution to one dimension, then an
Epps--Pulley characteristic-function statistic compares every slice with a
standard normal distribution.
"""

from __future__ import annotations

import math
from typing import Literal

import torch
import torch.distributed as dist
from torch import Tensor, nn

SIGRegReduction = Literal["mean", "sum", "none"]


def _validate_configuration(
    *,
    num_slices: int,
    num_points: int,
    t_max: float,
    eps: float,
) -> None:
    if num_slices <= 0:
        raise ValueError("num_slices must be positive")
    if num_points < 3 or num_points % 2 == 0:
        raise ValueError("num_points must be an odd integer greater than or equal to 3")
    if not math.isfinite(t_max) or t_max <= 0:
        raise ValueError("t_max must be finite and positive")
    if not math.isfinite(eps) or eps <= 0:
        raise ValueError("eps must be finite and positive")


def _validate_embeddings(embeddings: Tensor) -> Tensor:
    if embeddings.ndim < 2:
        raise ValueError(
            "embeddings must have shape (..., num_samples, embedding_dim), got "
            f"{tuple(embeddings.shape)}"
        )
    if not embeddings.is_floating_point():
        raise TypeError("embeddings must be floating point")
    if embeddings.shape[-2] <= 0:
        raise ValueError("SIGReg requires at least one sample")
    if embeddings.shape[-1] <= 0:
        raise ValueError("embedding_dim must be positive")
    if not torch.isfinite(embeddings).all():
        raise ValueError("embeddings must contain only finite values")
    if embeddings.dtype in (torch.float16, torch.bfloat16):
        return embeddings.float()
    return embeddings


def _distributed_sums_and_count(values: Tensor) -> tuple[Tensor, Tensor]:
    """Sum over samples, preserving gradients across distributed workers."""

    local_sum = values.sum(dim=-3)
    count = torch.tensor(
        values.shape[-3],
        device=values.device,
        dtype=values.dtype,
    )
    if not (dist.is_available() and dist.is_initialized()):
        return local_sum, count

    # torch.distributed.nn.functional keeps the all-reduce in the autograd graph.
    from torch.distributed.nn.functional import all_reduce

    global_sum = all_reduce(local_sum, op=dist.ReduceOp.SUM)
    dist.all_reduce(count, op=dist.ReduceOp.SUM)
    return global_sum, count


def epps_pulley_statistic(
    projected_samples: Tensor,
    *,
    t_max: float = 3.0,
    num_points: int = 17,
) -> Tensor:
    """Return one Epps--Pulley statistic per projected coordinate.

    Args:
        projected_samples: Tensor shaped ``(..., N, S)`` where ``N`` is the
            local sample count and ``S`` is the number of one-dimensional
            slices. Distributed workers contribute to a global statistic when
            ``torch.distributed`` is initialized.
        t_max: Positive endpoint of the symmetric quadrature interval.
        num_points: Odd number of trapezoid nodes on ``[0, t_max]``.

    Returns:
        A tensor shaped ``(..., S)``. The statistic is non-negative and is
        scaled by the global sample count, matching the LeJEPA formulation.
    """

    _validate_configuration(
        num_slices=1,
        num_points=num_points,
        t_max=t_max,
        eps=1e-12,
    )
    samples = _validate_embeddings(projected_samples)
    t = torch.linspace(0.0, t_max, num_points, device=samples.device, dtype=samples.dtype)
    dt = t_max / (num_points - 1)
    quadrature = torch.full_like(t, 2.0 * dt)
    quadrature[0] = dt
    quadrature[-1] = dt
    normal_cf = torch.exp(-0.5 * t.square())
    weights = quadrature * normal_cf

    phases = samples.unsqueeze(-1) * t
    cosine_sum, global_count = _distributed_sums_and_count(phases.cos())
    sine_sum, sine_count = _distributed_sums_and_count(phases.sin())
    if not torch.equal(global_count, sine_count):
        raise RuntimeError("distributed SIGReg sample counts diverged")
    empirical_cosine = cosine_sum / global_count
    empirical_sine = sine_sum / global_count
    discrepancy = (empirical_cosine - normal_cf).square() + empirical_sine.square()
    return (discrepancy @ weights) * global_count


class SIGReg(nn.Module):
    """Sliced Epps--Pulley regularizer toward ``N(0, I)`` embeddings.

    The input is ``(..., N, D)``. Prefix dimensions may represent views or
    independent groups; ``N`` is the sample axis and ``D`` is the embedding
    dimension. Random directions are deterministic from ``seed + step`` and the
    checkpointed ``step`` buffer advances once per forward call.
    """

    step: Tensor

    def __init__(
        self,
        *,
        num_slices: int = 256,
        num_points: int = 17,
        t_max: float = 3.0,
        reduction: SIGRegReduction = "mean",
        seed: int = 0,
        eps: float = 1e-12,
    ) -> None:
        super().__init__()
        _validate_configuration(
            num_slices=num_slices,
            num_points=num_points,
            t_max=t_max,
            eps=eps,
        )
        if reduction not in ("mean", "sum", "none"):
            raise ValueError(f"unsupported reduction: {reduction!r}")
        if seed < 0:
            raise ValueError("seed must be non-negative")
        self.num_slices = num_slices
        self.num_points = num_points
        self.t_max = t_max
        self.reduction = reduction
        self.seed = seed
        self.eps = eps
        self.register_buffer("step", torch.zeros((), dtype=torch.int64))

    def reset(self) -> None:
        """Reset the deterministic projection sequence to its first step."""

        self.step.zero_()

    def _synchronized_step(self) -> int:
        synchronized = self.step.detach().clone()
        if dist.is_available() and dist.is_initialized():
            dist.all_reduce(synchronized, op=dist.ReduceOp.MAX)
            self.step.copy_(synchronized)
        return int(synchronized.item())

    def _directions(self, embeddings: Tensor) -> Tensor:
        generator = torch.Generator(device=embeddings.device)
        generator.manual_seed(self.seed + self._synchronized_step())
        directions = torch.randn(
            embeddings.shape[-1],
            self.num_slices,
            device=embeddings.device,
            dtype=embeddings.dtype,
            generator=generator,
        )
        norms = directions.norm(p=2, dim=0, keepdim=True).clamp_min(self.eps)
        self.step.add_(1)
        return directions / norms

    def forward(self, embeddings: Tensor) -> Tensor:
        samples = _validate_embeddings(embeddings)
        directions = self._directions(samples)
        statistics = epps_pulley_statistic(
            samples @ directions,
            t_max=self.t_max,
            num_points=self.num_points,
        )
        if self.reduction == "mean":
            return statistics.mean()
        if self.reduction == "sum":
            return statistics.sum()
        return statistics

    def extra_repr(self) -> str:
        return (
            f"num_slices={self.num_slices}, num_points={self.num_points}, "
            f"t_max={self.t_max}, reduction={self.reduction!r}, seed={self.seed}"
        )
