from __future__ import annotations

import pytest
import torch

from jepa_anything_core import SIGReg, epps_pulley_statistic


def test_epps_pulley_distinguishes_standard_and_shifted_normals() -> None:
    generator = torch.Generator().manual_seed(12)
    standard = torch.randn(4096, 8, generator=generator)
    shifted = standard + 3.0

    standard_statistic = epps_pulley_statistic(standard)
    shifted_statistic = epps_pulley_statistic(shifted)

    assert standard_statistic.shape == (8,)
    assert torch.all(standard_statistic >= 0)
    assert shifted_statistic.mean() > 100 * standard_statistic.mean()


def test_sigreg_penalizes_collapsed_embeddings_and_propagates_gradients() -> None:
    generator = torch.Generator().manual_seed(4)
    gaussian = torch.randn(2048, 6, generator=generator, requires_grad=True)
    collapsed = torch.zeros_like(gaussian, requires_grad=True)
    gaussian_loss = SIGReg(num_slices=256, seed=9)(gaussian)
    collapsed_loss = SIGReg(num_slices=256, seed=9)(collapsed)

    gaussian_loss.backward()
    collapsed_loss.backward()

    assert collapsed_loss > 20 * gaussian_loss
    assert gaussian.grad is not None and torch.isfinite(gaussian.grad).all()
    assert collapsed.grad is not None and torch.isfinite(collapsed.grad).all()


def test_sigreg_projection_sequence_is_checkpointable_and_resettable() -> None:
    embeddings = torch.randn(128, 5, generator=torch.Generator().manual_seed(3))
    first = SIGReg(num_slices=32, seed=21)
    second = SIGReg(num_slices=32, seed=21)

    torch.testing.assert_close(first(embeddings), second(embeddings))
    checkpoint = {name: value.clone() for name, value in first.state_dict().items()}
    expected_next = first(embeddings)
    second.load_state_dict(checkpoint)
    torch.testing.assert_close(expected_next, second(embeddings))

    first.reset()
    fresh = SIGReg(num_slices=32, seed=21)
    torch.testing.assert_close(first(embeddings), fresh(embeddings))


def test_none_reduction_preserves_prefix_and_slice_dimensions() -> None:
    embeddings = torch.randn(3, 64, 7)
    statistics = SIGReg(num_slices=11, reduction="none")(embeddings)

    assert statistics.shape == (3, 11)


def test_low_precision_inputs_accumulate_in_float32() -> None:
    embeddings = torch.randn(64, 4).half()
    loss = SIGReg(num_slices=16)(embeddings)

    assert loss.dtype == torch.float32
    assert torch.isfinite(loss)


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"num_slices": 0}, "num_slices"),
        ({"num_points": 16}, "num_points"),
        ({"t_max": 0.0}, "t_max"),
        ({"reduction": "median"}, "reduction"),
        ({"seed": -1}, "seed"),
    ],
)
def test_sigreg_rejects_invalid_configuration(kwargs: dict[str, object], message: str) -> None:
    with pytest.raises(ValueError, match=message):
        SIGReg(**kwargs)  # type: ignore[arg-type]


def test_sigreg_rejects_invalid_embeddings() -> None:
    regularizer = SIGReg(num_slices=4)
    with pytest.raises(ValueError, match="shape"):
        regularizer(torch.ones(4))
    with pytest.raises(TypeError, match="floating point"):
        regularizer(torch.ones(4, 2, dtype=torch.int64))
    with pytest.raises(ValueError, match="finite"):
        regularizer(torch.tensor([[float("nan"), 0.0]]))
