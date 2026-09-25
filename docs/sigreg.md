# SIGReg integration

JEPA Anything now includes Sketched Isotropic Gaussian Regularization (SIGReg),
the anti-collapse objective introduced by LeJEPA. SIGReg samples random unit
directions, projects multivariate embeddings to one dimension, and applies an
Epps--Pulley characteristic-function statistic against a standard normal. By
the Cramér--Wold principle, matching many one-dimensional slices constrains the
joint representation distribution toward `N(0, I)`.

## Core API

```python
from jepa_anything_core import SIGReg, jepa_anything_objective

regularizer = SIGReg(num_slices=256, num_points=17, seed=0)
losses = jepa_anything_objective(
    predicted_factors,
    target_factors,
    analysis_basis,
    context_states,
    factor_activity_weight=0.0,
    encoder_variance_weight=0.0,
    sigreg_weight=0.02,
    sigreg=regularizer,
    sigreg_embeddings=projected_embeddings,
)
losses.total.backward()
```

If `sigreg_embeddings` is omitted, the objective flattens the target factors
back to their complete `d = K * r` state. Passing it explicitly is preferable
when the online encoder/projector output is the intended regularization site.
The input must have shape `(..., N, d)`, or `(N, d)` for a single group.

The module promotes fp16/bfloat16 reductions to fp32, checkpoints its random
projection step, and combines characteristic-function sums across initialized
distributed workers. Set `sigreg_weight=0` to disable the term without advancing
the projection sequence.

## Relationship to existing objectives

The pre-existing factor-activity and encoder-variance losses remain available
for backward compatibility and ablation. SIGReg is stronger: a coordinate
variance floor only checks marginal scale, while SIGReg also penalizes shifted,
anisotropic, and non-Gaussian joint embeddings through random slices. Projector
Gram orthogonality is separate and still governs OPF basis geometry.

## References

- Randall Balestriero and Yann LeCun, *LeJEPA: Provable and Scalable
  Self-Supervised Learning Without the Heuristics*, arXiv:2511.08544.
- Official reference implementation: <https://github.com/galilai-group/lejepa>
