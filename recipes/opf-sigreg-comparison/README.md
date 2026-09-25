# Original OPF objective versus SIGReg

This is a runnable, controlled comparison based on the public
[`synthetic-linear-dynamics` recipe](../synthetic-linear-dynamics/recipe.json)
and `jepa-anything-core` objective. It is **not** a reproduction of the
JEPA-Anything paper's seven-domain performance results: the public repository
does not ship those training pipelines or model weights.

The generator matches the source recipe's state transition, control, noisy
observation, seed, and trajectory split. It also matches the original generator's
SHA-256 fingerprint of the full generated stream. The model uses separate
online/EMA target encoders, a learned OPF analysis basis, two factor-specific
prediction pathways, and pseudoinverse synthesis for latent rollouts. The
source recipe is a *structural fixture*, not an executable training model; the
one-step encoder, factor-head MLP, optimizer, and EMA schedule below are explicit
implementation choices shared unchanged by every comparison arm. In particular,
the source design's eight-observation context window and multiple simultaneous
target offsets are simplified to one-step action-conditioned training, with
1/2/4/8-step rollouts as evaluation.

## Fixed comparison

| Arm | Prediction | Projector Gram | Factor activity | Online variance | SIGReg |
|---|---:|---:|---:|---:|---:|
| `original` | 1 | 0.05 | 0.1 | 0.1 | 0 |
| `original-plus-sigreg` | 1 | 0.05 | 0.1 | 0.1 | 0.005 |
| `sigreg-replaces-floors` | 1 | 0.05 | 0 | 0 | 0.005 |

The first four weights and both standard-deviation thresholds (`0.01`) come
from the source recipe. The SIGReg weight (`0.005`, 256 slices) comes from the
earlier synthetic weight sweep, not this test set. All arms have exactly the
same architecture and 12,012 trainable parameters, within the source fixture's
12,000 ±0.3% plan. The same trajectories, normalization, initialization seed,
batch draws, 1,000 updates, AdamW rate, and EMA momentum are used per seed.
The original recipe's planned predictor-FLOP target is not claimed to be
reproduced; this comparison matches FLOPs *between its three arms* by sharing
the exact predictor implementation.

The [protocol](PROTOCOL.md) was committed before the full experiment.
The [completed 3-seed results and interpretation](../../results/opf-sigreg-comparison-2026-09-25/REPORT.md)
are available separately from the raw metrics.
The [pre-registered geometry/floor/external-stream follow-up](FOLLOWUP_PROTOCOL.md)
and its [results](../../results/opf-sigreg-followup-2026-09-25/REPORT.md)
extend the comparison without reusing the viewed test set for selection.

```bash
PYTHONPATH=jepa-anything-core/src python3 \
  recipes/opf-sigreg-comparison/train.py \
  --device cpu --output-dir results/opf-sigreg-comparison-2026-09-25
```

The 16 training trajectories are used for unsupervised learning and for fitting
one fixed ridge linear probe from representations to simulator state. Four
validation trajectories are reserved and not used to select settings; four
held-out test trajectories supply current-state probe R² and open-loop rollout
R² at horizons 1, 2, 4, and 8. The simulator's true state is never used by the
representation-learning loss.
