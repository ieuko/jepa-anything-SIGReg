# Original OPF versus SIGReg: controlled public-core comparison

## Outcome

On this deterministic synthetic dynamics task, adding SIGReg to a matched
JEPA-Anything OPF implementation improved the mean held-out open-loop state
rollout R² at both predeclared horizons. The result is promising **for this
specific simulator and implementation**, not evidence that SIGReg improves the
paper's seven domain benchmarks. The repository contains a reusable core and a
structural synthetic recipe, not the full published training pipelines or
weights.

| Arm | Current-state probe R² | Rollout R², 1 step | Rollout R², 8 steps | Gram RMSE ↓ | Train time, s |
|---|---:|---:|---:|---:|---:|
| Original OPF objective | 0.743 ± 0.021 | 0.741 ± 0.038 | 0.496 ± 0.023 | 0.0105 ± 0.0030 | 2.73 ± 0.09 |
| Original + SIGReg | 0.858 ± 0.048 | 0.880 ± 0.024 | 0.730 ± 0.167 | 0.0981 ± 0.0181 | 10.45 ± 0.45 |
| SIGReg replaces variance floors | 0.858 ± 0.048 | 0.880 ± 0.024 | 0.730 ± 0.167 | 0.0981 ± 0.0181 | 10.37 ± 0.16 |

Entries are mean ± sample standard deviation across three paired seeds. R² is
computed against the simulator's true state on four held-out trajectories. A
single fixed ridge probe was fitted on the 16 training trajectories *after*
unsupervised representation training; simulator state labels did not enter the
JEPA loss. No test-based checkpoint or hyperparameter selection was used.

### Paired primary outcomes

| Seed | Original h1 | +SIGReg h1 | Δ h1 | Original h8 | +SIGReg h8 | Δ h8 |
|---:|---:|---:|---:|---:|---:|---:|
| 0 | 0.7203 | 0.9067 | +0.1864 | 0.4718 | 0.8036 | +0.3318 |
| 1 | 0.7849 | 0.8625 | +0.0777 | 0.5183 | 0.8483 | +0.3300 |
| 2 | 0.7168 | 0.8699 | +0.1531 | 0.4988 | 0.5394 | +0.0406 |
| Mean paired Δ | | | **+0.1391** | | | **+0.2341** |

The replacement arm has **exactly the same recorded losses and evaluation
metrics** as original + SIGReg for each matched seed (wall-clock times differ).
Both original variance-floor terms are zero at every logged checkpoint in all
arms, including the first update. Consequently, this experiment establishes
that removing those *inactive* floors has no observed cost here; it does **not**
show that SIGReg can replace an active floor in a harder setting. To test that,
the next experiment should use a setting where the baseline floors actually
activate, with the activation frequency logged every step.

### Secondary outcomes and trade-offs

| Arm | Rollout R², 2 steps | Rollout R², 4 steps | Effective rank | Min coordinate std |
|---|---:|---:|---:|---:|
| Original | 0.719 ± 0.044 | 0.645 ± 0.041 | 3.105 ± 0.117 | 0.618 ± 0.119 |
| +SIGReg / replacement | 0.887 ± 0.023 | 0.864 ± 0.050 | 2.815 ± 0.193 | 1.034 ± 0.138 |

SIGReg raised feature scale but slightly lowered covariance effective rank;
there is no simple across-the-board geometry win. Projector Gram RMSE rose
about 9× even while prediction improved, suggesting tension between the
Gaussian regularizer and the OPF orthogonality term at these fixed weights.
The compose/decompose round-trip remained numerically accurate (all RMSEs
below 6 × 10⁻⁷). On this CPU, SIGReg took roughly 3.8× the training time of
the original objective. The three seeds and four test trajectories are too
few for a broad statistical claim; seed 2 in particular had only a +0.041
8-step R² gain.

## Fidelity, execution, and reproduction

The [pre-result protocol](../../recipes/opf-sigreg-comparison/PROTOCOL.md) and
[implementation](../../recipes/opf-sigreg-comparison/train.py) were pushed in
commit `fc691f7` before the full 3 × 3 results were inspected. The source
recipe's deterministic generated-stream SHA-256 is
`01458bda039dc23e4a6af255ed0ae75792db293d69f7e7a7b574fc290bb3f2d0`,
and the comparison reproduces it exactly. The source's train/validation/test
trajectory IDs, OPF factor dimensions, prediction/orthogonality/activity
weights, minimum-std thresholds, 1,000 steps, and 12,000 ± 0.3% parameter
target were followed. All arms have 12,012 trainable parameters and share
initialization, batches, EMA target encoder, optimizer, model, and probe
procedure. SIGReg uses weight 0.005 and 256 slices, chosen in an earlier
synthetic sweep rather than from this test set.

Important deviations from the structural source recipe are shared across
arms: a one-observation encoder and one-step target loss replace its proposed
eight-observation context and multiple simultaneous target offsets. The
factor-head MLP, EMA momentum 0.996, and optimizer are explicit implementation
choices because the fixture does not provide an executable trained model.
Therefore this is a comparison *within a public-core instantiation*, not a
checkpoint-to-checkpoint comparison with a trained official JEPA-Anything
model.

All nine runs were executed locally on CPU (Python 3.11.2, PyTorch 2.11.0),
in about 71 seconds of aggregate training time. No Runpod pod or GPU was
created for this comparison, so it consumed **$0 of Runpod credit**.
The local CPU path was sufficient and preserved the available experiment
budget. The complete machine-readable per-seed metrics and logged training
history are in [results.json](results.json).

```bash
PYTHONPATH=jepa-anything-core/src .venv/bin/python \
  recipes/opf-sigreg-comparison/train.py --device cpu \
  --output-dir results/opf-sigreg-comparison-2026-09-25
```

The core and task-design test suites and repository checks passed in the same
environment before this run. The code can run on CUDA by changing the device,
but GPU values would require a separate run and should not be conflated with
these CPU results.
