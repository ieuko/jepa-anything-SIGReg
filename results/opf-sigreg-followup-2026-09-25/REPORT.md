# OPF + SIGReg follow-up: geometry, active floors, and external streams

## Bottom line

In this public-core synthetic-dynamics implementation, reducing the SIGReg
weight to 0.001 and raising the OPF Gram weight to 5.0 retained the predictive
gain while eliminating the earlier projector-geometry degradation. The cell
was selected **only on the original validation trajectories** under the
[pre-result protocol](../../recipes/opf-sigreg-comparison/FOLLOWUP_PROTOCOL.md),
committed and pushed as `35e3937` before running the full follow-up. Five
paired-seed runs on two newly generated streams favored this selected cell at
both 1- and 8-step horizons. This supports the trade-off for this simulator,
not a paper-wide or real-data conclusion.

| External stream; metric | Original | Selected (SIGReg 0.001, Gram 5) | Prior SIGReg (0.005, Gram 0.05) |
|---|---:|---:|---:|
| New IID stream: 1-step R² | 0.732 ± 0.130 | 0.892 ± 0.110 | 0.908 ± 0.033 |
| New IID stream: 8-step R² | 0.505 ± 0.055 | **0.849 ± 0.037** | 0.764 ± 0.118 |
| Shifted dynamics: 1-step R² | 0.812 ± 0.069 | **0.911 ± 0.048** | 0.893 ± 0.037 |
| Shifted dynamics: 8-step R² | 0.436 ± 0.041 | **0.772 ± 0.043** | 0.585 ± 0.242 |
| Projector Gram RMSE ↓ | 0.0107 ± 0.0021 | **0.00109 ± 0.00041** | 0.1029 ± 0.0200 |

Values are mean ± sample SD over five model seeds. The Gram metric is a
model-basis property, so it is identical on the two external streams for any
given trained model. Paired selected-minus-original 8-step R² improvements
were +0.343 ± 0.045 (new IID) and +0.336 ± 0.031 (shifted); every one of the
five paired differences was positive on both. On the shifted stream, the
prior SIGReg condition had one negative paired 8-step difference (seed 2),
and its dispersion was much higher. The selected condition is **not** best
on every metric: the prior condition had slightly higher mean 1-step R² on
the new IID stream. Five seeds and one synthetic system remain limited.

### Selection audit

The six-cell validation screen crossed SIGReg weights 0.001 and 0.005 with
Gram weights 0.05, 0.5, and 5.0, using seeds 0 and 1. The original baseline
had validation h1/h8 R² of 0.804/0.491. The selected cell had 0.938/0.882
and Gram RMSE 0.00120, satisfying the predeclared Gram ceiling of 0.03 and
1-step non-inferiority guard. The similar 0.001/0.5 cell scored 0.938/0.881
with Gram RMSE 0.00862; the tiny validation h8 difference selected Gram 5,
not a meaningful superiority claim between those two cells. The earlier
0.005/0.05 condition had validation h1/h8 0.906/0.837 but Gram RMSE 0.1047.
No cell was selected using external-stream scores or the previously viewed
test trajectories.

### Active-floor stress: replacement remains unproven

The source recipe's 0.01 minimum-std floors did not activate in the previous
comparison. To make their causal role observable, this diagnostic deliberately
raised both floor thresholds to 1.0. The 0.1 floor weights, SIGReg weight
0.005, Gram weight 0.05, three seeds, and 1,000 updates remained fixed.

| Stress arm | Factor floor active, % updates | Encoder floor active, % updates | Validation h1 R² | Validation h8 R² |
|---|---:|---:|---:|---:|
| Original with active floors | 28.7 ± 7.0 | 1.8 ± 0.4 | 0.893 ± 0.071 | **0.776 ± 0.039** |
| Original + SIGReg | 66.2 ± 2.2 | 9.9 ± 4.0 | 0.906 ± 0.054 | 0.717 ± 0.171 |
| SIGReg replaces floors | 88.4 ± 12.0 | 29.5 ± 25.7 | 0.909 ± 0.038 | 0.743 ± 0.164 |

The activation counter tests whether the *raw* floor loss is positive at
each update; it also evaluates the counterfactual floor in the replacement
arm, where the floor weight is zero. Removing active floors did **not** give
a reliable 8-step improvement over the original: replacement-minus-original
paired deltas were +0.004, +0.097, and −0.199. The high spread and the
artificially raised threshold prevent a strong claim either way. In
particular, the positive external-stream result above concerns **adding**
SIGReg with a stronger Gram constraint, not replacing necessary collapse
protection. This is the main unresolved question.

## Experimental scope and integrity

The original training stream is the source recipe's deterministic simulator,
with SHA-256 `01458bda039dc23e4a6af255ed0ae75792db293d69f7e7a7b574fc290bb3f2d0`.
The new IID stream uses simulator seed 18181 and otherwise identical dynamics.
The shifted stream uses seed 19191, changes two transition couplings from
0.12→0.18 and 0.17→0.23, and raises process/observation noise from
0.01→0.015 and 0.02→0.03. Its stream fingerprint is recorded in
[results.json](results.json). In both external streams, only trajectory IDs
20–23 are evaluated. Their observations are standardized with the **original
train-set** normalization. The linear state probe is fitted only on original
train states; external labels are evaluation-only. Original train, validation,
and previously viewed test trajectories are not used as the new test set.

All arms share the 12,012-parameter architecture, batch draws, initial model
seeds, 1,000 AdamW updates, EMA target encoder, and source simulator except
for the explicitly varied objective weights. This is still a one-observation,
one-step-loss instantiation of the public core, **not** the paper's full
eight-context/multi-target domain recipe or a comparison to published
checkpoints. The shifted stream is a modest synthetic distribution shift,
not proof of general out-of-distribution robustness.

The 14 screen runs, nine stress runs, and 15 external runs completed on local
CPU (Python 3.11.2, PyTorch 2.11.0), totaling about 301 seconds of measured
training time. No Runpod pod was created, and this follow-up spent **$0 of
Runpod credit**. Raw per-seed metrics, logged losses, selection values, and
stream fingerprints are in [results.json](results.json).

```bash
PYTHONPATH=jepa-anything-core/src .venv/bin/python \
  recipes/opf-sigreg-comparison/followup.py \
  --output-dir results/opf-sigreg-followup-2026-09-25
```

The next substantive study should use a separate, higher-dimensional or real
sequential dataset with a naturally active floor, predefine the probe and
external test split, and compare the selected SIGReg+Gram setting against
the original objective there. Avoid interpreting an artificially high floor
threshold as evidence for changing the source recipe.
