# Pre-result protocol: geometry, active-floor stress, and external streams

This follow-up must be committed before inspecting its validation or external
results. It follows the initial 3-seed public-core comparison but does not
reuse the previously viewed four test trajectories for selection or reporting.

## Stage 1: geometry versus prediction (validation only)

- Original OPF baseline: Gram weight 0.05, no SIGReg.
- Six SIGReg cells: SIGReg weight in {0.001, 0.005} crossed with Gram weight
  in {0.05, 0.5, 5.0}. Every cell and the baseline gets matched seeds 0, 1,
  1,000 updates, and the same source-recipe data stream.
- Selection reads only validation trajectories 16–19. Eligible cells must
  have mean projector Gram RMSE at most 0.03 and mean validation 1-step R²
  no more than 0.02 below the original baseline. Among eligible cells, choose
  the highest mean validation 8-step R²; break ties by lower Gram RMSE, then
  lower SIGReg weight. If none meet the geometry guard, choose the lowest-Gram
  cell with 8-step R² at least the baseline and 1-step R² no more than 0.02
  below; if still none, choose the lowest-Gram cell overall. Mark such a
  fallback as noncompliant rather than claiming a successful trade-off.
- Test metrics and the previously viewed test trajectories are not computed
  during selection. No early stopping or per-arm architecture changes.

## Stage 2: active-floor stress (validation only)

- Raise both minimum coordinate standard deviations from 0.01 to 1.0, while
  retaining each floor's 0.1 weight. This is an intentionally altered
  *stress condition*, not the source recipe's original objective.
- Compare original, original + SIGReg, and floors replaced by SIGReg at the
  pre-existing SIGReg weight 0.005 and original Gram weight 0.05, on seeds
  0, 1, 2. Count the fraction of all 1,000 updates on which each raw floor
  loss is positive, including when its weight is zero in the replacement arm.
- A replacement claim requires floor activation in the baseline and no
  material loss of validation rollout quality. This experiment is diagnostic,
  not a claim about the unmodified source thresholds.

## Stage 3: locked external evaluation

- Retrain original, the stage-1 selected cell, and the fixed prior +SIGReg
  cell (0.005, 0.05) if different, on original training trajectories with
  seeds 0–4. The model, probe, optimizer, 1,000 steps, and sample stream stay
  fixed. Stage 1 selection is not revisited after external results.
- Evaluate each model on two newly generated, non-overlapping streams:
  IID seed 18181 (same dynamics) and shifted-dynamics seed 19191. Both use
  trajectory IDs 20–23 only and normalization fitted on the original train
  set. The shifted stream uses A[0,1]=0.18 and A[2,3]=0.23 (source 0.12 and
  0.17), process noise 0.015 (source 0.01), and observation noise 0.03
  (source 0.02); other simulator settings are unchanged.
- Probe labels are original *training* states only. Primary metrics are
  external IID and shifted 1-step and 8-step rollout R², paired against the
  original arm, alongside Gram RMSE. Report every seed and dispersion, not
  only means. Five seeds and four test trajectories per stream remain small;
  no broad statistical or paper-domain claim is warranted.

All stages run locally on CPU if feasible. No Runpod resource is needed for a
12,012-parameter model; preserve the user's Runpod credit for larger studies.
