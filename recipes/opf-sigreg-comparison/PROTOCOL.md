# Pre-result OPF–SIGReg comparison protocol

Commit this file and the implementation before running or reading the full
three-seed test results. No arm-specific learning rate, stopping time, model
size, or SIGReg weight will be adjusted from test performance.

- **Question:** Relative to the public JEPA-Anything core/structural recipe,
  does adding SIGReg help, and can it replace the two coordinate variance
  floors without harming prediction?
- **Data:** exact deterministic stream from
  `recipes/synthetic-linear-dynamics/recipe.json` (24 trajectories × 96 steps).
  Train trajectory IDs 0–15, reserved validation IDs 16–19, test IDs 20–23.
  Normalize observations using train trajectories only.
- **Model:** online/EMA target encoders, learned 4D → 2×2 OPF basis, two
  factor-specific predictors conditioned on the known scalar control. All
  arms use the same 12,012-trainable-parameter architecture; target EMA
  momentum 0.996 and stop-gradient are fixed. The source's projector Gram
  weight is 0.05, factor activity and online variance weights are 0.1, and
  both minimum standard deviations are 0.01.
- **Arms:** unchanged original objective; original plus SIGReg at 0.005;
  original with both variance floors replaced by SIGReg at 0.005. SIGReg uses
  256 random slices and is applied to online context embeddings. The weight
  was selected in a prior *different synthetic* experiment.
- **Training:** seeds 0, 1, 2; 1,000 fixed steps; batch 256; AdamW at 0.001;
  CPU execution; no early stopping or checkpoint selection. All variants use paired initial
  model seeds and batch-index streams.
- **Primary evaluation:** test open-loop state-rollout R² at horizons 1 and 8,
  decoded by a frozen ridge (`0.001 × n`) probe fitted solely on train-state
  labels after unsupervised training. Report paired seed differences of each
  SIGReg arm against the original, whether favorable or not. The test true
  state is used only for evaluation.
- **Secondary evaluation:** horizons 2 and 4, current-state probe R², effective
  rank, minimum coordinate std, projector-Gram RMSE, synthesis roundtrip
  RMSE, and wall-clock training time. Read scale together with rank; a high
  normalized rank at near-zero scale is not evidence against collapse.
- **Claim boundary:** this is a same-simulator, core-API comparison, not a
  reproduction of the paper's seven-domain benchmarks. A positive result
  would support this local setting only; three seeds do not establish broad
  superiority. If the original arm already succeeds, preserving its rollout
  quality is part of the replacement criterion.
