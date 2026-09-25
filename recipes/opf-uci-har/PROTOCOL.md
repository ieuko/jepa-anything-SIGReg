# Pre-result protocol: JEPA-Anything OPF + SIGReg on real inertial sequences

Commit this protocol and implementation before reading any UCI HAR validation
or test scores. This is a fixed transfer experiment, not a weight search. The
SIGReg 0.001 / Gram 5.0 cell was selected on a *different synthetic* study.

## Data and split

- Use the [UCI Human Activity Recognition Using Smartphones dataset](https://archive.ics.uci.edu/dataset/240/human+activity+recognition+using+smartphones)
  (CC BY 4.0). Read its nine inertial-signal files, not the engineered 561
  features. Each example is a 128-sample, 9-channel recorded sensor window.
- The official test split is sealed. From the official training split, reserve
  the three numerically largest subject IDs as validation subjects; all other
  training subjects supply unlabeled JEPA examples and downstream probe fits.
  Do not place windows from one subject on both sides of a split.
- Compute one mean and standard deviation per sensor channel on the JEPA-train
  subjects only; apply that normalizer unchanged to validation and test.
- Partition each 128-sample window into sixteen non-overlapping 8-sample
  chunks. JEPA predicts chunk `t+1` from chunk `t` (fifteen pairs/window).
  Training pairs are sampled uniformly from the JEPA-train subjects. There
  is no activity-label use in the representation objective.

## Matched model and arms

- A shared-architecture 72→128→64 MLP encodes each 8×9 chunk; a stop-gradient
  EMA target encoder uses momentum 0.996. A learned OPF basis splits the 64D
  state into eight 8D factors. Eight separate 64→128→8 prediction pathways
  share the context state. Operational synthesis uses the core pseudoinverse.
- Train for 2,000 updates, batch 256, AdamW 0.001, with seeds 0, 1, 2.
  Keep initialization and sampled pair indices matched by seed across arms.
- Use the source recipe's prediction weight 1, projector-Gram weight 0.05,
  factor-activity and encoder-variance weights 0.1, and **both min-std
  thresholds 0.01**, unchanged. SIGReg uses 128 slices and weight 0.001.
- Five active arms:
  1. `original`: original objective, Gram 0.05;
  2. `plus`: original + SIGReg, Gram 0.05;
  3. `replace`: SIGReg replaces both floors, Gram 0.05;
  4. `plus-strong-gram`: original + SIGReg, Gram 5.0;
  5. `replace-strong-gram`: SIGReg replaces floors, Gram 5.0.
- Add an untrained `random` encoder reference for activity-probe accuracy.
  It is not a matched predictor baseline. No arm-specific learning rate,
  stopping point, encoder, or architecture change is allowed.

## Readout and decision

- Count at **every optimizer update** whether each raw activity-floor loss is
  positive, including its counterfactual value in replacement arms. Report
  both update fractions and mean raw losses. A replacement claim requires a
  naturally active floor in the original arm and no meaningful loss against
  its matched `plus` arm. If the original floors are inactive, explicitly
  call replacement untested on this dataset.
- Primary downstream metric: six-way activity accuracy from a frozen ridge
  linear probe on mean-pooled chunk embeddings. Fit it using only JEPA-train
  subjects' activity labels, evaluate on validation and official test
  subjects. Ridge coefficient is 0.001 × training-example count; no tuning.
- Secondary prediction metrics: fit a frozen ridge linear sensor decoder from
  each training chunk embedding to its 72 normalized sensor values. Roll the
  learned OPF predictor forward 1 and 4 chunks from test/validation contexts,
  decode each latent prediction, and report sensor R². Include feature scale,
  effective rank, projector Gram RMSE, and compute time to detect collapse or
  geometry trade-offs. Do not interpret a high activity score alone as
  predictive world modeling.
- Summarize paired seed differences, means and sample standard deviations.
  Validation is diagnostic only; do not select a different weight or rerun
  test conditions based on its scores. Three seeds and one sensor dataset do
  not establish a broad domain claim.

The official archive and code are cached on a temporary Runpod PyTorch GPU
pod only if a local smoke test passes. Budget guard: target a low-cost single
GPU, terminate it within two hours, retrieve results, then delete the pod and
temporary network volume. The data archive itself is not committed to Git.
