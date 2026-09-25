# Pre-result protocol: low-weight SIGReg selection and fresh SVHN transfer

This follows the unfavorable fixed-0.05 CIFAR-100 OPF result. It is a
**prospective, exploratory weight screen**, not an independent CIFAR-100 test:
the CIFAR-100 test set was viewed earlier. Commit this protocol and code before
running the screen. Do not view any SVHN validation or test scores until the
selected weight has been committed to GitHub.

## Stage 1: select one weight on CIFAR-100 validation only

- Reuse the exact CIFAR-100 45,000/5,000 split, 64D image encoder, learned
  eight-factor OPF, EMA 0.996, augmentations, batch 256, AdamW 0.001, 16 full
  epochs, and source floor/Gram terms from the [previous protocol](PROTOCOL.md).
- Seeds 0 and 1. Run original once and `plus` at SIGReg weights **0.0001,
  0.0005, 0.001, 0.005**. The prior 0.05 result is a known failed reference,
  not a selection candidate. No CIFAR-100 test features, labels, or probe
  scores are computed in this stage (`--validation-only`).
- Select the weight with the highest mean frozen 100-class ridge-probe accuracy
  on the 5,000 validation images. If any lower weight lies within 0.002
  accuracy (0.2 percentage points) of the maximum, select the smallest such
  weight. Selection happens even if every candidate trails original; state
  that failure. This one rule is implemented in `select_weight.py`.
- Commit and push `selection.json` before any SVHN accuracy is observed.
  Do not tune on SVHN validation, test, geometry, or floor counters.

## Stage 2: fixed transfer on previously unused SVHN

- Use the [SVHN](https://ufldl.stanford.edu/housenumbers/) cropped 32×32 RGB
  train and test files, verified against the MD5 values in torchvision's
  [SVHN loader](https://github.com/pytorch/vision/blob/main/torchvision/datasets/svhn.py).
  A UC San Diego mirror may supply the exact-checksum bytes if the Stanford
  host is unavailable. Do not use the `extra` split.
- Permute the official 73,257-image train split with seed 20260926; first
  65,000 examples are SSL/probe train, remaining 8,257 are diagnostic
  validation. The separate 26,032-image official test split is the primary
  transfer evaluation. Remap original label 10 to digit 0. No labels enter
  SSL training; the ridge probe uses only SSL-train labels.
- Same encoder, learned OPF, thresholds, weights other than selected SIGReg
  weight, EMA, optimizer, schedule (16 epochs), and seeds 0, 1, 2 as Stage 1.
  Apply the same crop/brightness/contrast augmentation to all SVHN arms, but
  **no horizontal flip** because mirrored digits change semantics.
- Arms: untrained random reference, original OPF, original + selected SIGReg,
  and selected SIGReg replacing both 0.01 floors. Gram weight remains 0.05.
  No strong-Gram rescue or weight sweep is allowed on SVHN.
- Primary outcome: official-test 10-class frozen ridge-probe accuracy and
  seed-paired differences against original and between plus/replace. Also
  report validation accuracy, floor-active fractions, feature scale/effective
  rank, and OPF Gram RMSE. An active original floor is necessary to claim the
  replacement contrast tested effective protection. If it remains inactive,
  state that replacement is untested despite differing objective switches.
- Run all arms for all three seeds, without early stopping or revising the
  selected weight. Report unfavorable outcomes and sample SD. Three seeds,
  one digit dataset, a small encoder, and prior CIFAR-100 inspection limit
  broad generalization; no published JEPA-Anything checkpoint is compared.

## Budget and run integrity

Use one low-cost Runpod GPU Pod, keep the two-stage selection order auditable,
retrieve raw JSON/logs, then delete the Pod and any temporary volume. The
account's existing credit is the hard spending ceiling; target <$2 for this
follow-up and avoid continuously billing resources after completion. Never
commit the image archives. If source checksum or expected dimensions fail,
stop before running the SVHN models.
