# Fixed-weight CIFAR-100 transfer protocol

This protocol was committed **before** running or inspecting CIFAR-100 probe
results. It addresses the selection bias of the CIFAR-10 weight sweep by
fixing all settings before evaluating a different dataset.

- Dataset: publisher's CIFAR-100 binary archive, MD5
  `03b5dce01913d631647c71ecec9e9cb8`; 50,000 train and 10,000 test images.
  Use the 100 *fine* labels for the probe; no labels enter pretraining.
- Encoder, predictor, 64-dimensional state, eight orthogonal factors, 128
  SIGReg slices, crop/flip/brightness/contrast augmentation, AdamW learning
  rate `0.001`, batch size 256: unchanged from CIFAR-10.
- Train each non-random variant for exactly 32 epochs at seeds 0, 1, and 2.
  Variants: untrained random encoder, prediction-only, variance floors at
  weight `4.0`, and SIGReg at the CIFAR-10-selected weight `0.05`.
- Primary measurement: 100-class test accuracy of the frozen, standardized
  ridge linear probe (ridge `0.001 × n_train`), with mean and sample standard
  deviation across three seeds. Primary contrasts: SIGReg minus random and
  SIGReg minus variance floors, paired by seed.
- Secondary measurements: effective rank, minimum coordinate standard
  deviation, and covariance-identity RMSE on test embeddings. Inspect these
  only alongside accuracy, not as a substitute for it.
- No CIFAR-100-specific tuning or additional weights after seeing test
  scores. If the transfer fails, report the failure. This is a small-model
  cross-dataset check, not a state-of-the-art benchmark.

The CIFAR-100 format and archive checksum come from the
[dataset publisher](https://cave.cs.toronto.edu/kriz/cifar.html).
