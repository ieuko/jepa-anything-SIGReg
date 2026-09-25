# Pre-result protocol: naturally active 0.01 floors on real images

This is a complementary, **exploratory** OPF experiment. It is not an
unbiased new-dataset confirmation: CIFAR-100 test scores were inspected in an
earlier project phase. Its narrow question is whether the source recipe's
unchanged minimum standard deviation of 0.01 becomes active under the
previously implemented small convolutional image encoder, and whether SIGReg
can replace an active floor without degrading the matched objective.

- Data: publisher's CIFAR-100 binary archive, verified against its MD5
  `03b5dce01913d631647c71ecec9e9cb8`. The first 45,000 images of the
  publisher's 50,000-image train split are SSL/probe train; the last 5,000
  are validation. The separate 10,000-image publisher test split is reported
  once after the fixed run. No labels enter SSL training.
- Model: the existing three-convolution, global-average-pool 64D encoder
  from `recipes/sigreg-cifar10/train.py`; an EMA stop-gradient target encoder
  (momentum 0.996); learned complete 64D OPF basis with eight 8D factors;
  eight separate 64→256→8 predictor pathways. Every active arm has identical
  architecture, initialization, batch order, and augmentations per seed.
- Two augmented views of the same image form context and target. Train seeds
  0, 1, 2 for 16 full epochs, batch 256, AdamW 0.001. No early stopping.
- Original weights and thresholds: prediction 1, projector Gram 0.05,
  factor activity 0.1, online variance 0.1, minimum std **0.01 for both**.
  SIGReg uses 128 slices and weight 0.05. The latter was fixed in the earlier
  CIFAR-10 study and is not tuned on this run.
- Arms: untrained random encoder; `original`; `plus` (original + SIGReg);
  `replace` (both floors zeroed, SIGReg added); `plus-strong-gram` and
  `replace-strong-gram` (same additions/replacement with Gram 5.0, fixed from
  the separate synthetic geometry study). The Gram-5 arms are compared to
  each other, not presented as source-config variants.
- Apply SIGReg to online context embeddings only, matching the synthetic
  OPF comparison; do not backpropagate through the EMA target encoder.
- Count every optimizer update with positive *raw* floor loss, even when
  the floor weight is zero in replacement arms. Record raw floor means.
  An active original floor is necessary to interpret replacement.
- Primary metric: frozen ridge linear probe of the 64D encoder features,
  fitted with labels of the 45,000 SSL-train images only; accuracy on the
  5,000 validation and 10,000 test images. Ridge 0.001×train count, fixed.
  Secondary: feature scale, effective rank, covariance shape, projector
  Gram RMSE, prediction loss, and wall-clock time. Report seed-level scores,
  paired differences and sample SD, including unfavorable outcomes.

This is an image-view prediction experiment, not temporal world modeling.
High random-feature accuracy, collapsed dimensions, or weak absolute accuracy
must temper interpretation. It tests the floor mechanism in one real-image
architecture, not whether SIGReg can universally replace OPF protections.
