# Naturally active floor test: OPF + SIGReg on CIFAR-100

## Result

The source objective's **unchanged 0.01 minimum-standard-deviation floors
did activate naturally** under the small image encoder: in the three original
runs, the encoder floor was positive on 0.71% ± 0.04% of optimizer updates and
the factor floor on 5.07% ± 0.48%. Both were sparse. Under the fixed
[pre-result protocol](../../recipes/opf-cifar100-floor/PROTOCOL.md), adding
SIGReg 0.05 or using it in place of these floors **reduced** frozen-probe
accuracy by roughly 4.5–4.8 percentage points relative to the original OPF
objective. This is evidence against this specific replacement configuration,
not against all SIGReg weights or architectures.

| Arm | Test accuracy, % ↑ | Validation accuracy, % ↑ | Effective rank / 64 ↑ | Mean coordinate std | OPF Gram RMSE ↓ |
|---|---:|---:|---:|---:|---:|
| Untrained random encoder | 11.61 ± 0.26 | 10.93 ± 0.18 | 2.97 ± 0.23 | 0.00166 | — |
| Original OPF | **15.54 ± 0.19** | **14.74 ± 0.35** | 7.01 ± 0.32 | 0.182 | 0.00053 |
| Original + SIGReg | 10.73 ± 0.03 | 9.78 ± 0.09 | 11.66 ± 0.57 | 1.174 | 0.02525 |
| SIGReg replaces floors | 10.88 ± 0.67 | 9.89 ± 0.62 | 11.97 ± 0.26 | 1.168 | 0.02380 |
| Plus, Gram 5.0 | 11.04 ± 0.33 | 10.27 ± 0.28 | 11.85 ± 0.86 | 1.181 | **0.00036** |
| Replace, Gram 5.0 | 11.03 ± 0.24 | 10.19 ± 0.42 | 11.76 ± 0.95 | 1.172 | **0.00033** |

Values are means ± sample SD for seeds 0–2, except mean coordinate std,
which shows the mean only; [results.json](results.json) holds all seed-level
measurements. Accuracy is a 100-way frozen ridge linear probe. The paired
test-accuracy changes versus original were −4.81 ± 0.17 points for `plus`,
−4.66 ± 0.77 for `replace`, −4.50 ± 0.52 for `plus-strong-gram`, and
−4.51 ± 0.28 for `replace-strong-gram`. Replacement minus matched addition
was only +0.15 ± 0.67 points at Gram 0.05 and −0.01 ± 0.42 at Gram 5.0:
with three seeds, this does not isolate a reliable cost or benefit from
removing the sparse floors. All four SIGReg arms scored below even the
untrained random-encoder probe, despite their larger feature scales and
effective ranks. A larger effective rank is therefore not sufficient for a
useful semantic representation here.

The Gram-5 conditions brought the learned factor basis close to orthogonal,
but did not recover the original probe accuracy. The fixed SIGReg weight
`0.05` came from the earlier *non-OPF* CIFAR-10 study. Its poor transfer to
this factorized objective suggests a loss-scale or objective-interaction
problem; that is an interpretation, **not** a measured gradient-causality
result. We did not retune the weight after seeing these scores.

## Scope and reproducibility

This is an **exploratory** image-view experiment: the CIFAR-100 official test
split was inspected during an earlier project phase, before this study.
The protocol and executable implementation were nevertheless committed as
`57d877f` before these OPF scores were observed. Do not treat the test
differences as a new unbiased benchmark confirmation. The model is a small
public-core OPF instantiation, not the JEPA-Anything paper's complete
architecture or a pretrained checkpoint; absolute accuracies are low.

The publisher's CIFAR-100 binary archive was verified by its MD5
`03b5dce01913d631647c71ecec9e9cb8`. SSL used the first 45,000 images of
the official training split without labels; its remaining 5,000 images were
validation. Probe fitting used only training labels, and the separate
10,000-image official test split was used for final scoring. Every active arm
used 16 epochs, 2,816 updates, batch 256, the same encoder/EMA/predictors,
and seed-matched initialization and augmentation order. The GPU was an
NVIDIA RTX PRO 6000 Blackwell Server Edition MIG 1g.24gb with PyTorch
2.8.0+cu128. The [run log](run.log) and [raw results](results.json) contain
every epoch and per-seed metric.

```bash
PYTHONPATH=jepa-anything-core/src python3 \
  recipes/opf-cifar100-floor/study.py \
  --data-dir /workspace/datasets --device cuda \
  --output-dir /workspace/opf-cifar100-floor-2026-09-26
```

The same temporary Runpod Pod also ran the
[real-sequence UCI HAR study](../opf-uci-har-2026-09-26/REPORT.md).
The account balance was `$45.4070307218` before provisioning and
`$45.1105698857` shortly before deletion, a provisional decrease of
`$0.2964608361` across both studies and setup. The final settled charge may
differ. Pod `mypa84j6y82co5` was deleted after both result files and logs
were retrieved; subsequent Pod and network-volume lists were empty.
Runpod subsequently reported `currentSpendPerHr: 0`.
