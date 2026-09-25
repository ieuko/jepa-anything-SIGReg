# CIFAR-100 validation-only SIGReg weight selection

The [pre-result protocol](../../recipes/opf-cifar100-floor/LOW_WEIGHT_TRANSFER_PROTOCOL.md)
selected **0.0001** from the fixed four-weight screen, before any SVHN
validation or test accuracy was viewed. The source-original mean validation
accuracy was 14.89% (seeds 0, 1); the selected `plus` mean was **13.27%**,
1.62 percentage points lower. Thus, lowering SIGReg's weight reduced the
earlier damage but **did not** match the original OPF objective on this
exploratory CIFAR-100 validation split.

| SIGReg weight | Plus validation accuracy, % (seed 0 / 1) | Mean, % |
|---:|---:|---:|
| **0.0001, selected** | 13.04 / 13.50 | **13.27** |
| 0.0005 | 12.18 / 11.84 | 12.01 |
| 0.001 | 12.22 / 12.36 | 12.29 |
| 0.005 | 10.88 / 11.10 | 10.99 |

No CIFAR-100 test probe was computed in this screen. The deterministic
selection rule chose the highest mean, with a 0.2-point near-tie preference
for the smaller weight; there was no near-tie affecting this choice.
[selection.json](selection.json) records the decision and the four cell
directories contain per-seed results and training logs. This selection is
now frozen for the fresh SVHN transfer. The CIFAR-100 test set had been
viewed in an earlier project phase, so this screen is not an independent
benchmark confirmation.
