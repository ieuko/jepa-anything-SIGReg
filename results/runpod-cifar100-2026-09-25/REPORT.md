# Fixed-weight CIFAR-100 transfer check

The [protocol](../../recipes/sigreg-cifar10/CIFAR100_PROTOCOL.md) and code were
committed as `5fad72b` **before** any CIFAR-100 probe scores were inspected.
The SIGReg weight `0.05` was selected in the earlier
[CIFAR-10 sweep](../runpod-cifar10-2026-09-25/REPORT.md), then kept fixed here.
The same small JEPA-style encoder, augmentations, 32-epoch schedule, three
seeds, and ridge probe were used. Self-supervised training never used class
labels; the frozen probe used the 100 *fine* labels from the official training
split and was scored on the official 10,000-image test split.

| Objective | Weight | Test accuracy (%) | Effective rank / 64 | Min. coordinate std |
|---|---:|---:|---:|---:|
| Random encoder | — | 11.65 ± 0.28 | 2.97 | 0.00056 |
| Prediction only | — | 6.53 ± 0.25 | 52.98* | 0.000035 |
| Variance floors | 4.0 | 5.64 ± 0.71 | 1.00 | 1.341 |
| SIGReg | 0.05 | **12.46 ± 0.11** | 22.27 | 0.761 |

Entries are means ± sample standard deviations of seeds 0, 1, and 2. The
pre-specified paired accuracy differences, in percentage points, were:

| Contrast | Seed 0 | Seed 1 | Seed 2 | Mean ± SD |
|---|---:|---:|---:|---:|
| SIGReg − random | +0.98 | +0.95 | +0.52 | **+0.82 ± 0.26** |
| SIGReg − variance floors | +6.83 | +6.06 | +7.59 | **+6.83 ± 0.77** |

The positive SIGReg-versus-random difference in all three seeds is a modest
cross-dataset improvement, not a large semantic gain. Accuracy remains low in
absolute terms for a 100-class task. The variance-floor baseline again has
high per-coordinate spread but nearly rank-one covariance, showing that
coordinate variance alone did not prevent correlated dimensional collapse.
`*` Prediction-only's high *normalized* effective rank is misleading because
all coordinates have near-zero scale; its mean coordinate standard deviation
is also tiny. Rank must be read together with scale and probe accuracy.

This check avoids tuning the SIGReg weight on CIFAR-100, but it is still one
small architecture and one dataset family, with only three seeds. The CIFAR-10
test set was used to select `0.05`, and the CIFAR-100 test set is now used to
assess it; further tuning against these scores would compromise this check.
No competitive CIFAR-100 claim is made. A different image domain, stronger
backbone, and genuinely held-out evaluation would be needed for broad claims.

## Reproduce and verify

```bash
PYTHONPATH=jepa-anything-core/src python3 recipes/sigreg-cifar10/train.py \
  --dataset cifar100 --device cuda --epochs 32 \
  --sigreg-weight 0.05 --variants random,prediction-only,variance,sigreg \
  --seeds 0,1,2 --output-dir results/runpod-cifar100-2026-09-25
```

The run used the HTTPS mirror URL recorded in `results.json`. Its full archive
MD5 matched the [publisher's CIFAR-100 binary checksum](https://cave.cs.toronto.edu/kriz/cifar.html)
`03b5dce01913d631647c71ecec9e9cb8`; the first 1 MiB was also compared
against the official download before switching to the faster mirror. All 12
results, epoch histories, exact settings, and software versions are in
[`results.json`](results.json); raw stdout is in [`run.log`](run.log).
The dataset is described in Krizhevsky's
[technical report](https://www.cs.toronto.edu/~kriz/learning-features-2009-TR.pdf).

The Runpod GPU was an NVIDIA RTX PRO 4000 Blackwell with PyTorch 2.8.0+cu128.
The account balance changed from `$45.5188998662` to `$45.4076659551` while
running this check, a decrease of `$0.1112339111`. The Pod was deleted;
Runpod then showed no Pods and `$0/hour` current spend.
