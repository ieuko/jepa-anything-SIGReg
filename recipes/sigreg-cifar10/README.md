# CIFAR-10/100 SIGReg transfer experiments

This recipe tests the earlier synthetic-dynamics weight choice on real images.
The encoder receives two independently augmented views of the same CIFAR-10
training image. A predictor maps one view's embedding toward the other. The
encoder and target share weights, as in the synthetic mechanism experiments.
No class labels enter the representation-learning loss.

The official CIFAR-10 binary archive is downloaded from the dataset author's
site (or an HTTPS mirror) and checked against its published MD5 before parsing.
The 50,000 official
training images are used for self-supervised training and for fitting a frozen
linear probe. Probe accuracy is measured on the separate 10,000-image test
split. The probe is ridge-regularized least squares on standardized features;
its labels are available only after encoder training.

Four variants share the same architecture and seeds:

- `random`: untrained encoder reference;
- `prediction-only`: predictive loss without collapse prevention;
- `variance`: existing coordinate variance floors, weight 4.0 (best tested
  synthetic baseline);
- `sigreg`: SIGReg, weight 0.005 (best tested synthetic setting).

```bash
PYTHONPATH=jepa-anything-core/src python3 recipes/sigreg-cifar10/train.py \
  --device cuda --data-dir /workspace/datasets \
  --download-url https://data.brainchip.com/dataset-mirror/cifar10/cifar-10-binary.tar.gz \
  --output-dir results/runpod-cifar10-2026-09-25
```

The model is deliberately small and trained for eight epochs; this is a
controlled transfer check, not a claim of competitive CIFAR-10 accuracy. It
does not use target-network EMA, contrastive negatives, or a large pretrained
backbone. Geometry and linear-probe accuracy should be read together because
effective rank is insensitive to the absolute scale of a collapsed embedding.

The checked-in [Runpod report](../../results/runpod-cifar10-2026-09-25/REPORT.md)
also includes 32-epoch runs and a five-weight SIGReg sweep. The figure can be
regenerated with Matplotlib:

```bash
python recipes/sigreg-cifar10/plot.py \
  --result-dir results/runpod-cifar10-2026-09-25
```

The sweep is exploratory because test-set probe scores were inspected while
extending it.

## Fixed-weight CIFAR-100 transfer check

The follow-up [protocol](CIFAR100_PROTOCOL.md) freezes the CIFAR-10-selected
SIGReg weight at `0.05` before inspecting any CIFAR-100 results. It keeps the
same encoder, augmentations, 32 epochs, three seeds, and fixed ridge probe,
and compares against random, prediction-only, and variance-floor controls.
The `--dataset cifar100` option downloads the publisher's binary archive,
verifies its published MD5, and uses the fine labels only for the linear probe.

```bash
PYTHONPATH=jepa-anything-core/src python3 recipes/sigreg-cifar10/train.py \
  --dataset cifar100 --device cuda --epochs 32 \
  --sigreg-weight 0.05 --variants random,prediction-only,variance,sigreg \
  --seeds 0,1,2 --output-dir results/runpod-cifar100-2026-09-25
```
