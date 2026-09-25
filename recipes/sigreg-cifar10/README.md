# CIFAR-10 SIGReg transfer experiment

This recipe tests the earlier synthetic-dynamics weight choice on real images.
The encoder receives two independently augmented views of the same CIFAR-10
training image. A predictor maps one view's embedding toward the other. The
encoder and target share weights, as in the synthetic mechanism experiments.
No class labels enter the representation-learning loss.

The official CIFAR-10 binary archive is downloaded from the dataset author's
site and checked against its published MD5 before parsing. The 50,000 official
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
  --output-dir results/runpod-cifar10-2026-09-25
```

The model is deliberately small and trained for eight epochs; this is a
controlled transfer check, not a claim of competitive CIFAR-10 accuracy. It
does not use target-network EMA, contrastive negatives, or a large pretrained
backbone. Geometry and linear-probe accuracy should be read together because
effective rank is insensitive to the absolute scale of a collapsed embedding.
