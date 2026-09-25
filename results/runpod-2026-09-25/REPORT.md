# JEPA Anything + SIGReg synthetic ablation

The experiment jointly trains one encoder and a next-state predictor. Prediction-only training admits the trivial constant solution; the variance baseline uses the existing coordinate activity floors; SIGReg replaces those floors with sliced Epps--Pulley regularization toward an isotropic Gaussian.

| variant | pred. MSE ↓ | min std ↑ | cov. RMSE ↓ | eff. rank ↑ | probe R² ↑ | SIGReg ↓ |
|---|---:|---:|---:|---:|---:|---:|
| prediction-only | 0.00000 | 0.0006 | 0.3536 | 7.448 | 0.0126 | 1651.240 |
| variance | 0.01154 | 1.0184 | 1.0302 | 1.251 | 0.4159 | 655.349 |
| sigreg | 0.01821 | 0.9880 | 0.0318 | 7.973 | 0.9513 | 5.081 |

## Interpretation

- Prediction-only reaches a misleadingly tiny predictive MSE by shrinking every
  coordinate (100% below the 0.1 collapse threshold); its probe R² is 0.013.
- Coordinate variance floors restore marginal scale but leave the representation
  nearly rank one (effective rank 1.25/8 and mean covariance condition number
  113,465).
- SIGReg prevents coordinate collapse while retaining almost all eight dimensions
  (effective rank 7.97/8), brings covariance close to identity (RMSE 0.0318), and
  preserves simulator state for the frozen linear probe (R² 0.951).

The prediction MSE values are not directly comparable as task quality without
the geometry metrics: a constant encoder makes prediction artificially easy.

## Reproduction

```bash
python recipes/sigreg-synthetic-dynamics/train.py \
  --device cuda --steps 1500 --batch-size 1024 --eval-samples 4096 \
  --num-slices 256 --seeds 0,1,2 \
  --output-dir results/runpod-2026-09-25
```

Device: `NVIDIA RTX PRO 4000 Blackwell`; PyTorch `2.8.0+cu128`; CUDA `12.8`;
3 variants × 3 seeds = 9 runs. Runpod balance changed from $46.0478759606 to
$46.0190725958, a measured cost of $0.0288033648. The experiment Pod was deleted
after artifacts were retrieved; post-run spend was $0/hour.

These are controlled synthetic results, not a domain benchmark. The linear probe uses ground-truth simulator state only for evaluation.
