# JEPA Anything + SIGReg synthetic ablation

The experiment jointly trains one encoder and a next-state predictor. Prediction-only training admits the trivial constant solution; the variance baseline uses the existing coordinate activity floors; SIGReg replaces those floors with sliced Epps--Pulley regularization toward an isotropic Gaussian.

| variant | pred. MSE ↓ | min std ↑ | cov. RMSE ↓ | eff. rank ↑ | probe R² ↑ | SIGReg ↓ |
|---|---:|---:|---:|---:|---:|---:|
| variance | 0.01552 | 1.0659 | 1.2367 | 1.435 | 0.5449 | 619.464 |

Device: `NVIDIA RTX PRO 4000 Blackwell`; PyTorch `2.8.0+cu128`; 3 total runs.

These are controlled synthetic results, not a domain benchmark. The linear probe uses ground-truth simulator state only for evaluation.
