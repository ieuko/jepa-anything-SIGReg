# JEPA Anything + SIGReg synthetic ablation

The experiment jointly trains one encoder and a next-state predictor. Prediction-only training admits the trivial constant solution; the variance baseline uses the existing coordinate activity floors; SIGReg replaces those floors with sliced Epps--Pulley regularization toward an isotropic Gaussian.

| variant | pred. MSE ↓ | min std ↑ | cov. RMSE ↓ | eff. rank ↑ | probe R² ↑ | SIGReg ↓ |
|---|---:|---:|---:|---:|---:|---:|
| sigreg | 0.01436 | 0.9742 | 0.0277 | 7.976 | 0.9607 | 4.507 |

Device: `NVIDIA RTX PRO 4000 Blackwell`; PyTorch `2.8.0+cu128`; 3 total runs.

These are controlled synthetic results, not a domain benchmark. The linear probe uses ground-truth simulator state only for evaluation.
