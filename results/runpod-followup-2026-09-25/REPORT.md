# SIGReg follow-up sweep

Each cell reports the mean across seeds; ± is sample standard deviation.
Bimodal state coordinates are independent standardized sign mixtures with 0.15 Gaussian jitter.

| latent distribution | objective | weight | probe R² | predicted next-state R² | effective rank / 8 | covariance RMSE | train seconds |
|---|---|---:|---:|---:|---:|---:|---:|
| gaussian | prediction-only | — | 0.0126 ± 0.0027 | 0.0008 ± 0.0022 | 7.4481 ± 0.2099 | 0.3536 ± 0.0000 | 4.2551 ± 0.2145 |
| gaussian | variance | 0.25 | 0.3505 ± 0.0538 | 0.2423 ± 0.0775 | 1.0718 ± 0.0210 | 1.0549 ± 0.0542 | 6.2719 ± 0.5388 |
| gaussian | variance | 1.0 | 0.4159 ± 0.0906 | 0.3599 ± 0.0813 | 1.2506 ± 0.3097 | 1.0302 ± 0.0918 | 6.0518 ± 0.4315 |
| gaussian | variance | 4.0 | 0.5449 ± 0.0203 | 0.4821 ± 0.0480 | 1.4351 ± 0.4293 | 1.2367 ± 0.2220 | 5.9836 ± 0.6160 |
| gaussian | sigreg | 0.002 | 0.9100 ± 0.0895 | 0.9001 ± 0.0934 | 7.9740 ± 0.0117 | 0.0280 ± 0.0061 | 7.8062 ± 0.2846 |
| gaussian | sigreg | 0.005 | 0.9607 ± 0.0006 | 0.9539 ± 0.0009 | 7.9757 ± 0.0078 | 0.0277 ± 0.0044 | 8.3469 ± 0.5814 |
| gaussian | sigreg | 0.01 | 0.9568 ± 0.0008 | 0.9504 ± 0.0008 | 7.9740 ± 0.0090 | 0.0299 ± 0.0047 | 8.5925 ± 0.2006 |
| gaussian | sigreg | 0.02 | 0.9513 ± 0.0011 | 0.9454 ± 0.0009 | 7.9728 ± 0.0096 | 0.0318 ± 0.0045 | 8.0606 ± 0.6901 |
| gaussian | sigreg | 0.05 | 0.9455 ± 0.0015 | 0.9401 ± 0.0013 | 7.9717 ± 0.0105 | 0.0335 ± 0.0045 | 8.6947 ± 0.1793 |
| bimodal | prediction-only | — | 0.0817 ± 0.0073 | 0.0095 ± 0.0033 | 7.1627 ± 0.2487 | 0.3536 ± 0.0000 | 4.1187 ± 0.5423 |
| bimodal | variance | 0.25 | 0.2499 ± 0.0349 | 0.1775 ± 0.1138 | 1.0030 ± 0.0034 | 1.1640 ± 0.1133 | 6.7872 ± 0.1834 |
| bimodal | variance | 1.0 | 0.3907 ± 0.0959 | 0.3558 ± 0.1208 | 1.1022 ± 0.0809 | 1.2815 ± 0.0796 | 6.5548 ± 0.2013 |
| bimodal | variance | 4.0 | 0.5594 ± 0.0142 | 0.4677 ± 0.1249 | 1.3406 ± 0.3198 | 1.4462 ± 0.2310 | 6.1513 ± 0.5566 |
| bimodal | sigreg | 0.002 | 0.9547 ± 0.0423 | 0.9453 ± 0.0423 | 7.9744 ± 0.0054 | 0.0316 ± 0.0010 | 8.7258 ± 0.1735 |
| bimodal | sigreg | 0.005 | 0.9810 ± 0.0012 | 0.9711 ± 0.0013 | 7.9730 ± 0.0062 | 0.0321 ± 0.0024 | 8.4108 ± 0.4397 |
| bimodal | sigreg | 0.01 | 0.9715 ± 0.0145 | 0.9616 ± 0.0135 | 7.9671 ± 0.0072 | 0.0339 ± 0.0030 | 8.9441 ± 0.2290 |
| bimodal | sigreg | 0.02 | 0.9188 ± 0.1010 | 0.9064 ± 0.1036 | 7.9661 ± 0.0044 | 0.0339 ± 0.0013 | 8.2540 ± 0.3084 |
| bimodal | sigreg | 0.05 | 0.9196 ± 0.0934 | 0.9062 ± 0.0972 | 7.9668 ± 0.0020 | 0.0335 ± 0.0011 | 8.8752 ± 0.3201 |

## Findings

- `0.005` is the best tested SIGReg weight on both distributions. Its frozen
  encoder probe R² is `0.9607 ± 0.0006` on Gaussian states and `0.9810 ± 0.0012`
  on bimodal states. Predicted next-state R² is `0.9539 ± 0.0009` and
  `0.9711 ± 0.0013`, respectively.
- The strongest tested variance baseline (`weight=4`) reaches probe R² `0.5449`
  on Gaussian and `0.5594` on bimodal states. Its effective rank remains below
  `1.5/8` on average, despite noncollapsed coordinate standard deviations.
- SIGReg retains nearly eight effective dimensions even when the simulator's
  latent state is bimodal rather than Gaussian. At higher weights, however,
  geometry is not a sufficient quality criterion: bimodal seed 1 falls to probe
  R² `0.8022` at weight `0.02` and `0.8118` at weight `0.05`, while effective
  rank remains approximately `7.96/8`. The regularizer can therefore overconstrain
  a useful representation without causing dimensional collapse.
- Prediction-only achieves near-zero embedding MSE by shrinking the representation.
  Its predicted next-state R² is `0.0008` on Gaussian and `0.0095` on bimodal
  states. The next-state probe exposes the failure more directly than raw MSE.

## Experimental design and limits

The encoder and predictor train together for 1,500 steps with batch size 1,024.
Each of 18 conditions uses the same three model/data seeds and the same fixed
observation mixing matrices. For each run, a linear probe is fitted on a separate
training sample of 4,096 current-state embeddings and evaluated on 4,096 fresh
examples. Ground-truth simulator state is used for evaluation only. The
predicted next-state metric applies that same probe to predictor outputs.

The bimodal state is an independent standardized sign mixture with Gaussian
jitter of standard deviation `0.15` per coordinate. The comparison across
Gaussian and bimodal conditions is a robustness check, not an equal-difficulty
score comparison. Weight `0.005` was selected using these evaluation results;
it has not been validated on new simulator worlds or a real dataset. The
variance baseline was swept over three weights, but stronger weights and
other decorrelation baselines remain untested. Training-time figures exclude
Pod provisioning, data transfer, and evaluation.

## Reproduce and cost

```bash
PYTHONPATH=jepa-anything-core/src python3 \
  recipes/sigreg-synthetic-dynamics/sweep.py \
  --device cuda --output-dir results/runpod-followup-2026-09-25
```

The Runpod GPU was `NVIDIA RTX PRO 4000 Blackwell` with PyTorch `2.8.0+cu128`.
The measured balance change was `$0.0875127241` (`$45.9996598208` to
`$45.9121470967`). The experiment Pod was deleted; post-run spend is `$0/hour`.
Per-seed metrics are in `metrics.csv`, full condition histories in each
subdirectory, and the combined machine-readable results in `summary.json`.
