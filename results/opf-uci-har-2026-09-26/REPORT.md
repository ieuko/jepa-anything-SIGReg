# Real sequential OPF + SIGReg: UCI HAR

## Result

Under the [pre-result protocol](../../recipes/opf-uci-har/PROTOCOL.md), the
source recipe's 0.01 standard-deviation floors were **inactive at every one of
the 2,000 updates** in all three original runs. The same was true of their
counterfactual raw losses in every SIGReg arm. Consequently, `plus` and
`replace` gave identical per-seed scores, as did their strong-Gram counterparts.
This experiment does **not** test whether SIGReg can replace an effective
floor. It tests the effect of adding SIGReg to a matched original objective
whose floors happen to be silent on these normalized sensor windows.

| Arm | Activity accuracy ↑ | Sensor R², 1 chunk ↑ | Sensor R², 4 chunks ↑ | Effective rank ↑ | Gram RMSE ↓ |
|---|---:|---:|---:|---:|---:|
| Original | 0.896 ± 0.005 | 0.605 ± 0.006 | 0.362 ± 0.001 | 10.35 ± 0.76 | 0.00628 ± 0.00022 |
| Original + SIGReg 0.001 | 0.888 ± 0.009 | **0.610 ± 0.006** | **0.369 ± 0.003** | **18.84 ± 0.84** | 0.02149 ± 0.00023 |
| SIGReg replaces floors | 0.888 ± 0.009 | **0.610 ± 0.006** | **0.369 ± 0.003** | **18.84 ± 0.84** | 0.02149 ± 0.00023 |
| Plus, Gram 5.0 | 0.888 ± 0.005 | 0.599 ± 0.003 | 0.362 ± 0.005 | 17.85 ± 0.69 | **0.00037 ± 0.00001** |
| Replace, Gram 5.0 | 0.888 ± 0.005 | 0.599 ± 0.003 | 0.362 ± 0.005 | 17.85 ± 0.69 | **0.00037 ± 0.00001** |

Means ± sample SD over paired seeds 0–2; [results.json](results.json) has every seed.
Relative to original, ordinary `plus` changed activity accuracy by −0.76
percentage points (paired SD 0.98 points), 1-chunk sensor R² by +0.0056
(paired SD 0.0114), and 4-chunk R² by +0.0067 (paired SD 0.0015).
The stronger Gram penalty recovered basis orthogonality, but did not improve
the sensor metrics in this real-sequence test. Those small, mixed changes
should not be read as a robust predictive gain. The untrained encoder's
activity probe already scored 0.846 on average, making activity accuracy a
particularly weak stand-alone measure of world-model quality.

## Scope and provenance

This is the public-core 8-factor implementation, **not** the paper's complete
domain architecture or published checkpoint. The experiment used the nine raw
inertial channels of [UCI HAR](https://archive.ics.uci.edu/dataset/240/human+activity+recognition+using+smartphones),
not its engineered 561-feature table. The train/validation split is by subject
within the official training data; the official test subjects remain disjoint.
SSL saw no activity labels. The frozen probe used labels only from training
subjects. The sensor readout is a fixed ridge decoder fitted on training
chunks and evaluated after one- or four-chunk OPF rollout.

The Runpod execution used a publisher-equivalent ZIP mirror after the official
download stalled. Its outer archive SHA-256 is
`50dabbc800629611831a85b8b71c87040525ca4af6a152c6ba9360ecee6b92dc`;
the concatenated SHA-256 of all 22 signal/subject/label files actually used is
`ff08c51162385a9b1fef71d768a0c361592e312ea6f4f482b884ecd31191614c`,
identical to the official archive checked locally. The GPU was an NVIDIA RTX
PRO 6000 Blackwell Server Edition MIG 1g.24gb with PyTorch 2.8.0+cu128.
The exact environment, split IDs, normalization, raw floor counters, and
per-seed metrics are in [results.json](results.json); the unabridged training
trace is [run.log](run.log).

This one dataset, three seeds, and small probe shifts do not warrant a broad
real-world advantage claim. A naturally active source-threshold floor is
required for a genuine replacement test; the separately preregistered
[CIFAR-100 image experiment](../../recipes/opf-cifar100-floor/PROTOCOL.md)
addresses that narrower mechanism in a different modality.
