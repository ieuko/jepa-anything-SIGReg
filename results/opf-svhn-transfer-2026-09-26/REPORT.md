# Frozen low-weight SIGReg transfer to SVHN

## Bottom line

The [pre-result protocol](../../recipes/opf-cifar100-floor/LOW_WEIGHT_TRANSFER_PROTOCOL.md)
selected SIGReg weight **0.0001** using *only* CIFAR-100 validation scores.
That choice and its unfavorable selection result were committed and pushed as
`f509ff2` **before any SVHN accuracy was observed**. On the fresh SVHN test
split, neither adding SIGReg nor replacing the source 0.01 floors improved
the original OPF objective on average. The differences are small and based
on three seeds, so the evidence supports **no demonstrated benefit**, not a
universal harm claim.

| Arm | Test digit accuracy, % ↑ | Validation accuracy, % ↑ | Effective rank / 64 ↑ | OPF Gram RMSE ↓ |
|---|---:|---:|---:|---:|
| Untrained random encoder | 21.72 ± 0.30 | 20.90 ± 0.10 | 1.60 ± 0.08 | — |
| Original OPF | **22.93 ± 0.29** | **21.99 ± 0.27** | 4.22 ± 0.22 | **0.00062 ± 0.00004** |
| Original + SIGReg 0.0001 | 22.25 ± 0.60 | 21.08 ± 0.58 | 7.50 ± 0.37 | 0.01417 ± 0.00037 |
| SIGReg replaces both floors | 22.33 ± 0.73 | 21.35 ± 0.61 | **7.65 ± 0.23** | 0.01437 ± 0.00067 |

Values are mean ± sample SD over matched seeds 0–2. The seed-paired
test-accuracy difference was **−0.69 ± 0.84 percentage points** for addition
minus original and **−0.60 ± 0.48 points** for replacement minus original.
Replacement minus matched addition was +0.08 ± 1.11 points. The per-seed
addition differences had mixed signs; replacement was below original in all
three seeds, but only by 0.05–0.88 points. No significance or non-inferiority
claim is justified from three runs. The absolute gain of original over the
untrained encoder is itself only 1.21 points, so this small model is a weak
semantic-representation test.

The original's online-encoder floor was active in **0.72% ± 0.07%** of
updates; its factor floor in **2.94% ± 0.26%**. The corresponding raw
counterfactual losses in the replacement arm were positive in 1.57% ± 0.02%
and 5.65% ± 0.42% of updates. Thus, unlike the earlier UCI HAR result, the
replacement arm actually removed occasionally active protections. It did not
produce a reliable accuracy benefit. SIGReg increased mean test feature
standard deviation from 0.188 to 1.116 (`plus`) or 1.094 (`replace`) and
raised effective rank, but projector Gram RMSE worsened roughly 23-fold.
These geometry/scale changes did not translate into better digit accuracy.

## Selection and scope

The [CIFAR-100 validation screen](../opf-lowweight-screen-2026-09-26/SELECTION.md)
compared four predeclared weights. The selected 0.0001 condition scored
13.27% validation accuracy versus **14.89% for original**; it was selected
because it was the best *SIGReg candidate*, not because it beat original.
The larger previously tested 0.05 weight was not eligible for selection.
No SVHN result changed the selected weight, augmentation, schedule, or arms.

SVHN's official 73,257-image train split was deterministically permuted and
divided into 65,000 SSL/probe-train and 8,257 validation examples. Its separate
26,032-image official test split was used for the final probe. SSL used no
digit labels; the frozen ridge probe used only training labels. We did not
use SVHN's additional `extra` split. Left-right flips were disabled for all
SVHN arms, because digit identity is not flip-invariant. Each active arm used
the same small 64D encoder, 8-factor learned OPF, EMA target, 16 epochs,
4,064 updates, and source 0.01 floors/Gram 0.05 except for the specified
SIGReg/floor changes. This is a public-core test, **not** the complete
JEPA-Anything paper model or a comparison with a pretrained checkpoint.

The UC San Diego SVHN mirror repeatedly interrupted transfers, so the files
were obtained from a public Hugging Face mirror and checked against the exact
train/test MD5 values used by [torchvision's SVHN loader](https://github.com/pytorch/vision/blob/main/torchvision/datasets/svhn.py):
`e26dedcc434d2e4c54c9b2d4a06d8373` and
`eb5a983be6a315427106f1b164d9cef3`. No dataset bytes are committed.
The GPU was an NVIDIA RTX PRO 6000 Blackwell Server Edition MIG 1g.24gb with
PyTorch 2.8.0+cu128. [Raw results](results.json) include all per-seed scores,
floor counters, data hashes, and environment metadata; [run.log](run.log)
contains the epoch trace.

```bash
PYTHONPATH=jepa-anything-core/src python3 \
  recipes/opf-cifar100-floor/study.py \
  --dataset svhn --data-dir /workspace/datasets/svhn \
  --download-url https://huggingface.co/datasets/darcook/DCPV-Artifacts/resolve/main/data/svhn \
  --device cuda --epochs 16 --seeds 0,1,2 \
  --variants random,original,plus,replace --sigreg-weight 0.0001 \
  --output-dir /workspace/opf-svhn-transfer-2026-09-26
```

The Runpod account balance fell from `$45.0835457551` before provisioning
to `$44.8847310431` after deletion, an observed decrease of
`$0.1988147120` for this follow-up. Pod `xcf3g08pc33y61` was deleted
after both raw artifacts were retrieved; subsequent Pod/network-volume lists
were empty and `currentSpendPerHr` was zero.

The next scientifically useful step is to measure the *separate gradient
scales* of prediction, SIGReg, floors, and Gram near initialization and late
training, then predefine a dimensionless normalization or gradient-ratio rule.
Further test-set weight tuning on CIFAR-100 or SVHN would not resolve this
objective-interaction question cleanly.
