# SIGReg synthetic dynamics ablation

This recipe is an executable collapse-prevention comparison for the SIGReg
integration. A shared encoder maps nonlinear observations of an 8-dimensional
linear dynamical system to the JEPA state. A predictor learns the next state and
OPF exposes four two-coordinate factors.

The three matched variants are:

- `prediction-only`: admits a trivial constant representation;
- `variance`: uses JEPA Anything's existing target-factor and online-encoder
  coordinate variance floors;
- `sigreg`: replaces both variance floors with one SIGReg term over context and
  target embeddings.

Run a smoke check:

```bash
python recipes/sigreg-synthetic-dynamics/train.py \
  --steps 50 --batch-size 128 --eval-samples 256 --seeds 0 \
  --output-dir runs/sigreg-smoke
```

Run the full ablation used for the checked-in Runpod result:

```bash
python recipes/sigreg-synthetic-dynamics/train.py \
  --device cuda --steps 1500 --batch-size 1024 --eval-samples 4096 \
  --seeds 0,1,2 --output-dir runs/sigreg-runpod
```

The output directory contains `summary.json`, `metrics.csv`, and a generated
`REPORT.md`. The linear probe sees simulator state only during evaluation; it is
not part of training. Consequently this recipe is a controlled mechanism test,
not a domain benchmark.
