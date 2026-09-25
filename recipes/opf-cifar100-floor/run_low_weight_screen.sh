#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 2 ]]; then
  echo "usage: $0 DATA_DIR OUTPUT_ROOT" >&2
  exit 2
fi

data_dir=$1
output_root=$2
mkdir -p "$output_root"

for cell in \
  "w0p0001:0.0001:original,plus" \
  "w0p0005:0.0005:plus" \
  "w0p001:0.001:plus" \
  "w0p005:0.005:plus"; do
  IFS=: read -r name weight variants <<< "$cell"
  mkdir -p "$output_root/$name"
  PYTHONPATH=jepa-anything-core/src python3 \
    recipes/opf-cifar100-floor/study.py \
    --dataset cifar100 --data-dir "$data_dir" \
    --device cuda --epochs 16 --seeds 0,1 \
    --variants "$variants" --sigreg-weight "$weight" \
    --validation-only --output-dir "$output_root/$name" \
    > "$output_root/$name/run.log" 2>&1
  echo "completed $name"
done

PYTHONPATH=jepa-anything-core/src python3 \
  recipes/opf-cifar100-floor/select_weight.py \
  --screen-root "$output_root" --output "$output_root/selection.json"
