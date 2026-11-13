#!/bin/bash
# Fase 1: 4 estimadores x 3 seeds no toy 2D, no Mac. Execução concluída é pulada.
#   bash scripts/run_toy.sh
set -euo pipefail
cd "$(dirname "$0")/.."
for seed in 0 1 2; do
  for est in ste gate_norm gate_cb dense_st; do
    uv run python -m vqrouter.train --config configs/khipu/toy2d.yaml \
      --override estimator="$est" seed="$seed" --resume
  done
done
