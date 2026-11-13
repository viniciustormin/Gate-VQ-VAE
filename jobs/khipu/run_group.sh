#!/bin/bash
# Roda um grupo de execuções, uma depois da outra, na GPU do job. Parâmetros por variável de
# ambiente (o bloco `env:` do yaml). Execução concluída é pulada e a incompleta retoma do
# checkpoint (--resume), então reenviar um job que estourou o max_time continua de onde parou.
#
#   ESTIMATORS SEEDS KS TAUS   listas separadas por espaço (obrigatórias)
#   TAU_SCALE                  median (padrão) ou null
#   EXP                        pasta em runs/ (padrão khipu)
#   MAX_STEPS EVAL_EVERY       só para testes curtos (padrão: os do YAML, 20000 e 2000)
#   REQUIRE_CUDA               true (padrão): recusa rodar em CPU/MPS
#   RUNNER                     comando do Python (padrão `python`; no Mac: `uv run python`)
set -euo pipefail
cd "$(dirname "$0")/../.."

EXP="${EXP:-khipu}"
TAU_SCALE="${TAU_SCALE:-median}"
CONFIG="${CONFIG:-configs/khipu/cifar10.yaml}"
REQUIRE_CUDA="${REQUIRE_CUDA:-true}"
read -r -a runner <<<"${RUNNER:-python}"
: "${ESTIMATORS:?defina ESTIMATORS}" "${SEEDS:?defina SEEDS}" "${KS:?defina KS}" "${TAUS:?defina TAUS}"

extra=()
[[ -n "${MAX_STEPS:-}" ]] && extra+=("train.max_steps=${MAX_STEPS}")
[[ -n "${EVAL_EVERY:-}" ]] && extra+=("train.eval_every=${EVAL_EVERY}")

for est in ${ESTIMATORS}; do
  for k in ${KS}; do
    for tau in ${TAUS}; do
      for seed in ${SEEDS}; do
        echo "=== ${est} K=${k} tau=${tau} (${TAU_SCALE}) seed=${seed} -> runs/${EXP} ==="
        "${runner[@]}" -m vqrouter.train --config "${CONFIG}" --resume --override \
          "estimator=${est}" "seed=${seed}" "exp=${EXP}" "vq.num_codes=${k}" "vq.tau=${tau}" \
          "vq.tau_scale=${TAU_SCALE}" "require_cuda=${REQUIRE_CUDA}" ${extra[@]+"${extra[@]}"}
      done
    done
  done
done

echo "=== tempo por execução ==="
"${runner[@]}" scripts/run_times.py --exp "${EXP}"
