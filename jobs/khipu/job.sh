# Corpo do job da grade khipu. Quais execuções rodar vem do bloco `env:` do yaml
# (EXP, ESTIMATORS, SEEDS, KS, TAUS, TAU_SCALE). Enviado com:
#   sauron submit -c jobs/khipu/<job>.yaml jobs/khipu/job.sh
# Fino de propósito (o guia limita o script a 4096 caracteres): a lógica está em env.sh e
# run_group.sh. Sem `source .env`: este job não usa segredo nenhum.
set -euo pipefail
source jobs/khipu/env.sh
setup_environment
bash jobs/khipu/run_group.sh
