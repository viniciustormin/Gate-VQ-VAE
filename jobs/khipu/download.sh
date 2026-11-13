# Baixa o CIFAR-10 para data/ (a única etapa que usa a internet para dados). Enviado com:
#   sauron submit -c jobs/khipu/download.yaml jobs/khipu/download.sh
# Sem GPU no corpo (o yaml pede a menor classe só porque todo job do Sauron aloca uma).
set -euo pipefail
source jobs/khipu/env.sh
ensure_deps
print_job_context
trap relatorio_memoria EXIT
python scripts/download_data.py --datasets cifar10
du -sh data/cifar-10-batches-py
