#!/bin/bash
# Submete a fase rerun do khipu. Rode no LOGIN do sauron, na raiz do projeto:
#   cd ~/workspace/Gate-VQ-VAE && bash jobs/khipu/submit-rerun.sh
set -euo pipefail
sauron submit -c jobs/khipu/rerun-gate-norm-k1024.yaml jobs/khipu/job.sh
sauron submit -c jobs/khipu/rerun-dense-st-k1024.yaml jobs/khipu/job.sh
sauron submit -c jobs/khipu/rerun-dense-st-k8192.yaml jobs/khipu/job.sh
