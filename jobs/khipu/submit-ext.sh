#!/bin/bash
# Submete a fase ext do khipu. Rode no LOGIN do sauron, na raiz do projeto:
#   cd ~/workspace/Gate-VQ-VAE && bash jobs/khipu/submit-ext.sh
set -euo pipefail
sauron submit -c jobs/khipu/ext-ste-ema-k8192.yaml jobs/khipu/job.sh
sauron submit -c jobs/khipu/ext-gate-norm-k8192.yaml jobs/khipu/job.sh
