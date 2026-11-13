#!/bin/bash
# Submete a fase k8192 do khipu. Rode no LOGIN do sauron, na raiz do projeto:
#   cd ~/workspace/Gate-VQ-VAE && bash jobs/khipu/submit-k8192.sh
set -euo pipefail
sauron submit -c jobs/khipu/k8192-ste.yaml jobs/khipu/job.sh
sauron submit -c jobs/khipu/k8192-gate-cb.yaml jobs/khipu/job.sh
sauron submit -c jobs/khipu/k8192-dense-st.yaml jobs/khipu/job.sh
