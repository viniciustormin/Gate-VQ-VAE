#!/bin/bash
# Submete a fase main do khipu. Rode no LOGIN do sauron, na raiz do projeto:
#   cd ~/workspace/Gate-VQ-VAE && bash jobs/khipu/submit-main.sh
set -euo pipefail
sauron submit -c jobs/khipu/main-ste.yaml jobs/khipu/job.sh
sauron submit -c jobs/khipu/main-ste-ema.yaml jobs/khipu/job.sh
sauron submit -c jobs/khipu/main-gate-norm.yaml jobs/khipu/job.sh
sauron submit -c jobs/khipu/main-gate-cb.yaml jobs/khipu/job.sh
sauron submit -c jobs/khipu/main-dense-st.yaml jobs/khipu/job.sh
