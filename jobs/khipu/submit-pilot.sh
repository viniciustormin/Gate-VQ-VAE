#!/bin/bash
# Submete a fase pilot do khipu. Rode no LOGIN do sauron, na raiz do projeto:
#   cd ~/workspace/Gate-VQ-VAE && bash jobs/khipu/submit-pilot.sh
set -euo pipefail
sauron submit -c jobs/khipu/pilot-gate-norm.yaml jobs/khipu/job.sh
sauron submit -c jobs/khipu/pilot-gate-cb.yaml jobs/khipu/job.sh
sauron submit -c jobs/khipu/pilot-dense-st.yaml jobs/khipu/job.sh
