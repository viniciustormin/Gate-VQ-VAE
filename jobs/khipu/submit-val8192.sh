#!/bin/bash
# Submete a fase val8192 do khipu. Rode no LOGIN do sauron, na raiz do projeto:
#   cd ~/workspace/Gate-VQ-VAE && bash jobs/khipu/submit-val8192.sh
set -euo pipefail
sauron submit -c jobs/khipu/val8192-gate-cb-s0.yaml jobs/khipu/job.sh
sauron submit -c jobs/khipu/val8192-gate-cb-s1.yaml jobs/khipu/job.sh
sauron submit -c jobs/khipu/val8192-gate-cb-s2.yaml jobs/khipu/job.sh
sauron submit -c jobs/khipu/val8192-dense-st-s0.yaml jobs/khipu/job.sh
sauron submit -c jobs/khipu/val8192-dense-st-s1.yaml jobs/khipu/job.sh
sauron submit -c jobs/khipu/val8192-dense-st-s2.yaml jobs/khipu/job.sh
