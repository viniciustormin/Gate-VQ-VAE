#!/bin/bash
# Submete a fase val8192gn do khipu. Rode no LOGIN do sauron, na raiz do projeto:
#   cd ~/workspace/Gate-VQ-VAE && bash jobs/khipu/submit-val8192gn.sh
set -euo pipefail
sauron submit -c jobs/khipu/val8192gn-gate-norm-s0.yaml jobs/khipu/job.sh
sauron submit -c jobs/khipu/val8192gn-gate-norm-s1.yaml jobs/khipu/job.sh
sauron submit -c jobs/khipu/val8192gn-gate-norm-s2.yaml jobs/khipu/job.sh
