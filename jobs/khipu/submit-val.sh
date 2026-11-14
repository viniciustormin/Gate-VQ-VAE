#!/bin/bash
# Submete a fase val do khipu. Rode no LOGIN do sauron, na raiz do projeto:
#   cd ~/workspace/Gate-VQ-VAE && bash jobs/khipu/submit-val.sh
set -euo pipefail
sauron submit -c jobs/khipu/val-gate-norm-s0.yaml jobs/khipu/job.sh
sauron submit -c jobs/khipu/val-gate-norm-s1.yaml jobs/khipu/job.sh
sauron submit -c jobs/khipu/val-gate-norm-s2.yaml jobs/khipu/job.sh
sauron submit -c jobs/khipu/val-gate-cb-s0.yaml jobs/khipu/job.sh
sauron submit -c jobs/khipu/val-gate-cb-s1.yaml jobs/khipu/job.sh
sauron submit -c jobs/khipu/val-gate-cb-s2.yaml jobs/khipu/job.sh
sauron submit -c jobs/khipu/val-dense-st-s0.yaml jobs/khipu/job.sh
sauron submit -c jobs/khipu/val-dense-st-s1.yaml jobs/khipu/job.sh
sauron submit -c jobs/khipu/val-dense-st-s2.yaml jobs/khipu/job.sh
