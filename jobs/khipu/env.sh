#!/bin/bash
# env.sh — ambiente dos jobs do Sauron para o Gate-VQ-VAE. Uso, dentro do job:
#   source jobs/khipu/env.sh && setup_environment
#
# Segue o env.sh dos exemplos do MOE-SAE, que mediu o que funciona na imagem
# pytorch/pytorch:2.13.0-cuda13.0-cudnn9-devel: `python -m venv` morre (sem ensurepip), e
# `pip --target` reinstala o torch por cima da imagem. O que funciona é `pip install --user`
# com PIP_BREAK_SYSTEM_PACKAGES=1 e PYTHONUSERBASE no cache persistente.
#
# torch, torchvision e numpy vêm SEMPRE da imagem. Só o pyyaml é instalado. NÃO rode
# `pip install .`: o pyproject pede torch>=2.15 (versão do Mac) e o pip tentaria trocar o
# torch 2.13 da imagem. O código roda direto de src/ (PYTHONPATH).
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
# o Sauron monta ~/workspace como /workspace; o cache fica FORA do projeto (persiste entre jobs)
WORKSPACE_ROOT="${WORKSPACE_ROOT:-$(cd "${PROJECT_ROOT}/.." && pwd)}"
CACHE_ROOT="${CACHE_ROOT:-${WORKSPACE_ROOT}/cache}"
DEPS_TAG="${DEPS_TAG:-$(echo "${SAURON_IMAGE:-pytorch-2.13.0-cu130}" | tr -c 'A-Za-z0-9_.-' '_')}"
PYTHONUSERBASE="${PYTHONUSERBASE:-${CACHE_ROOT}/pyuser-gatevq-${DEPS_TAG}}"
export PYTHONUSERBASE PIP_BREAK_SYSTEM_PACKAGES=1
export PIP_CACHE_DIR="${PIP_CACHE_DIR:-/cache/pip}"   # cache de wheels do nó (visto nos exemplos)
export PYTHONPATH="${PROJECT_ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}"
DEPS_MARK="${PYTHONUSERBASE}/.deps-ok"

log() { echo "[khipu $(date '+%Y-%m-%d %H:%M:%S')] $*"; }
die() { echo "[khipu] ERRO: $*" >&2; exit 1; }

ensure_deps() {
    mkdir -p "${PYTHONUSERBASE}"
    # trava: jobs simultâneos não podem instalar no mesmo prefixo ao mesmo tempo
    exec 9>"${PYTHONUSERBASE}.lock"
    flock 9 || log "AVISO: flock indisponível; seguindo sem trava"
    if [[ -f "${DEPS_MARK}" ]]; then
        log "dependências já em ${PYTHONUSERBASE}"
    else
        log "instalando pyyaml em ${PYTHONUSERBASE} (torch vem da imagem)"
        python -m pip install --quiet --user pyyaml
        touch "${DEPS_MARK}"
    fi
    exec 9>&-
}

verify_gpu() {
    python - <<'PYEOF'
import os, sys
import torch
if not torch.cuda.is_available():
    print("[ERRO] torch.cuda.is_available() == False: o job não pegou GPU.", torch.__version__)
    sys.exit(1)
p = torch.cuda.get_device_properties(0)
print(f"[OK] torch {torch.__version__} | {p.name} | {p.total_memory / 2**30:.0f} GiB")
# o torch TEM de ser o da imagem; se algo o reinstalou em PYTHONUSERBASE, o torchvision quebra
base = os.path.realpath(os.environ.get("PYTHONUSERBASE", "/nao-existe"))
if os.path.realpath(torch.__file__).startswith(base):
    print("[ERRO] o torch veio do PYTHONUSERBASE, não da imagem:", torch.__file__)
    sys.exit(1)
import torchvision, yaml  # noqa: F401
print(f"[OK] torchvision {torchvision.__version__} e pyyaml importam")
PYEOF
}

print_job_context() {
    log "projeto   : ${PROJECT_ROOT}"
    log "cache     : ${CACHE_ROOT}"
    # QUAL commit está rodando (o workspace do cluster é um clone que precisa de git pull);
    # a imagem não tem o binário git, então git_commit() lê o .git em Python
    log "commit    : $(python -c 'from vqrouter.train import git_commit; print(git_commit())')"
}

# Pico de RAM do job (cgroup), para dimensionar `memory:` pela medida e não por palpite.
relatorio_memoria() {
    local pico="" limite="" f
    for f in /sys/fs/cgroup/memory.peak /sys/fs/cgroup/memory/memory.max_usage_in_bytes; do
        [[ -r "${f}" ]] && { pico="$(cat "${f}" 2>/dev/null)"; break; }
    done
    for f in /sys/fs/cgroup/memory.max /sys/fs/cgroup/memory/memory.limit_in_bytes; do
        [[ -r "${f}" ]] && { limite="$(cat "${f}" 2>/dev/null)"; break; }
    done
    [[ "${pico}" =~ ^[0-9]+$ ]] || { log "RAM: o cgroup não expôs o pico neste nó"; return 0; }
    awk -v p="${pico}" -v l="${limite}" 'BEGIN {
        g = 1024 ^ 3
        if (l ~ /^[0-9]+$/ && l > 0 && l < 2 ^ 62) {
            printf "[khipu] RAM do job: pico %.1f GiB de %.0f GiB pedidos (%.0f%% do limite)\n", p/g, l/g, 100*p/l
            u = 100 * p / l
            if (u < 50) printf "  -> sobra demais: cabe em %.0fg. Baixe `memory:` no yaml.\n", (p/g)*1.3 + 1
            else if (u > 90) print "  -> perto do teto: suba `memory:` antes de crescer."
        } else printf "[khipu] RAM do job: pico %.1f GiB (sem limite legível)\n", p/g
    }'
}

setup_environment() {
    ensure_deps
    verify_gpu
    print_job_context
    trap relatorio_memoria EXIT   # EXIT: o número sai também quando um passo falha
}
