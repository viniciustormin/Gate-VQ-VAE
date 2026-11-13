#!/bin/bash
# Acompanha jobs do Sauron: imprime UMA linha a cada mudança de estado e termina quando todos os
# jobs chegam a um estado final. Feito para o Monitor do Claude Code (cada linha vira um evento).
# Só LÊ do cluster (`sauron status`, `sauron logs`); não submete, cancela nem apaga nada.
#
#   bash scripts/watch_sauron.sh 3338 [3339 ...]        INTERVAL=30 (segundos entre consultas)
#
# Estados finais: succeeded, failed_final, cancelled, timeout. Qualquer outro estado novo
# (inclusive um desconhecido) é impresso como mudança e o acompanhamento continua.
set -u
INTERVAL="${INTERVAL:-30}"
[[ $# -ge 1 ]] || { echo "uso: $0 <ref> [<ref> ...]"; exit 2; }

parse() {  # lê o JSON do `sauron status --json` e imprime: estado|nome|mem_peak_mb|motivo
  python3 -c '
import json, sys
d = json.load(sys.stdin)
print("|".join(str(x) for x in (d.get("state"), d["spec"]["name"], d.get("mem_peak_mb"), d.get("wait_reason") or d.get("reason") or "")))'
}

declare_last() { eval "last_$1=\"$2\""; }
get_last() { eval "echo \"\${last_$1:-}\""; }
declare_done() { eval "done_$1=1"; }
is_done() { eval "[[ -n \"\${done_$1:-}\" ]]"; }
falhas=0
while true; do
  restantes=0
  for ref in "$@"; do
    is_done "$ref" && continue
    if ! out=$(ssh -o ConnectTimeout=20 -o BatchMode=yes sauron "sauron status $ref --json" 2>/dev/null) \
       || ! linha=$(parse <<<"$out" 2>/dev/null); then
      falhas=$((falhas + 1))
      (( falhas == 5 )) && echo "AVISO: 5 consultas seguidas ao sauron falharam (ssh ou JSON); sigo tentando"
      restantes=$((restantes + 1)); continue
    fi
    falhas=0
    IFS='|' read -r estado nome mem motivo <<<"${linha}"
    if [[ "${estado}" != "$(get_last "$ref")" ]]; then
      echo "job ${ref} (${nome}): ${estado}${motivo:+ [${motivo}]}"
      declare_last "$ref" "${estado}"
    fi
    case "${estado}" in
      succeeded|failed_final|cancelled|timeout)
        echo "TERMINOU job ${ref} (${nome}): ${estado}; pico de RAM do host: ${mem} MB"
        if [[ "${estado}" != "succeeded" ]]; then
          echo "--- últimas linhas do log do job ${ref}:"
          ssh -o ConnectTimeout=20 -o BatchMode=yes sauron "sauron logs $ref -n 25" 2>&1 | tail -25
        fi
        declare_done "$ref" ;;
      *) restantes=$((restantes + 1)) ;;
    esac
  done
  (( restantes == 0 )) && exit 0
  sleep "${INTERVAL}"
done
