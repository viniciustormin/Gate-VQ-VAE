"""Gera os yamls dos jobs do sauron e o script de submissão de uma fase da grade khipu.

    uv run python jobs/khipu/make_grid.py pilot --ms-per-step 40
    uv run python jobs/khipu/make_grid.py main  --ms-per-step 40 --tau 1
    uv run python jobs/khipu/make_grid.py k8192 --ms-per-step 60 --tau 1

Fases (um job por estimador, as execuções do job rodam em sequência):
    pilot  gate_norm, gate_cb, dense_st | K=1024 | seed 0         | tau em {0.1, 1, 10} x mediana
    main   os 5 estimadores             | K=1024 | seeds 0, 1, 2  | tau escolhido (--tau)
    k8192  ste, gate_cb, dense_st       | K=8192 | seeds 0, 1, 2  | tau escolhido (--tau)

O script NÃO submete nada: escreve jobs/khipu/<fase>-<estimador>.yaml e jobs/khipu/submit-<fase>.sh,
que o Vinícius roda no login do sauron. Só usa chaves de yaml do guia e dos exemplos.
`--ms-per-step` é obrigatório: o max_time sai dele (o piloto mede, o smoke mede antes).
"""

import argparse
import math
from pathlib import Path

HERE = Path(__file__).resolve().parent
IMAGE = "pytorch/pytorch:2.13.0-cuda13.0-cudnn9-devel"  # a recomendada pelo guia
MARGIN = 1.3  # folga sobre o tempo estimado (avaliações, checkpoints, fila de disco)

PHASES = {
    "pilot": dict(
        estimators=["gate_norm", "gate_cb", "dense_st"],
        ks=[1024],
        seeds=[0],
        taus=["0.1", "1", "10"],
    ),
    "main": dict(
        estimators=["ste", "ste_ema", "gate_norm", "gate_cb", "dense_st"],
        ks=[1024],
        seeds=[0, 1, 2],
        taus=None,
    ),
    "k8192": dict(estimators=["ste", "gate_cb", "dense_st"], ks=[8192], seeds=[0, 1, 2], taus=None),
}


def max_time(n_runs: int, steps: int, ms_per_step: float) -> str:
    minutes = n_runs * steps * ms_per_step / 1000 / 60 * MARGIN
    if minutes <= 30:
        return "30m"
    return f"{math.ceil(minutes / 60)}h"  # os exemplos só usam minutos e horas inteiras


def render(
    phase, est, ks, taus, seeds, tau_scale, steps, ms, memory, time_evidence, memory_evidence
):
    ks_s, taus_s, seeds_s = (" ".join(map(str, v)) for v in (ks, taus, seeds))
    n = len(ks) * len(taus) * len(seeds)
    mt = max_time(n, steps, ms)
    name = f"khipu-{phase}-{est.replace('_', '-')}"
    text = f"""# Fase {phase} do khipu: {est}, K={ks_s}, tau={taus_s} (escala {tau_scale}),
# seeds {seeds_s}: {n} execuções de {steps} passos, em sequência, na mesma GPU.
#
#   sauron submit -c jobs/khipu/{name.removeprefix("khipu-")}.yaml jobs/khipu/job.sh
#
# Reenviar este job retoma do último checkpoint e pula o que já concluiu (run_group.sh --resume).
#
# max_time {mt}: de {ms:g} ms/passo x {n} execuções x {steps} passos x {MARGIN:g} de folga.
#   {time_evidence}
# memory {memory}: {memory_evidence}
name: {name}
image: {IMAGE}
gpus: 1
vram: 89     # menor classe; o VQ-VAE é pequeno e o job gasta 89 GB da cota de qualquer jeito
memory: {memory}
cpus: 2
max_time: {mt}

workspace: Gate-VQ-VAE

env:
  EXP: "khipu"
  ESTIMATORS: "{est}"
  SEEDS: "{seeds_s}"
  KS: "{ks_s}"
  TAUS: "{taus_s}"
  TAU_SCALE: "{tau_scale}"
"""
    return name, text, mt


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("phase", choices=sorted(PHASES))
    ap.add_argument("--ms-per-step", type=float, required=True)
    ap.add_argument("--tau", help="tau escolhido (obrigatório em main e k8192)")
    ap.add_argument("--tau-scale", default="median", choices=["median", "null"])
    ap.add_argument("--steps", type=int, default=20000)
    ap.add_argument("--memory", default="8g")
    ap.add_argument("--time-evidence", default="1ª ESTIMATIVA: ms/passo sem medida no sauron.")
    ap.add_argument("--memory-evidence", default="1ª ESTIMATIVA, sem pico medido.")
    ap.add_argument("--out", default=str(HERE))
    args = ap.parse_args()

    cfg = PHASES[args.phase]
    taus = cfg["taus"]
    if taus is None:
        if args.tau is None:
            raise SystemExit(f"a fase {args.phase} exige --tau (escolhido depois do piloto)")
        taus = [args.tau]
    out = Path(args.out)
    submit = [
        "#!/bin/bash",
        f"# Submete a fase {args.phase} do khipu. Rode no LOGIN do sauron, na raiz do projeto:",
        "#   cd ~/workspace/Gate-VQ-VAE && bash jobs/khipu/submit-" + args.phase + ".sh",
        "set -euo pipefail",
    ]
    for est in cfg["estimators"]:
        name, text, mt = render(
            args.phase, est, cfg["ks"], taus, cfg["seeds"], args.tau_scale, args.steps,
            args.ms_per_step, args.memory, args.time_evidence, args.memory_evidence,
        )  # fmt: skip
        fname = name.removeprefix("khipu-") + ".yaml"
        (out / fname).write_text(text)
        submit.append(f"sauron submit -c jobs/khipu/{fname} jobs/khipu/job.sh")
        print(f"{fname}: max_time {mt}")
    (out / f"submit-{args.phase}.sh").write_text("\n".join(submit) + "\n")


if __name__ == "__main__":
    main()
