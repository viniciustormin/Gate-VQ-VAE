"""Diagnóstico de um treino: por que uma execução fica presa no platô da imagem média?

Treina com o MESMO código do train.py (mesmo init, mesmo gerador de batches, mesmo otimizador) e,
a cada `--log-every` passos, mede num batch fixo de teste:
    h_norm    entropia média de p = softmax(-d2/tau_eff) sobre log K (1 = uniforme, 0 = one-hot)
    z_rms     escala (RMS) das saídas do encoder        code_rms  escala (RMS) do codebook
    quant_err média de ||z - e_k*||^2                    ratio     log10(quant_err / tau_eff)
    mse       MSE de reconstrução desse batch (pixels em [0, 1])    perp  perplexidade do argmin
Não substitui o train.py: não grava checkpoint nem retoma. Notas em notes/followup_ressalvas.md.

    uv run python scripts/diag_dense.py --estimator dense_st --K 8192 --tau 0.1 --seed 0
"""

import argparse
import csv
import math
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from vqrouter.data import batch_indices, load_dataset
from vqrouter.metrics import perplexity
from vqrouter.models import build_model
from vqrouter.quantizers import sqdist
from vqrouter.train import load_config, pick_device

FIELDS = [
    "step",
    "mse",
    "train_recon",
    "h_norm",
    "z_rms",
    "code_rms",
    "quant_err",
    "tau_eff",
    "ratio",
    "perp",
]


@torch.no_grad()
def measure(model, data, device, tau_eff: float, n: int = 1000) -> dict:
    was_training = model.training
    model.eval()
    x = data.float(data.test[:n].to(device))
    z = model.encoder(x)
    flat = z.permute(0, 2, 3, 1).reshape(-1, z.shape[1]) if model.channels_last else z
    e = model.vq.codebook
    d2 = sqdist(flat, e)
    p = torch.softmax(-d2 / tau_eff, dim=1)
    h = -(p * p.clamp_min(1e-30).log()).sum(1).mean() / math.log(e.shape[0])
    k = d2.argmin(1)
    qe = d2.gather(1, k[:, None]).mean()
    recon, _ = model(x)
    mse = F.mse_loss(recon.clamp(0, 1) if data.scale != 1.0 else recon, x)
    model.train(was_training)
    counts = torch.bincount(k, minlength=e.shape[0])
    return {
        "mse": float(mse),
        "h_norm": float(h),
        "z_rms": float(flat.pow(2).mean().sqrt()),
        "code_rms": float(e.pow(2).mean().sqrt()),
        "quant_err": float(qe),
        "ratio": math.log10(max(float(qe), 1e-30) / tau_eff),
        "perp": perplexity(counts),
    }


def run_diag(cfg: dict, steps: int, log_every: int, extra_steps: tuple[int, ...] = (1, 50, 100)):
    device = pick_device(cfg)
    seed = int(cfg["seed"])
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    data = load_dataset(cfg["dataset"], cfg.get("data", {})).to(device)
    vq_kwargs = {**cfg["vq"], "estimator": cfg["estimator"], "seed": seed}
    mk = {k: v for k, v in cfg["model"].items() if k != "name"}
    model = build_model(cfg["model"]["name"], data.in_shape, vq_kwargs, **mk).to(device)
    opt = torch.optim.Adam(
        [p for p in model.parameters() if p.requires_grad], lr=float(cfg["train"]["lr"])
    )
    bs, n = int(cfg["train"]["batch_size"]), len(data.train)
    model.train()
    rows, acc, cnt = [], torch.zeros((), device=device), 0
    for step in range(steps):
        x = data.float(data.train[batch_indices(step, bs, n, seed).to(device)])
        recon, out = model(x)
        rec = F.mse_loss(recon, x)
        opt.zero_grad(set_to_none=True)
        (rec + out.loss).backward()
        opt.step()
        acc += rec.detach()
        cnt += 1
        s = step + 1
        if s % log_every == 0 or s in extra_steps or s == steps:
            row = {
                "step": s,
                "train_recon": float(acc) / cnt,
                **measure(model, data, device, float(model.vq.tau_eff)),
            }
            row["tau_eff"] = float(model.vq.tau_eff)
            rows.append(row)
            acc, cnt = torch.zeros((), device=device), 0
            print(
                f"  passo {s:5d} mse={row['mse']:.5f} H/logK={row['h_norm']:.3f} "
                f"z_rms={row['z_rms']:.3g} code_rms={row['code_rms']:.3g} "
                f"qe={row['quant_err']:.3g} log10(qe/tau)={row['ratio']:.2f}"
            )
    return rows, model


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--config", default="configs/khipu/cifar10.yaml")
    ap.add_argument("--estimator", default="dense_st")
    ap.add_argument("--K", type=int, default=8192)
    ap.add_argument("--tau", type=float, default=0.1)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--steps", type=int, default=3000)
    ap.add_argument("--log-every", type=int, default=250)
    ap.add_argument("--out", default="runs/diag")
    ap.add_argument("--override", nargs="*", default=[])
    args = ap.parse_args()
    over = [f"estimator={args.estimator}", f"seed={args.seed}", f"vq.num_codes={args.K}"]
    over += [f"vq.tau={args.tau}", "require_cuda=false", *args.override]
    cfg = load_config(args.config, over)
    name = f"{args.estimator}_K{args.K}_tau{args.tau:g}_s{args.seed}"
    print(f"[diag] {name}, {args.steps} passos")
    rows, _ = run_diag(cfg, args.steps, args.log_every)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    with (out / f"{name}.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        w.writeheader()
        w.writerows(
            [{k: repr(v) if isinstance(v, float) else v for k, v in r.items()} for r in rows]
        )


if __name__ == "__main__":
    main()
