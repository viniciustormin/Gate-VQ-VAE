"""Treino de um VQ-VAE com um estimador de gradiente.

    uv run python -m vqrouter.train --config configs/base/mnist.yaml --override estimator=gate_cb

Cada execução grava em runs/<exp>/<estimator>_K<K>_tau<tau>_s<seed>/: config.yaml resolvido,
metrics.csv, commit.txt, ckpt.pt (modelo, otimizador, passo, RNG, tau congelado) e done.json.
Execução concluída é pulada; `--resume` retoma de ckpt.pt e, com `train.max_steps` maior, estende.
"""

from __future__ import annotations

import argparse
import copy
import csv
import json
import os
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import yaml

from vqrouter.data import batch_indices, load_dataset
from vqrouter.metrics import evaluate
from vqrouter.models import build_model

REPO_ROOT = Path(__file__).resolve().parents[2]
CSV_COLUMNS = [
    "step",
    "train_loss",
    "train_recon",
    "test_mse",
    "test_psnr",
    "codes_used_pct",
    "perplexity",
    "quant_err",
    "tau_eff",
    "wall_s",
]
# chaves que podem mudar entre a execução original e a retomada (máquina, extensão do treino)
RESUME_MAY_DIFFER = {("device",), ("require_cuda",), ("train", "max_steps")}


# --------------------------------------------------------------------------- configuração
def _deep_merge(base: dict, new: dict) -> dict:
    out = copy.deepcopy(base)
    for k, v in new.items():
        out[k] = (
            _deep_merge(out[k], v) if isinstance(v, dict) and isinstance(out.get(k), dict) else v
        )
    return out


def parse_value(text: str):
    """JSON primeiro (3e-4 vira float; o YAML 1.1 do PyYAML o leria como string), depois YAML."""
    try:
        return json.loads(text)
    except ValueError:
        return yaml.safe_load(text)


def load_config(path: str | Path, overrides: list[str] | None = None) -> dict:
    path = Path(path)
    cfg = yaml.safe_load(path.read_text()) or {}
    if "base" in cfg:
        base = load_config((path.parent / cfg.pop("base")).resolve())
        cfg = _deep_merge(base, cfg)
    for item in overrides or []:
        key, sep, val = item.partition("=")
        if not sep:
            raise ValueError(f"override inválido (use chave=valor): {item!r}")
        node = cfg
        *parents, leaf = key.split(".")
        for p in parents:
            node = node.setdefault(p, {})
        node[leaf] = parse_value(val)
    return cfg


def run_name(cfg: dict) -> str:
    vq = cfg["vq"]
    med = "med" if vq.get("tau_scale") == "median" else ""
    return f"{cfg['estimator']}_K{vq['num_codes']}_tau{float(vq['tau']):g}{med}_s{cfg['seed']}"


def run_dir(cfg: dict) -> Path:
    return Path(cfg.get("runs_dir", "runs")) / cfg["exp"] / run_name(cfg)


def _flatten(d: dict, prefix=()) -> dict:
    out = {}
    for k, v in d.items():
        out.update(_flatten(v, prefix + (k,)) if isinstance(v, dict) else {prefix + (k,): v})
    return out


def _check_resume_config(old: dict, new: dict) -> None:
    a, b = _flatten(old), _flatten(new)
    diff = [k for k in a.keys() | b.keys() if a.get(k) != b.get(k) and k not in RESUME_MAY_DIFFER]
    if diff:
        names = ", ".join(".".join(k) for k in sorted(diff))
        raise SystemExit(f"config difere da execução original em: {names}. Não retomo.")


# --------------------------------------------------------------------------- ambiente
def git_commit() -> str:
    """Commit em uso, lido de .git sem o binário git (a imagem do Sauron não o tem)."""
    g = REPO_ROOT / ".git"
    try:
        head = (g / "HEAD").read_text().strip()
        if not head.startswith("ref:"):
            return f"{head[:12]} (HEAD solto)"
        ref = head[5:]
        name = ref.removeprefix("refs/heads/")
        if (g / ref).is_file():
            return f"{(g / ref).read_text().strip()[:12]} ({name})"
        for line in (g / "packed-refs").read_text().splitlines():
            if line.endswith(" " + ref) and not line.startswith(("#", "^")):
                return f"{line.split()[0][:12]} ({name}, packed)"
    except OSError:
        pass
    return "unknown"


def pick_device(cfg: dict) -> torch.device:
    want = cfg.get("device", "auto")
    if want == "auto":
        want = (
            "cuda"
            if torch.cuda.is_available()
            else "mps"
            if torch.backends.mps.is_available()
            else "cpu"
        )
    if (cfg.get("require_cuda") or os.environ.get("REQUIRE_CUDA") == "1") and want != "cuda":
        raise SystemExit(f"require_cuda ligado mas o device seria {want!r}: o job não pegou GPU.")
    return torch.device(want)


def get_rng_state(device: torch.device) -> dict:
    st = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
    }
    if device.type == "cuda":
        st["cuda"] = torch.cuda.get_rng_state_all()
    elif device.type == "mps":
        st["mps"] = torch.mps.get_rng_state()
    return st


def set_rng_state(st: dict) -> None:
    random.setstate(st["python"])
    np.random.set_state(st["numpy"])
    torch.set_rng_state(st["torch"])
    if "cuda" in st and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(st["cuda"])
    if "mps" in st and torch.backends.mps.is_available():
        torch.mps.set_rng_state(st["mps"])


def save_atomic(obj, path: Path) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(obj, tmp)
    os.replace(tmp, path)


# --------------------------------------------------------------------------- CSV
def _read_rows(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with path.open() as f:
        return list(csv.DictReader(f))


def _write_rows(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=CSV_COLUMNS)
        w.writeheader()
        w.writerows(rows)


# --------------------------------------------------------------------------- treino
def train(cfg: dict, resume: bool = False) -> Path:
    tc = cfg["train"]
    max_steps, eval_every, bs = int(tc["max_steps"]), int(tc["eval_every"]), int(tc["batch_size"])
    rd = run_dir(cfg)
    done_path, ckpt_path, csv_path = rd / "done.json", rd / "ckpt.pt", rd / "metrics.csv"

    if done_path.exists() and json.loads(done_path.read_text())["step"] >= max_steps:
        print(f"[skip] {rd} já concluída ({max_steps} passos)")
        return rd
    if ckpt_path.exists() and not resume:
        raise SystemExit(f"{rd} tem checkpoint de uma execução incompleta ou menor. Use --resume.")
    if resume and ckpt_path.exists() and (rd / "config.yaml").exists():
        _check_resume_config(yaml.safe_load((rd / "config.yaml").read_text()), cfg)

    device = pick_device(cfg)
    seed = int(cfg["seed"])
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    data = load_dataset(cfg["dataset"], cfg.get("data", {})).to(device)
    vq_kwargs = {**cfg["vq"], "estimator": cfg["estimator"], "seed": seed}
    model = build_model(
        cfg["model"]["name"],
        data.in_shape,
        vq_kwargs,
        **{k: v for k, v in cfg["model"].items() if k != "name"},
    ).to(device)
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.Adam(params, lr=float(tc["lr"]))

    start, wall0 = 0, 0.0
    rd.mkdir(parents=True, exist_ok=True)
    if resume and ckpt_path.exists():
        ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        model.load_state_dict(ck["model"])
        opt.load_state_dict(ck["optimizer"])
        set_rng_state(ck["rng"])
        start, wall0 = int(ck["step"]), float(ck["wall_s"])
        rows = [r for r in _read_rows(csv_path) if int(r["step"]) <= start]
        _write_rows(csv_path, rows)
        with (rd / "commit.txt").open("a") as f:
            f.write(f"resume@{start}: {git_commit()}\n")
        print(f"[resume] {rd} no passo {start}")
    else:
        _write_rows(csv_path, [])
        (rd / "commit.txt").write_text(f"{git_commit()}\n")
    (rd / "config.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False, allow_unicode=True))

    model.train()
    n = len(data.train)
    zero = torch.zeros((), device=device)
    acc_loss, acc_rec, acc_n = zero.clone(), zero.clone(), 0
    t0, final = time.time(), {}
    for step in range(start, max_steps):
        x = data.float(data.train[batch_indices(step, bs, n, seed).to(device)])
        recon, out = model(x)
        rec = F.mse_loss(recon, x)
        loss = rec + out.loss
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        acc_loss += loss.detach()
        acc_rec += rec.detach()
        acc_n += 1

        s = step + 1
        if s % eval_every == 0 or s == max_steps:
            tl, tr = float(acc_loss) / acc_n, float(acc_rec) / acc_n
            if not (np.isfinite(tl) and np.isfinite(tr)):
                raise RuntimeError(f"perda não finita no passo {s}: loss={tl} recon={tr}")
            final = evaluate(model, data, cfg["vq"]["num_codes"], device)
            wall = wall0 + time.time() - t0
            row = {
                "step": s,
                "train_loss": tl,
                "train_recon": tr,
                **final,
                "tau_eff": float(model.vq.tau_eff),
                "wall_s": wall,
            }
            rows = _read_rows(csv_path)
            rows.append({k: repr(v) if isinstance(v, float) else v for k, v in row.items()})
            _write_rows(csv_path, rows)
            save_atomic(
                {
                    "step": s,
                    "model": model.state_dict(),
                    "optimizer": opt.state_dict(),
                    "rng": get_rng_state(device),
                    "tau_eff": float(model.vq.tau_eff),
                    "wall_s": wall,
                    "config": cfg,
                },
                ckpt_path,
            )
            acc_loss, acc_rec, acc_n = zero.clone(), zero.clone(), 0
            print(
                f"[{run_name(cfg)}] passo {s}/{max_steps} mse={final['test_mse']:.5f} "
                f"psnr={final['test_psnr']:.2f} usados={final['codes_used_pct']:.1f}% "
                f"perp={final['perplexity']:.1f}"
            )

    done_path.write_text(
        json.dumps(
            {
                "step": max_steps,
                "wall_s": wall0 + time.time() - t0,
                "commit": git_commit(),
                "final": final,
            },
            indent=1,
        )
    )
    return rd


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--config", required=True)
    ap.add_argument("--override", nargs="+", action="extend", default=[], metavar="CHAVE=VALOR")
    ap.add_argument("--resume", action="store_true")
    args = ap.parse_args(argv)
    train(load_config(args.config, args.override), resume=args.resume)


if __name__ == "__main__":
    main()
