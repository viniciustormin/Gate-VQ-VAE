"""Figuras do toy 2D (PDF, em inglês): campo de gradiente no plano latente e trajetória dos códigos.

    uv run python scripts/fig_toy.py --exp khipu_toy --seed 0

Campo: para cada ponto z de uma grade no plano latente, o alvo é a amostra de teste cuja latente é
a mais próxima de z, e a seta é -dL/dz com L = ||dec(z_q(z)) - x_alvo||^2, calculado pelo
estimador, usando o decoder e o codebook finais da execução. Cada painel tem a própria escala de
setas (a norma do gradiente difere por ordens de grandeza entre estimadores); a norma mediana de
cada painel vem anotada.
Trajetória: caminho de cada codeword (snapshots de `codebook_traj.npz`) sobre as latentes finais.
"""

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml
from matplotlib.collections import LineCollection
from matplotlib.lines import Line2D

from vqrouter.data import load_dataset
from vqrouter.models import build_model

ESTIMATORS = ["ste", "gate_norm", "gate_cb", "dense_st"]
# papéis de cor (paleta de referência, modo claro): tinta, cinza de fundo, azul = codewords
INK, MUTED, GRID, BLUE = "#0b0b0b", "#52514e", "#e1e0db", "#2a78d6"
BLUES = matplotlib.colormaps["Blues"]

plt.rcParams.update(
    {
        "font.size": 9,
        "pdf.fonttype": 42,
        "axes.edgecolor": MUTED,
        "axes.labelcolor": MUTED,
        "xtick.color": MUTED,
        "ytick.color": MUTED,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.color": GRID,
        "grid.linewidth": 0.5,
    }
)


def run_path(root: Path, exp: str, estimator: str, seed: int) -> Path:
    hits = sorted((root / exp).glob(f"{estimator}_K*_s{seed}"))
    if len(hits) != 1:
        raise SystemExit(f"esperava 1 execução de {estimator} seed {seed} em {root / exp}: {hits}")
    return hits[0]


def load_run(rd: Path):
    cfg = yaml.safe_load((rd / "config.yaml").read_text())
    data = load_dataset(cfg["dataset"], cfg.get("data", {}))
    vq_kwargs = {**cfg["vq"], "estimator": cfg["estimator"], "seed": cfg["seed"]}
    extra = {k: v for k, v in cfg["model"].items() if k != "name"}
    model = build_model(cfg["model"]["name"], data.in_shape, vq_kwargs, **extra)
    ck = torch.load(rd / "ckpt.pt", map_location="cpu", weights_only=False)
    model.load_state_dict(ck["model"])
    return model.eval(), data


def grid_field(model, data, n=17, pad=0.15):
    """Retorna latentes de teste, codewords, pontos da grade e o campo -dL/dz."""
    x_test = data.test
    with torch.no_grad():
        z_test = model.encoder(x_test)
    codes = model.vq.codebook.detach()
    pts = torch.cat([z_test, codes])
    lo, hi = pts.min(0).values, pts.max(0).values
    span = hi - lo
    lo, hi = lo - pad * span, hi + pad * span
    gx, gy = torch.meshgrid(
        torch.linspace(lo[0], hi[0], n), torch.linspace(lo[1], hi[1], n), indexing="xy"
    )
    grid = torch.stack([gx.ravel(), gy.ravel()], 1)
    target = x_test[torch.cdist(grid, z_test).argmin(1)]
    z = grid.clone().requires_grad_(True)
    out = model.vq(z)
    loss = ((model.decoder(out.z_q) - target) ** 2).sum()  # soma: as linhas são independentes
    loss.backward()
    spacing = float((hi - lo).min() / (n - 1))
    return z_test.numpy(), codes.numpy(), grid.numpy(), (-z.grad).numpy(), spacing, (lo, hi)


def style_axes(ax, bounds, title):
    lo, hi = bounds
    ax.set_xlim(float(lo[0]), float(hi[0]))
    ax.set_ylim(float(lo[1]), float(hi[1]))
    ax.set_title(title, loc="left", color=INK, fontsize=10)
    ax.set_xlabel("$z_1$")
    ax.set_ylabel("$z_2$")


def fig_field(root, exp, seed, out):
    panels = {e: grid_field(*load_run(run_path(root, exp, e, seed))) for e in ESTIMATORS}
    fig, axes = plt.subplots(1, 4, figsize=(13, 3.6))
    for ax, (est, (zt, codes, grid, g, spacing, bounds)) in zip(axes, panels.items(), strict=True):
        ax.scatter(*zt.T, s=3, color=GRID, zorder=1, linewidths=0)
        norm = np.linalg.norm(g, axis=1)
        s = np.percentile(norm, 95) / spacing  # comprimento da seta = norma / s, um s por painel
        ax.quiver(
            *grid.T, *g.T, angles="xy", scale_units="xy", scale=s, color=INK, width=0.004, zorder=2
        )
        ax.scatter(*codes.T, s=36, color=BLUE, edgecolor="white", linewidth=1.0, zorder=3)
        style_axes(ax, bounds, est)
        ax.set_title(
            f"median $\\|g_z\\|$ = {np.median(norm):.1e}", loc="right", fontsize=8, color=MUTED
        )
    handles = [
        Line2D([], [], marker="o", ls="", color=GRID, markersize=4, label="encoder outputs (test)"),
        Line2D(
            [],
            [],
            marker="o",
            ls="",
            color=BLUE,
            markeredgecolor="white",
            markersize=7,
            label="codewords",
        ),
        Line2D([], [], color=INK, lw=1.5, label=r"$-\nabla_z L_{rec}$ (arrow scale set per panel)"),
    ]
    fig.legend(handles=handles, loc="lower center", ncol=3, frameon=False, labelcolor=MUTED)
    fig.tight_layout(rect=(0, 0.08, 1, 0.97))
    fig.savefig(out)
    plt.close(fig)


def fig_traj(root, exp, seed, out):
    fig, axes = plt.subplots(1, 4, figsize=(13, 3.6))
    steps_max = 1
    for ax, est in zip(axes, ESTIMATORS, strict=True):
        rd = run_path(root, exp, est, seed)
        model, data = load_run(rd)
        with torch.no_grad():
            zt = model.encoder(data.test).numpy()
        tr = np.load(rd / "codebook_traj.npz")
        steps, books = tr["steps"], tr["codebooks"]  # (T,), (T, K, 2)
        steps_max = max(steps_max, int(steps[-1]))
        pts = np.concatenate([zt, books.reshape(-1, 2)])
        lo, hi = pts.min(0), pts.max(0)
        pad = 0.08 * (hi - lo)
        ax.scatter(*zt.T, s=3, color=GRID, zorder=1, linewidths=0)
        t = steps / steps[-1]
        for k in range(books.shape[1]):
            seg = np.stack([books[:-1, k], books[1:, k]], 1)
            lc = LineCollection(seg, colors=BLUES(0.3 + 0.7 * t[1:]), linewidths=1.4, zorder=2)
            ax.add_collection(lc)
        ax.scatter(*books[0].T, s=18, facecolor="white", edgecolor=MUTED, linewidth=0.8, zorder=3)
        ax.scatter(*books[-1].T, s=36, color=BLUE, edgecolor="white", linewidth=1.0, zorder=4)
        style_axes(ax, (torch.tensor(lo - pad), torch.tensor(hi + pad)), est)
    sm = plt.cm.ScalarMappable(
        cmap=matplotlib.colors.ListedColormap(BLUES(np.linspace(0.3, 1, 256)))
    )
    sm.set_clim(0, steps_max)
    fig.tight_layout(rect=(0, 0.17, 1, 0.97))
    cax = fig.add_axes((0.62, 0.1, 0.2, 0.02))
    cb = fig.colorbar(sm, cax=cax, orientation="horizontal")
    cb.set_label("training step", color=MUTED)
    cb.outline.set_visible(False)
    cax.grid(False)
    handles = [
        Line2D(
            [],
            [],
            marker="o",
            ls="",
            color=GRID,
            markersize=4,
            label="encoder outputs (test, final)",
        ),
        Line2D(
            [],
            [],
            marker="o",
            ls="",
            markerfacecolor="white",
            markeredgecolor=MUTED,
            label="codeword at init",
        ),
        Line2D(
            [],
            [],
            marker="o",
            ls="",
            color=BLUE,
            markeredgecolor="white",
            markersize=7,
            label="codeword at end",
        ),
    ]
    fig.legend(
        handles=handles,
        loc="lower left",
        bbox_to_anchor=(0.03, 0.0),
        ncol=3,
        frameon=False,
        labelcolor=MUTED,
    )
    fig.savefig(out)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--exp", default="khipu_toy")
    ap.add_argument("--runs", default="runs")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="paper/khipu/figures")
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    fig_field(Path(args.runs), args.exp, args.seed, out / f"fig_toy_field_s{args.seed}.pdf")
    fig_traj(Path(args.runs), args.exp, args.seed, out / f"fig_toy_traj_s{args.seed}.pdf")
    print(f"figuras em {out}/")


if __name__ == "__main__":
    main()
