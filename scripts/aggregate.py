"""Agrega runs/khipu em tabelas (CSV e LaTeX) para o paper. TODO número do texto vem daqui.

    uv run python scripts/aggregate.py                      # runs/khipu -> paper/khipu/tables/

Gera em --out:
    runs_all.csv      uma linha por execução concluída (inclui o tau e o tempo)
    table_main.csv/.tex   por (K, estimador) no tau escolhido de configs/khipu/main_tau.yaml:
                          média e desvio amostral (ddof = 1) sobre as seeds
    table_pilot.csv/.tex  piloto: K = 1024, seed 0, todos os tau das execuções gate_* e dense_st
    numbers.tex       macros \\resMseSteKA etc. com os números citados no texto do paper (o artigo
                      usa só macros, nunca número digitado). KA, KB... = K em ordem crescente

Duas contagens de falha, ambas DESCRITIVAS e definidas DEPOIS de ver os dados (não são limiares
de decisão):
    failed          MSE final de teste >= 0,95 x o MSE de prever sempre a imagem média do treino
                    (baseline calculado dos dados, em baseline.json): a reconstrução não
                    aprendeu
    code_collapsed  perplexidade final < 2: o codebook colapsou em ~1 código
As duas não coincidem: no dense_st com K = 8192 a seed 0 não aprendeu (MSE = baseline) com 69% dos
códigos em uso (perplexidade 751). A tabela principal mostra `failed`; `code_collapsed` fica no CSV.
Execução sem done.json (incompleta) é ignorada com aviso.
"""

import argparse
import csv
import json
import statistics as st
from collections import defaultdict
from pathlib import Path

import yaml

from vqrouter.data import load_dataset

ORDER = ["ste", "ste_ema", "gate_norm", "gate_cb", "dense_st"]
PILOT_TAUS = {0.1, 1.0, 10.0}  # o piloto original; tau novos (reexecuções) não entram
NO_TAU = {"ste", "ste_ema"}  # não usam tau: ele só entra no nome da pasta
COLLAPSE_PERPLEXITY = 2.0
FAILED_FRACTION = 0.95  # de MSE_imagem_média
METRICS = ["test_mse", "test_psnr", "codes_used_pct", "perplexity", "quant_err"]


def load_runs(runs_dir: Path) -> list[dict]:
    runs = []
    for d in sorted(runs_dir.iterdir()):
        if not d.is_dir():
            continue
        if not (d / "done.json").exists():
            print(f"[aviso] {d.name}: sem done.json, ignorada")
            continue
        cfg = yaml.safe_load((d / "config.yaml").read_text())
        done = json.loads((d / "done.json").read_text())
        with (d / "metrics.csv").open() as f:
            rows = list(csv.DictReader(f))
        final = rows[-1]
        if int(final["step"]) != done["step"]:
            raise SystemExit(f"{d.name}: último passo do CSV ({final['step']}) != done.json")
        runs.append(
            {
                "run": d.name,
                "estimator": cfg["estimator"],
                "K": int(cfg["vq"]["num_codes"]),
                "tau": float(cfg["vq"]["tau"]),
                "tau_scale": cfg["vq"].get("tau_scale"),
                "seed": int(cfg["seed"]),
                "step": int(final["step"]),
                "wall_s": float(final["wall_s"]),
                "tau_eff": float(final["tau_eff"]),
                **{m: float(final[m]) for m in METRICS},
            }
        )
    return runs


def mean_image_mse(data_root: str) -> float:
    """MSE no teste (pixels em [0, 1]) de prever sempre a imagem média por pixel do treino."""
    d = load_dataset("cifar10", {"root": data_root})
    train, test = d.train.float() / 255.0, d.test.float() / 255.0
    return float(((test - train.mean(0, keepdim=True)) ** 2).mean())


def chosen_tau(cfg: dict, estimator: str, k: int | None = None) -> float:
    """tau escolhido: por (estimador, K) se `by_estimator_k` o define, senão por estimador."""
    by_k = cfg.get("by_estimator_k", {}).get(estimator, {})
    if k is not None and k in by_k:
        return float(by_k[k])
    return float(cfg.get("by_estimator", {}).get(estimator, cfg["default"]))


def mean_std(values: list[float]) -> tuple[float, float]:
    return st.mean(values), (st.stdev(values) if len(values) > 1 else float("nan"))


def is_failed(run: dict, baseline: float) -> bool:
    return run["test_mse"] >= FAILED_FRACTION * baseline


def main_rows(runs: list[dict], tau_cfg: dict, baseline: float) -> list[dict]:
    groups = defaultdict(list)
    for r in runs:
        if r["tau"] == chosen_tau(tau_cfg, r["estimator"], r["K"]):
            groups[(r["K"], r["estimator"])].append(r)
    rows = []
    for (k, est), rs in sorted(groups.items(), key=lambda kv: (kv[0][0], ORDER.index(kv[0][1]))):
        row = {"K": k, "estimator": est, "tau": rs[0]["tau"], "n_seeds": len(rs)}
        row["seeds"] = " ".join(str(r["seed"]) for r in sorted(rs, key=lambda r: r["seed"]))
        for m in METRICS:
            mean, std = mean_std([r[m] for r in rs])
            row[f"{m}_mean"], row[f"{m}_std"] = mean, std
        row["n_failed"] = sum(is_failed(r, baseline) for r in rs)
        row["n_code_collapsed"] = sum(r["perplexity"] < COLLAPSE_PERPLEXITY for r in rs)
        row["mse_per_seed"] = " ".join(
            f"{r['test_mse']:.6g}" for r in sorted(rs, key=lambda r: r["seed"])
        )
        rows.append(row)
    return rows


def pilot_rows(runs: list[dict]) -> list[dict]:
    rows = [
        r
        for r in runs
        if r["K"] == 1024
        and r["seed"] == 0
        and r["estimator"] not in NO_TAU
        and r["tau"] in PILOT_TAUS
    ]
    return sorted(rows, key=lambda r: (ORDER.index(r["estimator"]), r["tau"]))


def write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


# --------------------------------------------------------------------------- LaTeX
CAMEL = {"ste": "Ste", "ste_ema": "SteEma", "gate_norm": "GateNorm", "gate_cb": "GateCb",
         "dense_st": "DenseSt"}  # fmt: skip
TAGS = "ABCDEFGH"
NUMBER_METRICS = [  # (nome da macro, coluna, escala, casas decimais)
    ("Mse", "test_mse", 1e3, 2),
    ("Psnr", "test_psnr", 1, 2),
    ("Used", "codes_used_pct", 1, 1),
    ("Perp", "perplexity", 1, 1),
]


def numbers_tex(rows: list[dict], pilot: list[dict], baseline: float, prefix: str = "res") -> str:
    """Macros LaTeX com os números do texto. Nomes só com letras (o LaTeX não aceita dígitos):
    \\res<Métrica><Estimador>K<A|B>, com K<A|B> = K em ordem crescente (\\resKA = menor K)."""
    ks = sorted({r["K"] for r in rows})
    tag = {k: "K" + TAGS[i] for i, k in enumerate(ks)}
    defs = []

    def d(name: str, value: str) -> None:
        defs.append(f"\\newcommand{{\\{prefix}{name}}}{{{value}}}")

    for k in ks:
        d(tag[k], str(k))
    by = {(r["K"], r["estimator"]): r for r in rows}
    for (k, est), r in by.items():
        t = f"{CAMEL[est]}{tag[k]}"
        for name, key, scale, nd in NUMBER_METRICS:
            d(f"{name}{t}", f"{r[key + '_mean'] * scale:.{nd}f}")
            d(f"{name}Sd{t}", f"{r[key + '_std'] * scale:.{nd}f}")
        d(f"Failed{t}", f"{r['n_failed']}/{r['n_seeds']}")
        if est != "ste" and (k, "ste") in by:  # razão das médias em relação ao ste, mesmo K
            ref = by[(k, "ste")]
            for name, key in [
                ("Mse", "test_mse"),
                ("Perp", "perplexity"),
                ("Used", "codes_used_pct"),
            ]:
                d(f"Ratio{name}{t}OverSte", f"{r[key + '_mean'] / ref[key + '_mean']:.2f}")
    d("BaselineMse", f"{baseline * 1e3:.1f}")
    d("FailFraction", f"{FAILED_FRACTION:g}")
    d("NSeeds", str(max(r["n_seeds"] for r in rows)))
    d("PilotRuns", str(len(pilot)))
    d("PilotFailed", str(sum(is_failed(r, baseline) for r in pilot)))
    return "% gerado por scripts/aggregate.py: não editar à mão\n" + "\n".join(defs) + "\n"


def tex_name(est: str) -> str:
    return r"\texttt{" + est.replace("_", r"\_") + "}"


def pm(mean: float, std: float, scale: float = 1.0, nd: int = 2) -> str:
    if std != std:  # NaN: uma seed só
        return f"{mean * scale:.{nd}f}"
    return f"{mean * scale:.{nd}f} $\\pm$ {std * scale:.{nd}f}"


def tex_main(rows: list[dict]) -> str:
    head = (
        r"\begin{tabular}{rllrccccr}" "\n" r"\toprule" "\n"
        r"$K$ & Estimator & $\tau$ & $n$ & MSE ($\times 10^{-3}$) & PSNR (dB) & "
        r"Codes used (\%) & Perplexity & Failed$^\dagger$ \\" "\n" r"\midrule" "\n"
    )  # fmt: skip
    lines, last_k = [], None
    for r in rows:
        if last_k is not None and r["K"] != last_k:
            lines.append(r"\midrule")
        tau = "--" if r["estimator"] in NO_TAU else f"{r['tau']:g}"
        k = str(r["K"]) if r["K"] != last_k else ""
        last_k = r["K"]
        lines.append(
            f"{k} & {tex_name(r['estimator'])} & {tau} & {r['n_seeds']} & "
            f"{pm(r['test_mse_mean'], r['test_mse_std'], 1e3, 2)} & "
            f"{pm(r['test_psnr_mean'], r['test_psnr_std'], 1, 2)} & "
            f"{pm(r['codes_used_pct_mean'], r['codes_used_pct_std'], 1, 1)} & "
            f"{pm(r['perplexity_mean'], r['perplexity_std'], 1, 1)} & "
            f"{r['n_failed']}/{r['n_seeds']} \\\\"
        )
    return head + "\n".join(lines) + "\n" + r"\bottomrule" "\n" r"\end{tabular}" "\n"


def tex_pilot(runs: list[dict], baseline: float) -> str:
    head = (
        r"\begin{tabular}{lrccccc}" "\n" r"\toprule" "\n"
        r"Estimator & $\tau$ ($\times$ median) & MSE ($\times 10^{-3}$) & PSNR (dB) & "
        r"Codes used (\%) & Perplexity & Failed$^\dagger$ \\" "\n" r"\midrule" "\n"
    )  # fmt: skip
    lines, last = [], None
    for r in runs:
        if last is not None and r["estimator"] != last:
            lines.append(r"\midrule")
        name = tex_name(r["estimator"]) if r["estimator"] != last else ""
        last = r["estimator"]
        lines.append(
            f"{name} & {r['tau']:g} & {r['test_mse'] * 1e3:.2f} & {r['test_psnr']:.2f} & "
            f"{r['codes_used_pct']:.1f} & {r['perplexity']:.1f} & "
            f"{'yes' if is_failed(r, baseline) else '--'} \\\\"
        )
    return head + "\n".join(lines) + "\n" + r"\bottomrule" "\n" r"\end{tabular}" "\n"


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--runs", default="runs/khipu")
    ap.add_argument("--tau-config", default="configs/khipu/main_tau.yaml")
    ap.add_argument("--out", default="paper/khipu/tables")
    ap.add_argument("--data-root", default="data", help="onde está o CIFAR-10 (baseline)")
    ap.add_argument("--macro-prefix", default="res", help="prefixo dos macros de numbers.tex")
    args = ap.parse_args()

    runs = load_runs(Path(args.runs))
    if not runs:
        raise SystemExit(f"nenhuma execução concluída em {args.runs}")
    tau_cfg = yaml.safe_load(Path(args.tau_config).read_text())
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    baseline = mean_image_mse(args.data_root)
    (out / "baseline.json").write_text(
        json.dumps({"mean_image_mse": baseline, "failed_fraction": FAILED_FRACTION}, indent=1)
    )
    for r in runs:
        r["failed"] = int(is_failed(r, baseline))
        r["code_collapsed"] = int(r["perplexity"] < COLLAPSE_PERPLEXITY)
    write_csv(
        out / "runs_all.csv",
        sorted(runs, key=lambda r: (r["K"], ORDER.index(r["estimator"]), r["tau"], r["seed"])),
    )
    main = main_rows(runs, tau_cfg, baseline)
    write_csv(out / "table_main.csv", main)
    (out / "table_main.tex").write_text(tex_main(main))
    pilot = pilot_rows(runs)
    write_csv(out / "table_pilot.csv", pilot)
    (out / "table_pilot.tex").write_text(tex_pilot(pilot, baseline))
    (out / "numbers.tex").write_text(numbers_tex(main, pilot, baseline, args.macro_prefix))
    print(f"{len(runs)} execuções; principal {len(main)} linhas, piloto {len(pilot)}; em {out}/")
    limite = FAILED_FRACTION * baseline
    print(f"baseline (imagem média): MSE {baseline:.5f}; falha = MSE >= {limite:.5f}")


if __name__ == "__main__":
    main()
