"""Escolhe o tau de cada (estimador, K) por VALIDAÇÃO (runs/khipu_val), sem tocar no teste.

    uv run python scripts/select_tau.py

REGRA, fixada em notes/followup_ressalvas.md ANTES de ver os dados. Por (estimador, K):
  1. só entram os tau sem nenhuma seed "falha" (MSE de validação >= 0,95 x o MSE de prever a
     imagem média do treino, medido na própria validação) e com todas as seeds esperadas;
  2. melhor = o menor MSE de validação médio sobre as seeds;
  3. candidatos = os tau elegíveis com média <= média do melhor + 1 desvio amostral do melhor;
  4. escolhido = o MENOR tau entre os candidatos.
Grava configs/khipu/main_tau_val.yaml (by_estimator_k: estimador -> K -> tau) e
paper/khipu/tables/table_tau_val.csv/.tex. As métricas "test_*" dessas execuções são de
VALIDAÇÃO (data.holdout separa imagens do treino).
"""

import argparse
import csv
import sys
from collections import defaultdict
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent))
import aggregate as agg  # noqa: E402

from vqrouter.data import load_dataset  # noqa: E402

FAILED_FRACTION = agg.FAILED_FRACTION


def val_baseline(data_root: str, holdout: int) -> float:
    """MSE na validação de prever a imagem média do treino (sem a validação)."""
    d = load_dataset("cifar10", {"root": data_root, "holdout": holdout})
    train, val = d.train.float() / 255.0, d.test.float() / 255.0
    return float(((val - train.mean(0, keepdim=True)) ** 2).mean())


def table_rows(runs: list[dict], baseline: float) -> list[dict]:
    groups = defaultdict(list)
    for r in runs:
        groups[(r["estimator"], r["K"], r["tau"])].append(r)
    rows = []
    for (est, k, tau), rs in groups.items():
        mean, std = agg.mean_std([r["test_mse"] for r in rs])
        rows.append(
            {
                "estimator": est,
                "K": k,
                "tau": tau,
                "n_seeds": len(rs),
                "val_mse_mean": mean,
                "val_mse_std": std,
                "n_failed": sum(r["test_mse"] >= FAILED_FRACTION * baseline for r in rs),
            }
        )
    return sorted(rows, key=lambda r: (r["K"], agg.ORDER.index(r["estimator"]), r["tau"]))


def choose(rows: list[dict], n_seeds: int) -> dict[tuple[str, int], float]:
    """Regra pré-registrada por (estimador, K). Erro se alguma célula não tiver tau elegível."""
    chosen = {}
    cells = {(r["estimator"], r["K"]) for r in rows}
    for est, k in sorted(cells, key=lambda c: (c[1], agg.ORDER.index(c[0]))):
        ok = [
            r
            for r in rows
            if (r["estimator"], r["K"]) == (est, k)
            and r["n_failed"] == 0
            and r["n_seeds"] == n_seeds
        ]
        if not ok:
            raise SystemExit(f"{est} K={k}: nenhum tau elegível (sem falhas, {n_seeds} seeds)")
        best = min(ok, key=lambda r: r["val_mse_mean"])
        limit = best["val_mse_mean"] + best["val_mse_std"]
        chosen[(est, k)] = min(r["tau"] for r in ok if r["val_mse_mean"] <= limit)
    return chosen


def tex_table(rows: list[dict], chosen: dict[tuple[str, int], float]) -> str:
    head = (
        r"\begin{tabular}{rllcccc}" "\n" r"\toprule" "\n"
        r"$K$ & Estimator & $\tau$ & $n$ & Val.\ MSE ($\times 10^{-3}$) & Failed$^\dagger$ & "
        r"Chosen \\" "\n" r"\midrule" "\n"
    )  # fmt: skip
    lines, last_k, last_cell = [], None, None
    for r in rows:
        cell = (r["K"], r["estimator"])
        if last_k is not None and r["K"] != last_k:
            lines.append(r"\midrule")
        k = str(r["K"]) if r["K"] != last_k else ""
        name = agg.tex_name(r["estimator"]) if cell != last_cell else ""
        last_k, last_cell = r["K"], cell
        star = r"$\star$" if chosen.get((r["estimator"], r["K"])) == r["tau"] else ""
        mse = agg.pm(r["val_mse_mean"], r["val_mse_std"], 1e3, 2)
        lines.append(
            f"{k} & {name} & {r['tau']:g} & {r['n_seeds']} & {mse} & "
            f"{r['n_failed']}/{r['n_seeds']} & {star} \\\\"
        )
    return head + "\n".join(lines) + "\n" + r"\bottomrule" "\n" r"\end{tabular}" "\n"


def numbers_val_tex(rows: list[dict], chosen: dict, holdout: int, n_train: int) -> str:
    """Macros LaTeX da varredura (nomes só com letras): \\resVal<Mse|MseSd|Failed><Est>K<A|B>T<A..>,
    com T<A..> = tau em ordem crescente (\\resTauA = o menor), e \\resChosen<Est>K<A|B>."""
    ks = sorted({r["K"] for r in rows})
    taus = sorted({r["tau"] for r in rows})
    ktag = {k: "K" + agg.TAGS[i] for i, k in enumerate(ks)}
    ttag = {t: "T" + agg.TAGS[i] for i, t in enumerate(taus)}
    defs = []

    def d(name: str, value: str) -> None:
        defs.append(f"\\newcommand{{\\res{name}}}{{{value}}}")

    for t in taus:
        d(f"Tau{ttag[t][1:]}", f"{t:g}")
    d("Holdout", str(holdout))
    d("ValTrain", str(n_train))
    for r in rows:
        key = f"{agg.CAMEL[r['estimator']]}{ktag[r['K']]}{ttag[r['tau']]}"
        d(f"ValMse{key}", f"{r['val_mse_mean'] * 1e3:.2f}")
        d(f"ValMseSd{key}", f"{r['val_mse_std'] * 1e3:.2f}")
        d(f"ValFailed{key}", f"{r['n_failed']}/{r['n_seeds']}")
    for (est, k), tau in chosen.items():
        d(f"Chosen{agg.CAMEL[est]}{ktag[k]}", f"{tau:g}")
    # menor tau a partir do qual TODAS as seeds falham, para esse tau e todos os maiores
    for est, k in sorted({(r["estimator"], r["K"]) for r in rows}):
        cell = sorted(
            (r for r in rows if (r["estimator"], r["K"]) == (est, k)), key=lambda r: r["tau"]
        )
        tail = []
        for r in reversed(cell):
            if r["n_failed"] != r["n_seeds"]:
                break
            tail.append(r["tau"])
        if tail:
            d(f"AllFailFrom{agg.CAMEL[est]}{ktag[k]}", f"{min(tail):g}")
    return "% gerado por scripts/select_tau.py: não editar à mão\n" + "\n".join(defs) + "\n"


def tau_yaml(chosen: dict[tuple[str, int], float]) -> str:
    by = defaultdict(dict)
    for (est, k), tau in chosen.items():
        by[est][int(k)] = float(tau)
    return (
        "# tau por (estimador, K) escolhido por VALIDAÇÃO (scripts/select_tau.py; regra em\n"
        "# notes/followup_ressalvas.md). ste e ste_ema não usam tau (valem o default).\n"
        + yaml.safe_dump({"default": 1.0, "by_estimator_k": dict(by)})
    )


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--runs", default="runs/khipu_val")
    ap.add_argument("--holdout", type=int, default=5000)
    ap.add_argument("--seeds", type=int, default=3, help="seeds esperadas por tau")
    ap.add_argument("--data-root", default="data")
    ap.add_argument("--out-yaml", default="configs/khipu/main_tau_val.yaml")
    ap.add_argument("--out-tables", default="paper/khipu/tables")
    args = ap.parse_args()

    runs = agg.load_runs(Path(args.runs))
    for d in Path(args.runs).glob("*/config.yaml"):  # confere que TODAS usaram a validação
        got = int(yaml.safe_load(d.read_text()).get("data", {}).get("holdout", 0))
        if got != args.holdout:
            raise SystemExit(f"{d.parent.name}: data.holdout = {got}, esperado {args.holdout}")
    baseline = val_baseline(args.data_root, args.holdout)
    rows = table_rows(runs, baseline)
    chosen = choose(rows, args.seeds)

    out = Path(args.out_tables)
    out.mkdir(parents=True, exist_ok=True)
    with (out / "table_tau_val.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    (out / "table_tau_val.tex").write_text(tex_table(rows, chosen))
    n_train = len(load_dataset("cifar10", {"root": args.data_root, "holdout": args.holdout}).train)
    (out / "numbers_val.tex").write_text(numbers_val_tex(rows, chosen, args.holdout, n_train))
    Path(args.out_yaml).write_text(tau_yaml(chosen))

    limite = FAILED_FRACTION * baseline
    print(f"baseline (validação): MSE {baseline:.5f}; falha = MSE >= {limite:.5f}")
    for r in rows:
        mark = "  <-- escolhido" if chosen.get((r["estimator"], r["K"])) == r["tau"] else ""
        print(
            f"  K={r['K']:<5d} {r['estimator']:9s} tau={r['tau']:<5g} n={r['n_seeds']} "
            f"val_mse={r['val_mse_mean']:.5f} ± {r['val_mse_std']:.5f} "
            f"falhas={r['n_failed']}{mark}"
        )
    print("escolhidos:", {f"{e} K={k}": t for (e, k), t in chosen.items()})


if __name__ == "__main__":
    main()
