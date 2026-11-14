"""aggregate.py: agrupamento no tau escolhido, desvio amostral, critérios de falha e LaTeX."""

import csv
import importlib.util
import json
import math
import re
import statistics as st
from pathlib import Path

import pytest
import yaml

SPEC = importlib.util.spec_from_file_location(
    "aggregate", Path(__file__).resolve().parents[1] / "scripts" / "aggregate.py"
)
agg = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(agg)

COLS = [
    "step",
    "test_mse",
    "test_psnr",
    "codes_used_pct",
    "perplexity",
    "quant_err",
    "tau_eff",
    "wall_s",
]
BASELINE = 0.06
TAU_CFG = {"default": 1.0, "by_estimator": {"dense_st": 0.1}}


def make_run(root, est, k, tau, seed, mse, perp=100.0, used=50.0, step=20000, done=True):
    name = f"{est}_K{k}_tau{tau:g}med_s{seed}"
    d = root / name
    d.mkdir(parents=True)
    cfg = {
        "estimator": est,
        "seed": seed,
        "vq": {"num_codes": k, "tau": tau, "tau_scale": "median"},
    }
    (d / "config.yaml").write_text(yaml.safe_dump(cfg))
    with (d / "metrics.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=COLS)
        w.writeheader()
        w.writerow(dict(step=step, test_mse=mse, test_psnr=20.0, codes_used_pct=used,
                        perplexity=perp, quant_err=0.1, tau_eff=0.5, wall_s=10.0))  # fmt: skip
    if done:
        (d / "done.json").write_text(json.dumps({"step": step, "wall_s": 10.0}))


@pytest.fixture
def runs_dir(tmp_path):
    root = tmp_path / "khipu"
    root.mkdir()
    for seed, mse in enumerate([0.0036, 0.0040, 0.0044]):
        make_run(root, "ste", 1024, 1.0, seed, mse)
    for seed, mse in enumerate([0.0030, 0.0032, 0.0034]):
        make_run(root, "dense_st", 1024, 0.1, seed, mse)
    make_run(root, "dense_st", 1024, 10.0, 0, 0.0601, perp=1.0)  # tau do piloto, fora da tabela
    return root


def test_main_table_uses_only_the_chosen_tau_and_sample_std(runs_dir):
    runs = agg.load_runs(runs_dir)
    rows = {r["estimator"]: r for r in agg.main_rows(runs, TAU_CFG, BASELINE)}
    assert set(rows) == {"ste", "dense_st"}
    assert rows["dense_st"]["tau"] == 0.1 and rows["dense_st"]["n_seeds"] == 3  # sem o tau = 10
    vals = [0.0036, 0.0040, 0.0044]
    assert rows["ste"]["test_mse_mean"] == pytest.approx(st.mean(vals))
    assert rows["ste"]["test_mse_std"] == pytest.approx(st.stdev(vals))  # ddof = 1
    assert rows["ste"]["test_mse_std"] != pytest.approx(st.pstdev(vals))


def test_single_seed_has_nan_std_and_no_pm_in_latex(tmp_path):
    root = tmp_path / "khipu"
    root.mkdir()
    make_run(root, "gate_cb", 1024, 1.0, 0, 0.0043)
    row = agg.main_rows(agg.load_runs(root), TAU_CFG, BASELINE)[0]
    assert math.isnan(row["test_mse_std"])
    tex = agg.tex_main([row])
    assert "4.30 &" in tex and r"\pm" not in tex.split(r"\midrule")[1]


def test_failure_criteria_are_independent(tmp_path):
    root = tmp_path / "khipu"
    root.mkdir()
    make_run(root, "dense_st", 8192, 0.1, 0, 0.0618, perp=751.0)  # não aprendeu, códigos em uso
    make_run(root, "dense_st", 8192, 0.1, 1, 0.0031, perp=900.0)
    make_run(root, "dense_st", 8192, 0.1, 2, 0.0618, perp=1.0)  # não aprendeu E colapsou
    row = agg.main_rows(agg.load_runs(root), TAU_CFG, baseline=0.0618)[0]
    assert row["n_failed"] == 2 and row["n_code_collapsed"] == 1
    assert agg.is_failed({"test_mse": 0.95 * 0.0618}, 0.0618)  # a fronteira conta como falha
    assert not agg.is_failed({"test_mse": 0.94 * 0.0618}, 0.0618)


def test_incomplete_run_is_ignored_and_step_mismatch_is_an_error(tmp_path, capsys):
    root = tmp_path / "khipu"
    root.mkdir()
    make_run(root, "ste", 1024, 1.0, 0, 0.004)
    make_run(root, "ste", 1024, 1.0, 1, 0.004, done=False)
    assert len(agg.load_runs(root)) == 1 and "sem done.json" in capsys.readouterr().out
    bad = tmp_path / "bad"
    bad.mkdir()
    make_run(bad, "ste", 1024, 1.0, 0, 0.004)
    (bad / "ste_K1024_tau1med_s0" / "done.json").write_text(json.dumps({"step": 999}))
    with pytest.raises(SystemExit, match="último passo"):
        agg.load_runs(bad)


def test_latex_escapes_names_and_marks_tau_unused(runs_dir):
    make_run(runs_dir, "ste_ema", 1024, 1.0, 0, 0.0037)
    rows = agg.main_rows(agg.load_runs(runs_dir), TAU_CFG, BASELINE)
    tex = agg.tex_main(rows)
    assert r"\texttt{ste\_ema} & -- &" in tex  # sem tau; underscore escapado
    assert r"\texttt{dense\_st} & 0.1 &" in tex
    pilot = agg.tex_pilot(agg.pilot_rows(agg.load_runs(runs_dir)), BASELINE)
    assert "yes" in pilot and r"\texttt{ste" not in pilot  # ste não entra no piloto


def test_numbers_macros_match_the_table(runs_dir):
    make_run(runs_dir, "gate_norm", 1024, 1.0, 0, 0.0070)
    runs = agg.load_runs(runs_dir)
    rows = agg.main_rows(runs, TAU_CFG, BASELINE)
    tex = agg.numbers_tex(rows, agg.pilot_rows(runs), BASELINE)
    defs = dict(re.findall(r"\\newcommand\{\\res(\w+)\}\{([^}]*)\}", tex))
    assert defs["KA"] == "1024"
    assert defs["MseSteKA"] == f"{st.mean([0.0036, 0.0040, 0.0044]) * 1e3:.2f}"  # 4.00
    assert defs["FailedSteKA"] == "0/3" and defs["BaselineMse"] == "60.0"
    assert defs["RatioMseGateNormKAOverSte"] == "1.75"  # 7.0 / 4.0, razão das médias
    assert "RatioMseSteKAOverSte" not in defs  # o ste não é comparado consigo mesmo
    # piloto = K 1024, seed 0, sem ste: dense_st tau 0,1 + dense_st tau 10 + gate_norm tau 1;
    # só o dense_st tau 10 (MSE 0,0601 >= 0,95 x 0,06) falha
    assert defs["PilotRuns"] == "3" and defs["PilotFailed"] == "1"
    assert all(re.fullmatch(r"[A-Za-z]+", name) for name in defs)  # só letras: macro válida
