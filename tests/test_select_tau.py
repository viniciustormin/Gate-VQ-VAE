"""select_tau.py: a regra pré-registrada (elegibilidade, melhor, +1 desvio, menor tau)."""

import importlib.util
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("select_tau", ROOT / "scripts" / "select_tau.py")
sel = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(sel)


def row(est, tau, mean, std, failed=0, n=3, k=1024):
    return {
        "estimator": est,
        "K": k,
        "tau": tau,
        "n_seeds": n,
        "val_mse_mean": mean,
        "val_mse_std": std,
        "n_failed": failed,
    }


def test_picks_the_lowest_mean_when_it_is_clearly_best():
    rows = [
        row("gate_cb", 0.1, 0.0050, 0.0001),
        row("gate_cb", 1.0, 0.0040, 0.0001),
        row("gate_cb", 3.0, 0.0060, 0.0001),
    ]
    assert sel.choose(rows, 3) == {("gate_cb", 1024): 1.0}


def test_tie_within_one_std_goes_to_the_smaller_tau():
    # melhor = tau 1 (0.0040 ± 0.0004); tau 0.1 está a 0.0003 dele (< 1 desvio) -> o menor ganha
    rows = [row("dense_st", 0.1, 0.0043, 0.0002), row("dense_st", 1.0, 0.0040, 0.0004)]
    assert sel.choose(rows, 3) == {("dense_st", 1024): 0.1}
    # fora de 1 desvio: o melhor fica
    rows = [row("dense_st", 0.1, 0.0050, 0.0002), row("dense_st", 1.0, 0.0040, 0.0004)]
    assert sel.choose(rows, 3) == {("dense_st", 1024): 1.0}


def test_taus_with_a_failed_seed_or_missing_seeds_are_not_eligible():
    rows = [
        row("dense_st", 0.03, 0.0010, 0.0001, failed=1),  # melhor MSE, mas uma seed falhou
        row("dense_st", 0.1, 0.0035, 0.0001, n=2),  # só 2 seeds
        row("dense_st", 0.3, 0.0038, 0.0001),
    ]
    assert sel.choose(rows, 3) == {("dense_st", 1024): 0.3}


def test_no_eligible_tau_is_an_error():
    with pytest.raises(SystemExit, match="nenhum tau elegível"):
        sel.choose([row("gate_norm", 1.0, 0.006, 0.0001, failed=2)], 3)


def test_rule_is_applied_per_estimator():
    rows = [
        row("gate_norm", 1.0, 0.006, 0.0001),
        row("gate_norm", 0.1, 0.011, 0.0001),
        row("dense_st", 0.1, 0.0033, 0.0001),
        row("dense_st", 1.0, 0.0034, 0.0001),
    ]
    assert sel.choose(rows, 3) == {("gate_norm", 1024): 1.0, ("dense_st", 1024): 0.1}


def test_rule_is_applied_per_k_and_the_best_tau_can_differ():
    rows = [
        row("dense_st", 0.1, 0.0035, 0.0001, k=1024),
        row("dense_st", 0.01, 0.0050, 0.0001, k=1024),
        row("dense_st", 0.1, 0.0040, 0.0001, k=8192),
        row("dense_st", 0.01, 0.0030, 0.0001, k=8192),
    ]
    assert sel.choose(rows, 3) == {("dense_st", 1024): 0.1, ("dense_st", 8192): 0.01}


def test_yaml_roundtrips_into_aggregate_chosen_tau():
    import yaml

    chosen = {("dense_st", 1024): 0.1, ("dense_st", 8192): 0.01, ("gate_cb", 1024): 1.0}
    cfg = yaml.safe_load(sel.tau_yaml(chosen))
    agg = sel.agg
    assert (
        agg.chosen_tau(cfg, "dense_st", 8192) == 0.01
        and agg.chosen_tau(cfg, "dense_st", 1024) == 0.1
    )
    assert agg.chosen_tau(cfg, "gate_cb", 1024) == 1.0
    assert agg.chosen_tau(cfg, "ste", 1024) == 1.0  # ste não usa tau: o default
    # sem entrada para o K pedido, cai no by_estimator (formato antigo de main_tau.yaml)
    old = {"default": 1.0, "by_estimator": {"dense_st": 0.1}}
    assert agg.chosen_tau(old, "dense_st", 8192) == 0.1
