"""Treino: retomada reproduz o treino contínuo, execução concluída é pulada, config protegida."""

import csv
import json
from pathlib import Path

import pytest
import torch

from vqrouter.train import load_config, run_dir, train

ROOT = Path(__file__).resolve().parents[1]


def cfg_for(tmp_path, estimator="gate_cb", exp="t", max_steps=40, **extra):
    over = [
        f"estimator={estimator}",
        f"exp={exp}",
        f"runs_dir={tmp_path}",
        "device=cpu",
        f"train.max_steps={max_steps}",
        "train.eval_every=10",
        "train.batch_size=64",
        "vq.num_codes=8",
        "vq.tau_scale=median",
        "data.n_train=300",
        "data.n_test=100",
    ]
    over += [f"{k}={v}" for k, v in extra.items()]
    return load_config(ROOT / "configs/base/toy2d.yaml", over)


def rows_without_time(rd):
    with (rd / "metrics.csv").open() as f:
        return [{k: v for k, v in r.items() if k != "wall_s"} for r in csv.DictReader(f)]


def assert_same_tree(a, b):
    assert a.keys() == b.keys()
    for k in a:
        if isinstance(a[k], dict):
            assert_same_tree(a[k], b[k])
        elif isinstance(a[k], torch.Tensor):
            assert torch.equal(a[k], b[k]), k
        else:
            assert a[k] == b[k], k


@pytest.mark.parametrize("estimator", ["ste", "ste_ema", "gate_norm", "gate_cb", "dense_st"])
def test_e_resume_reproduces_continuous_training(tmp_path, estimator):
    cont = train(cfg_for(tmp_path, estimator, exp="cont"))
    # metade do treino, depois retoma até o fim estendendo max_steps
    half = cfg_for(tmp_path, estimator, exp="res", max_steps=20)
    rd = train(half)
    assert json.loads((rd / "done.json").read_text())["step"] == 20
    assert run_dir(half) == rd
    train(cfg_for(tmp_path, estimator, exp="res", max_steps=40), resume=True)

    a = torch.load(cont / "ckpt.pt", weights_only=False)
    b = torch.load(rd / "ckpt.pt", weights_only=False)
    assert a["step"] == b["step"] == 40
    assert_same_tree(a["model"], b["model"])
    assert_same_tree(a["optimizer"], b["optimizer"])
    assert a["tau_eff"] == b["tau_eff"]
    assert rows_without_time(cont) == rows_without_time(rd)


def test_completed_run_is_skipped_and_incomplete_needs_resume(tmp_path, capsys):
    cfg = cfg_for(tmp_path)
    rd = train(cfg)
    before = (rd / "ckpt.pt").stat().st_mtime_ns
    train(cfg)
    assert "[skip]" in capsys.readouterr().out
    assert (rd / "ckpt.pt").stat().st_mtime_ns == before
    with pytest.raises(SystemExit):  # max_steps maior sem --resume não sobrescreve
        train(cfg_for(tmp_path, max_steps=60))


def test_resume_refuses_changed_config(tmp_path):
    train(cfg_for(tmp_path, max_steps=20))
    with pytest.raises(SystemExit, match="train.lr"):
        train(cfg_for(tmp_path, max_steps=40, **{"train.lr": 1e-3}), resume=True)


def test_run_dir_layout_and_artifacts(tmp_path):
    rd = train(cfg_for(tmp_path, "gate_norm", exp="khipu"))
    assert rd.parent.name == "khipu" and rd.name == "gate_norm_K8_tau1med_s0"
    for f in ["config.yaml", "metrics.csv", "commit.txt", "ckpt.pt", "done.json"]:
        assert (rd / f).exists(), f
    ck = torch.load(rd / "ckpt.pt", weights_only=False)
    assert {"step", "model", "optimizer", "rng", "tau_eff"} <= ck.keys()
    assert ck["tau_eff"] > 0


def test_codebook_trajectory_survives_resume(tmp_path):
    import numpy as np

    extra = {"train.codebook_log_every": 5}
    cont = train(cfg_for(tmp_path, exp="tc", **extra))
    train(cfg_for(tmp_path, exp="tr", max_steps=20, **extra))
    rd = train(cfg_for(tmp_path, exp="tr", max_steps=40, **extra), resume=True)
    a, b = np.load(cont / "codebook_traj.npz"), np.load(rd / "codebook_traj.npz")
    assert list(a["steps"]) == list(range(0, 41, 5)) == list(b["steps"])
    assert np.array_equal(a["codebooks"], b["codebooks"])
    assert a["codebooks"].shape == (9, 8, 2)
