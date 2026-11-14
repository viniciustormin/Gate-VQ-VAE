"""diag_dense.py treina exatamente como o train.py (mesmos pesos finais), no toy em CPU."""

import importlib.util
from pathlib import Path

import torch

from vqrouter.train import load_config, train

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("diag_dense", ROOT / "scripts" / "diag_dense.py")
diag = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(diag)


def test_diag_loop_matches_train(tmp_path):
    over = [
        "estimator=dense_st", "device=cpu", "train.max_steps=30", "train.eval_every=30",
        "train.batch_size=64", "vq.num_codes=8", "vq.tau_scale=median", "data.n_train=300",
        "data.n_test=100", f"runs_dir={tmp_path}", "exp=eq",
    ]  # fmt: skip
    cfg = load_config(ROOT / "configs/base/toy2d.yaml", over)
    rd = train(cfg)
    ref = torch.load(rd / "ckpt.pt", weights_only=False)["model"]
    rows, model = diag.run_diag(cfg, steps=30, log_every=30)
    got = model.state_dict()
    assert ref.keys() == got.keys()
    assert all(torch.equal(ref[k], got[k]) for k in ref)
    assert rows[-1]["step"] == 30 and 0.0 <= rows[-1]["h_norm"] <= 1.0
