"""data.holdout: validação separada do treino; com holdout = 0 nada muda."""

import numpy as np
import pytest
import torch

import vqrouter.data as data_mod


class FakeCifar:
    """Imita torchvision.datasets.CIFAR10: só .data, 40 imagens de treino e 10 de teste."""

    def __init__(self, root, train, download):
        n = 40 if train else 10
        base = 0 if train else 1000
        # cada imagem tem um valor único no 1º pixel: dá para rastrear quem foi para onde
        self.data = np.zeros((n, 4, 4, 3), dtype=np.uint8)
        self.data[:, 0, 0, 0] = np.arange(n) + (0 if train else 100)
        self.base = base


@pytest.fixture(autouse=True)
def fake_torchvision(monkeypatch):
    import torchvision

    monkeypatch.setattr(torchvision.datasets, "CIFAR10", FakeCifar)


def ids(x):
    return x[:, 0, 0, 0].tolist()


def test_default_is_unchanged():
    d = data_mod.load_dataset("cifar10", {"root": "x"})
    assert len(d.train) == 40 and len(d.test) == 10
    assert ids(d.test) == list(range(100, 110))  # o teste de verdade


def test_holdout_comes_from_train_is_disjoint_and_deterministic():
    cfg = {"root": "x", "holdout": 8}
    d1, d2 = data_mod.load_dataset("cifar10", cfg), data_mod.load_dataset("cifar10", cfg)
    assert len(d1.train) == 32 and len(d1.test) == 8
    assert set(ids(d1.train)).isdisjoint(ids(d1.test))
    assert set(ids(d1.train)) | set(ids(d1.test)) == set(range(40))  # nada do teste real entra
    assert ids(d1.test) == ids(d2.test) and torch.equal(d1.train, d2.train)  # split fixo
    other = data_mod.load_dataset("cifar10", {**cfg, "holdout_seed": 7})
    assert ids(other.test) != ids(d1.test)
