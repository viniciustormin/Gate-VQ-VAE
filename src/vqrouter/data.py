"""Datasets (registro extensível em `DATASETS`) e amostragem determinística de batches.

Os dados ficam inteiros num tensor no device. A ordem dos batches depende só de (seed, passo):
uma permutação por época, semeada por `seed` e pela época. Assim retomar de um checkpoint não
precisa guardar estado de DataLoader.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import torch


@dataclass
class Data:
    train: torch.Tensor  # imagens: uint8 (N, C, H, W); toy: float32 (N, 2)
    test: torch.Tensor
    scale: float  # divisor para levar a float (255 em imagens, 1 no toy)

    @property
    def in_shape(self) -> tuple[int, ...]:
        return tuple(self.train.shape[1:])

    def to(self, device) -> Data:
        return Data(self.train.to(device), self.test.to(device), self.scale)

    def float(self, x: torch.Tensor) -> torch.Tensor:
        return x.float() / self.scale


DatasetBuilder = Callable[[dict], Data]
DATASETS: dict[str, DatasetBuilder] = {}


def register_dataset(name: str):
    def deco(fn: DatasetBuilder) -> DatasetBuilder:
        if name in DATASETS:
            raise ValueError(f"dataset já registrado: {name}")
        DATASETS[name] = fn
        return fn

    return deco


def _ring_gaussians(n: int, k: int, radius: float, sigma: float, seed: int) -> torch.Tensor:
    g = torch.Generator().manual_seed(seed)
    comp = torch.randint(0, k, (n,), generator=g)
    ang = 2 * math.pi * comp.float() / k
    centers = radius * torch.stack([ang.cos(), ang.sin()], dim=1)
    return centers + sigma * torch.randn(n, 2, generator=g)


@register_dataset("toy2d")
def toy2d(cfg: dict) -> Data:
    """k gaussianas isotrópicas igualmente espaçadas num anel."""
    k, radius, sigma = cfg.get("components", 8), cfg.get("radius", 2.0), cfg.get("sigma", 0.1)
    seed = cfg.get("data_seed", 0)
    train = _ring_gaussians(cfg.get("n_train", 8192), k, radius, sigma, seed)
    test = _ring_gaussians(cfg.get("n_test", 2048), k, radius, sigma, seed + 1)
    return Data(train, test, 1.0)


def _torchvision(cls_name: str, cfg: dict) -> Data:
    import torchvision

    root = Path(cfg.get("root", "data"))
    cls = getattr(torchvision.datasets, cls_name)
    imgs = []
    for train in (True, False):
        try:
            ds = cls(root=str(root), train=train, download=False)
        except RuntimeError as exc:
            raise RuntimeError(
                f"{cls_name} não encontrado em {root}/. Baixe com: "
                f"uv run python scripts/download_data.py --datasets {cls_name.lower()}"
            ) from exc
        x = torch.as_tensor(ds.data)
        x = x[:, None] if x.ndim == 3 else x.permute(0, 3, 1, 2)  # -> (N, C, H, W)
        imgs.append(x.contiguous())
    return Data(imgs[0], imgs[1], 255.0)


@register_dataset("mnist")
def mnist(cfg: dict) -> Data:
    return _torchvision("MNIST", cfg)


@register_dataset("cifar10")
def cifar10(cfg: dict) -> Data:
    return _torchvision("CIFAR10", cfg)


def load_dataset(name: str, cfg: dict) -> Data:
    if name not in DATASETS:
        raise KeyError(f"dataset desconhecido: {name!r}. Registrados: {sorted(DATASETS)}")
    return DATASETS[name](cfg)


@lru_cache(maxsize=4)
def _perm(epoch: int, n: int, seed: int) -> torch.Tensor:
    g = torch.Generator().manual_seed(seed * 7919 + epoch)
    return torch.randperm(n, generator=g)


def batch_indices(step: int, batch_size: int, n: int, seed: int) -> torch.Tensor:
    """Índices do batch do `step` (0-based), contínuos através das épocas."""
    out, pos = [], step * batch_size
    while sum(len(o) for o in out) < batch_size:
        epoch, off = divmod(pos, n)
        take = min(batch_size - sum(len(o) for o in out), n - off)
        out.append(_perm(epoch, n, seed)[off : off + take])
        pos += take
    return torch.cat(out)
