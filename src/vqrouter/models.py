"""VQ-VAEs pequenos: conv (imagens) e MLP (toy 2D). Registro extensível em `MODELS`."""

from __future__ import annotations

from collections.abc import Callable

import torch
from torch import nn

from vqrouter.quantizers import VectorQuantizer, VQOutput


class ResBlock(nn.Module):
    def __init__(self, ch: int):
        super().__init__()
        self.body = nn.Sequential(
            nn.ReLU(), nn.Conv2d(ch, ch, 3, padding=1), nn.ReLU(), nn.Conv2d(ch, ch, 1)
        )

    def forward(self, x):
        return x + self.body(x)


class ConvEncoder(nn.Module):
    """2 convs 4x4 stride 2 + 2 blocos residuais + projeção 1x1 para d."""

    def __init__(self, in_ch: int, hidden: int, dim: int, num_res: int = 2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, hidden, 4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(hidden, hidden, 4, stride=2, padding=1),
            *[ResBlock(hidden) for _ in range(num_res)],
            nn.ReLU(),
            nn.Conv2d(hidden, dim, 1),
        )

    def forward(self, x):
        return self.net(x)


class ConvDecoder(nn.Module):
    """Espelho do encoder: 1x1 -> blocos residuais -> 2 ConvTranspose 4x4 stride 2."""

    def __init__(self, out_ch: int, hidden: int, dim: int, num_res: int = 2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(dim, hidden, 1),
            *[ResBlock(hidden) for _ in range(num_res)],
            nn.ReLU(),
            nn.ConvTranspose2d(hidden, hidden, 4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(hidden, out_ch, 4, stride=2, padding=1),
        )

    def forward(self, z):
        return self.net(z)


class VQVAE(nn.Module):
    """x -> encoder -> VQ (por token) -> decoder. `forward` devolve (reconstrução, VQOutput)."""

    def __init__(
        self, encoder: nn.Module, decoder: nn.Module, vq: VectorQuantizer, channels_last: bool
    ):
        super().__init__()
        self.encoder, self.decoder, self.vq = encoder, decoder, vq
        self.channels_last = channels_last  # True: z (B, d, H, W); False: z (B, d)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, VQOutput]:
        z = self.encoder(x)
        if self.channels_last:
            b, d, h, w = z.shape
            flat = z.permute(0, 2, 3, 1).reshape(-1, d)
        else:
            flat = z
        out = self.vq(flat)
        z_q = out.z_q
        if self.channels_last:
            z_q = z_q.reshape(b, h, w, d).permute(0, 3, 1, 2)
        return self.decoder(z_q), out


ModelBuilder = Callable[..., VQVAE]
MODELS: dict[str, ModelBuilder] = {}


def register_model(name: str):
    def deco(fn: ModelBuilder) -> ModelBuilder:
        if name in MODELS:
            raise ValueError(f"modelo já registrado: {name}")
        MODELS[name] = fn
        return fn

    return deco


@register_model("conv_vqvae")
def conv_vqvae(in_shape, vq_kwargs, hidden=128, dim=64, num_res=2) -> VQVAE:
    in_ch = in_shape[0]
    vq = VectorQuantizer(dim=dim, **vq_kwargs)
    return VQVAE(
        ConvEncoder(in_ch, hidden, dim, num_res), ConvDecoder(in_ch, hidden, dim, num_res), vq, True
    )


@register_model("mlp_vqvae")
def mlp_vqvae(in_shape, vq_kwargs, hidden=64, dim=2) -> VQVAE:
    n = in_shape[0]
    enc = nn.Sequential(
        nn.Linear(n, hidden),
        nn.ReLU(),
        nn.Linear(hidden, hidden),
        nn.ReLU(),
        nn.Linear(hidden, dim),
    )
    dec = nn.Sequential(
        nn.Linear(dim, hidden),
        nn.ReLU(),
        nn.Linear(hidden, hidden),
        nn.ReLU(),
        nn.Linear(hidden, n),
    )
    return VQVAE(enc, dec, VectorQuantizer(dim=dim, **vq_kwargs), False)


def build_model(name: str, in_shape, vq_kwargs: dict, **model_kwargs) -> VQVAE:
    if name not in MODELS:
        raise KeyError(f"modelo desconhecido: {name!r}. Registrados: {sorted(MODELS)}")
    return MODELS[name](in_shape, vq_kwargs, **model_kwargs)
