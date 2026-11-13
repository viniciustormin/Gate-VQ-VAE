"""Camada de quantização vetorial e registro de estimadores de gradiente.

O VQ é visto como um MoE degenerado: os codewords são experts constantes e o vizinho mais
próximo é o router top-1. Cada estimador define apenas COMO o gradiente atravessa a decisão
discreta; o forward de todos é exatamente e_k* (k* = argmin_j ||z - e_j||^2).

Notação: z (N, d) saída do encoder, e (K, d) codebook, d2 (N, K) distâncias ao quadrado,
p = softmax(-d2 / tau), sg = stop-gradient (`.detach()`).

Para o forward ser BIT A BIT igual a e_k*, as fórmulas são escritas na forma algebricamente
equivalente `valor.detach() + (x - x.detach())`, cujo forward é exato (x - x.detach() == 0) e
cujo gradiente é o mesmo de `x + sg(valor - x)`. Exemplo: ste usa `e_k.detach() + (z - z.detach())`
no lugar de `z + sg(e_k - z)`, que em ponto flutuante não devolve e_k exatamente.

Para acrescentar um estimador: escreva uma função `(z, codebook, k, d2, tau) -> z_q` e decore com
`@register_estimator("nome", ...)`.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import nn

EstimatorFn = Callable[
    [torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor], torch.Tensor
]


@dataclass(frozen=True)
class Estimator:
    """Um estimador de gradiente e seus padrões.

    codebook_loss: se usa a codebook loss ||sg(z) - e_k*||^2 por padrão.
    ema: se o codebook é atualizado por EMA (sem gradiente).
    soft: se o estimador diferencia através de p (precisa do grafo de d2).
    """

    fn: EstimatorFn
    codebook_loss: bool = False
    ema: bool = False
    soft: bool = False


ESTIMATORS: dict[str, Estimator] = {}


def register_estimator(name: str, *, codebook_loss=False, ema=False, soft=False):
    def deco(fn: EstimatorFn) -> EstimatorFn:
        if name in ESTIMATORS:
            raise ValueError(f"estimador já registrado: {name}")
        ESTIMATORS[name] = Estimator(fn, codebook_loss=codebook_loss, ema=ema, soft=soft)
        return fn

    return deco


def get_estimator(name: str) -> Estimator:
    if name not in ESTIMATORS:
        raise KeyError(f"estimador desconhecido: {name!r}. Registrados: {sorted(ESTIMATORS)}")
    return ESTIMATORS[name]


def sqdist(z: torch.Tensor, e: torch.Tensor) -> torch.Tensor:
    """||z_i - e_j||^2 para todo par, shape (N, K)."""
    return z.pow(2).sum(1, keepdim=True) - 2 * z @ e.t() + e.pow(2).sum(1)[None, :]


def _pick(p: torch.Tensor, k: torch.Tensor) -> torch.Tensor:
    return p.gather(1, k[:, None])  # (N, 1)


@register_estimator("ste", codebook_loss=True)
def ste(z, e, k, d2, tau):
    """z_q = z + sg(e_k* - z): o gradiente chega em z inalterado; o codebook só sai da perda."""
    return e[k].detach() + (z - z.detach())


@register_estimator("ste_ema", ema=True)
def ste_ema(z, e, k, d2, tau):
    """Como ste; o codebook é atualizado por EMA no VectorQuantizer, sem codebook loss."""
    return e[k].detach() + (z - z.detach())


@register_estimator("gate_norm", soft=True)
def gate_norm(z, e, k, d2, tau):
    """z_q = (p_k* / sg(p_k*)) e_k*: gate-value normalizado; o gradiente do router chega em z."""
    p = torch.softmax(-d2 / tau, dim=1)
    pk = _pick(p, k)
    return (pk / pk.detach()) * e[k]


@register_estimator("gate_cb")
def gate_cb(z, e, k, d2, tau):
    """z_q = z - sg(z) + (ph_k*/sg(ph_k*)) e_k*, com ph calculado a partir de sg(z).

    STE no encoder (o gradiente em z é exatamente g); o gate-value age só no codebook.
    """
    zd = z.detach()
    ph = torch.softmax(-sqdist(zd, e) / tau, dim=1)
    phk = _pick(ph, k)
    return (z - zd) + (phk / phk.detach()) * e[k]


@register_estimator("dense_st", soft=True)
def dense_st(z, e, k, d2, tau):
    """z_q = sum_j p_j e_j + sg(e_k* - sum_j p_j e_j): forward e_k*, gradiente da mistura densa."""
    p = torch.softmax(-d2 / tau, dim=1)
    ebar = p @ e
    return e[k].detach() + (ebar - ebar.detach())


@dataclass
class VQOutput:
    z_q: torch.Tensor
    loss: torch.Tensor
    indices: torch.Tensor
    dmin: torch.Tensor  # ||z - e_k*||^2 por token


class VectorQuantizer(nn.Module):
    def __init__(
        self,
        num_codes: int,
        dim: int,
        estimator: str = "ste",
        beta: float = 0.25,
        tau: float = 1.0,
        tau_scale: str | None = None,
        codebook_loss: bool | None = None,
        ema_decay: float = 0.99,
        ema_eps: float = 1e-5,
        seed: int = 0,
    ):
        super().__init__()
        if tau_scale not in (None, "median"):
            raise ValueError(f"tau_scale inválido: {tau_scale!r} (use null ou 'median')")
        self.est = get_estimator(estimator)
        self.estimator = estimator
        self.K, self.d = num_codes, dim
        self.beta, self.tau, self.tau_scale = beta, tau, tau_scale
        self.use_codebook_loss = self.est.codebook_loss if codebook_loss is None else codebook_loss
        self.ema_decay, self.ema_eps = ema_decay, ema_eps

        # Init uniforme U(-1/K, 1/K) com gerador próprio: idêntica entre estimadores,
        # independente do RNG global (e portanto do que o encoder/decoder consumiram).
        g = torch.Generator().manual_seed(seed)
        init = (torch.rand(num_codes, dim, generator=g) * 2 - 1) / num_codes
        self.codebook = nn.Parameter(init, requires_grad=not self.est.ema)

        # tau efetivo: congelado depois do primeiro batch quando tau_scale == "median".
        self.register_buffer("tau_eff", torch.tensor(float(tau)))
        self.register_buffer("tau_frozen", torch.tensor(tau_scale is None))
        if self.est.ema:
            # cluster_size = 1 e embed_avg = codebook: o primeiro update reproduz o init, e um
            # código morto não é jogado para longe pela divisão por um tamanho ~0.
            self.register_buffer("cluster_size", torch.ones(num_codes))
            self.register_buffer("embed_avg", init.clone())

    def _tau(self, dmin: torch.Tensor) -> torch.Tensor:
        if not bool(self.tau_frozen):
            self.tau_eff.fill_(float(self.tau * dmin.detach().median()))
            self.tau_frozen.fill_(True)
        return self.tau_eff.to(dmin.dtype)

    @torch.no_grad()
    def _ema_update(self, z: torch.Tensor, k: torch.Tensor) -> None:
        onehot = F.one_hot(k, self.K).to(z.dtype)
        self.cluster_size.mul_(self.ema_decay).add_(onehot.sum(0), alpha=1 - self.ema_decay)
        total = self.cluster_size.sum()
        smoothed = (self.cluster_size + self.ema_eps) / (total + self.K * self.ema_eps) * total
        self.embed_avg.mul_(self.ema_decay).add_(onehot.t() @ z, alpha=1 - self.ema_decay)
        self.codebook.data.copy_(self.embed_avg / smoothed[:, None])

    def forward(self, z: torch.Tensor) -> VQOutput:
        e = self.codebook
        with torch.set_grad_enabled(torch.is_grad_enabled() and self.est.soft):
            d2 = sqdist(z, e)
        k = d2.argmin(1)
        dmin = d2.detach().gather(1, k[:, None]).squeeze(1)
        tau = self._tau(dmin)

        z_q = self.est.fn(z, e, k, d2, tau)

        e_k = e[k]
        loss = self.beta * F.mse_loss(z, e_k.detach())
        if self.use_codebook_loss:
            loss = loss + F.mse_loss(z.detach(), e_k)
        if self.est.ema and self.training:
            self._ema_update(z.detach(), k)
        return VQOutput(z_q=z_q, loss=loss, indices=k, dmin=dmin)
