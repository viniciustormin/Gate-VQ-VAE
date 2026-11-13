"""Métricas de reconstrução e de uso do codebook, no conjunto de teste inteiro."""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F


def psnr(mse: float, peak: float = 1.0) -> float:
    return float("inf") if mse <= 0 else 10 * math.log10(peak**2 / mse)


def codes_used_pct(counts: torch.Tensor) -> float:
    return 100.0 * float((counts > 0).sum()) / counts.numel()


def perplexity(counts: torch.Tensor) -> float:
    """exp(H) do histograma de uso dos códigos; vale K se uso uniforme e 1 se colapsa."""
    counts = counts.cpu()  # MPS não tem float64
    p = counts.double() / counts.sum()
    p = p[p > 0]
    return float(torch.exp(-(p * p.log()).sum()))


@torch.no_grad()
def evaluate(model, data, num_codes: int, device, batch_size: int = 1000) -> dict[str, float]:
    """MSE e PSNR de reconstrução (dados em [0, 1]), % de códigos usados, perplexidade e
    erro de quantização médio ||z - e_k*||^2, sobre todo o conjunto de teste."""
    was_training = model.training
    model.eval()
    sq_err, n_elem, qerr, n_tok = 0.0, 0, 0.0, 0
    counts = torch.zeros(num_codes, dtype=torch.long, device=device)
    for i in range(0, len(data.test), batch_size):
        x = data.float(data.test[i : i + batch_size].to(device))
        recon, out = model(x)
        recon = recon.clamp(0, 1) if data.scale != 1.0 else recon
        sq_err += float(F.mse_loss(recon, x, reduction="sum"))
        n_elem += x.numel()
        qerr += float(out.dmin.sum())
        n_tok += out.dmin.numel()
        counts += torch.bincount(out.indices, minlength=num_codes)
    model.train(was_training)
    mse = sq_err / n_elem
    return {
        "test_mse": mse,
        "test_psnr": psnr(mse),
        "codes_used_pct": codes_used_pct(counts),
        "perplexity": perplexity(counts),
        "quant_err": qerr / n_tok,
    }
