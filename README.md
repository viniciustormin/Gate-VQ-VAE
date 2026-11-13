# Gate-VQ-VAE

Sparse Gating Networks for Vector Quantized Variational Autoencoders.

Compara estimadores de gradiente para a camada de quantização de um VQ-VAE. O VQ é visto como um
MoE degenerado (codewords = experts constantes, vizinho mais próximo = router top-1), e o
gradiente do router de MoE (gate-value) é testado no lugar do straight-through (STE).

## Estimadores (`src/vqrouter/quantizers.py`)

Todos têm forward exatamente `e_k*`; o que muda é o gradiente. Notação: `p = softmax(-||z-e_j||²/τ)`,
`sg` = stop-gradient, `ē = Σ_j p_j e_j`.

| `estimator` | `z_q` | gradiente em `z` (g = dL/dz_q) |
|---|---|---|
| `ste` | `z + sg(e_k* - z)`; codebook loss + commitment | `g` |
| `ste_ema` | como `ste`, codebook por EMA (0,99), sem codebook loss | `g` |
| `gate_norm` | `(p_k*/sg(p_k*)) e_k*` | `(2/τ)(g·e_k*)(e_k* - ē)` |
| `gate_cb` | `z - sg(z) + (ph_k*/sg(ph_k*)) e_k*`, `ph` de `sg(z)` | `g` |
| `dense_st` | `Σ_j p_j e_j + sg(e_k* - Σ_j p_j e_j)` | `(2/τ) Σ_j p_j (g·e_j)(e_j - ē)` |

Commitment (β = 0,25) em todos. Para o forward ser bit a bit `e_k*`, o código escreve
`e_k.detach() + (z - z.detach())` no lugar de `z + sg(e_k - z)` (mesmo gradiente, forward exato).
`tau_scale: median` multiplica τ pela mediana de `||z - e_k*||²` do primeiro batch e congela.
Para acrescentar um estimador, registre uma função com `@register_estimator`.

## Uso local

```bash
uv sync
uv run pytest -q
uv run python scripts/download_data.py --datasets mnist
uv run python -m vqrouter.train --config configs/base/mnist.yaml \
    --override estimator=gate_cb train.max_steps=500 train.eval_every=250 exp=smoke
```

Cada execução grava em `runs/<exp>/<estimator>_K<K>_tau<tau>_s<seed>/` (`config.yaml`,
`metrics.csv`, `commit.txt`, `ckpt.pt`, `done.json`). Execução concluída é pulada; `--resume`
retoma do checkpoint e, com `train.max_steps` maior, estende o treino.
