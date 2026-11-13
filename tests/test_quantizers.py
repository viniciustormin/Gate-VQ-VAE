"""Testes dos estimadores: forward exato, gradientes em forma fechada e fluxo ao codebook.

Tudo em float64. Se uma fórmula não bater com o autograd, o teste FALHA e a derivação é que deve
ser revista: a tolerância não se afrouxa para passar.
"""

import pytest
import torch

from vqrouter.quantizers import ESTIMATORS, VectorQuantizer, get_estimator

ALL = ["ste", "ste_ema", "gate_norm", "gate_cb", "dense_st"]
N, K, D = 32, 16, 5
TAU = 0.7


def make(estimator, seed=0, **kw):
    torch.manual_seed(seed)
    vq = VectorQuantizer(K, D, estimator=estimator, tau=TAU, seed=seed, **kw).double()
    # codebook com escala comparável a z, para que p não degenere em one-hot
    with torch.no_grad():
        vq.codebook.copy_(torch.randn(K, D, dtype=torch.float64))
        if vq.est.ema:
            vq.embed_avg.copy_(vq.codebook)
    z = torch.randn(N, D, dtype=torch.float64, requires_grad=True)
    return vq, z


def test_registry():
    assert set(ALL) <= set(ESTIMATORS)
    with pytest.raises(KeyError):
        get_estimator("nao_existe")


@pytest.mark.parametrize("name", ALL)
def test_a_forward_is_exactly_nearest_code(name):
    vq, z = make(name)
    book = vq.codebook.detach().clone()  # o EMA atualiza o codebook DEPOIS do forward
    out = vq(z)
    assert torch.equal(out.z_q, book[out.indices])
    # k* é de fato o vizinho mais próximo
    d2 = torch.cdist(z.detach(), book).pow(2)
    assert torch.equal(out.indices, d2.argmin(1))


def _grad_z(name, g):
    vq, z = make(name)
    out = vq(z)
    (out.z_q * g).sum().backward()
    return vq, z.detach(), z.grad


def _probs(vq, z):
    e = vq.codebook.detach()
    d2 = torch.cdist(z, e).pow(2)
    return torch.softmax(-d2 / TAU, dim=1), e, d2.argmin(1)


@pytest.mark.parametrize("name", ["ste", "gate_cb"])
def test_b_grad_z_is_exactly_g(name):
    g = torch.randn(N, D, dtype=torch.float64)
    _, _, grad = _grad_z(name, g)
    assert torch.equal(grad, g)


def test_b_gate_norm_grad_matches_closed_form():
    g = torch.randn(N, D, dtype=torch.float64)
    vq, z, grad = _grad_z("gate_norm", g)
    p, e, k = _probs(vq, z)
    ebar = p @ e
    ek = e[k]
    expected = (2 / TAU) * (g * ek).sum(1, keepdim=True) * (ek - ebar)
    torch.testing.assert_close(grad, expected, atol=1e-6, rtol=1e-6)


def test_b_dense_st_grad_matches_closed_form():
    g = torch.randn(N, D, dtype=torch.float64)
    vq, z, grad = _grad_z("dense_st", g)
    p, e, _ = _probs(vq, z)
    ebar = p @ e
    ge = g @ e.t()  # (N, K): g . e_j
    # (2/tau) sum_j p_j (g.e_j)(e_j - ebar)
    expected = (2 / TAU) * ((p * ge) @ e - (p * ge).sum(1, keepdim=True) * ebar)
    torch.testing.assert_close(grad, expected, atol=1e-6, rtol=1e-6)


def _recon_grad_codebook(name, **kw):
    vq, z = make(name, **kw)
    g = torch.randn(N, D, dtype=torch.float64)
    out = vq(z)
    (out.z_q * g).sum().backward()  # só a "reconstrução": sem a perda do VQ
    return vq.codebook.grad, z.grad, g


def test_c_gate_cb_encoder_gets_g_and_codebook_gets_gradient():
    cb_grad, z_grad, g = _recon_grad_codebook("gate_cb")
    assert torch.equal(z_grad, g)
    assert cb_grad is not None and cb_grad.abs().max() > 0


def test_d_ste_codebook_gets_zero_gradient_from_reconstruction():
    cb_grad, z_grad, g = _recon_grad_codebook("ste")
    assert cb_grad is None or torch.count_nonzero(cb_grad) == 0
    assert torch.equal(z_grad, g)


def test_loss_composition():
    """Commitment em todos; codebook loss só no ste por padrão; opção do gate_cb liga/desliga."""
    for name in ALL:
        vq, z = make(name)
        book = vq.codebook.detach().clone()  # antes do update do EMA
        out = vq(z)
        commit = 0.25 * torch.mean((z - book[out.indices]) ** 2)
        cb = torch.mean((z.detach() - book[out.indices]) ** 2)
        want = commit + (cb if name == "ste" else 0)
        torch.testing.assert_close(out.loss, want)
    vq, z = make("gate_cb", codebook_loss=True)
    out = vq(z)
    commit = 0.25 * torch.mean((z - vq.codebook[out.indices].detach()) ** 2)
    cb = torch.mean((z.detach() - vq.codebook[out.indices]) ** 2)
    torch.testing.assert_close(out.loss, commit + cb)


def test_ste_ema_has_no_trainable_codebook_and_updates_in_training():
    vq, z = make("ste_ema")
    assert not vq.codebook.requires_grad
    before = vq.codebook.clone()
    vq(z.detach())
    assert not torch.equal(before, vq.codebook)
    vq.eval()
    before = vq.codebook.clone()
    vq(z.detach())
    assert torch.equal(before, vq.codebook)


def test_init_identical_across_estimators():
    books = [VectorQuantizer(64, 8, estimator=n, seed=3).codebook.detach() for n in ALL]
    for b in books[1:]:
        assert torch.equal(books[0], b)
    assert books[0].abs().max() <= 1 / 64


def test_tau_median_freezes_on_first_batch():
    vq = VectorQuantizer(K, D, estimator="gate_norm", tau=2.0, tau_scale="median", seed=0)
    z1 = torch.randn(64, D)
    out = vq(z1)
    expected = 2.0 * out.dmin.median()
    assert bool(vq.tau_frozen)
    torch.testing.assert_close(vq.tau_eff, expected)
    frozen = vq.tau_eff.clone()
    vq(torch.randn(64, D) * 10)
    assert torch.equal(vq.tau_eff, frozen)
    # volta do state_dict
    vq2 = VectorQuantizer(K, D, estimator="gate_norm", tau=2.0, tau_scale="median", seed=0)
    vq2.load_state_dict(vq.state_dict())
    assert bool(vq2.tau_frozen) and torch.equal(vq2.tau_eff, frozen)
