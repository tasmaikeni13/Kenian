"""Numerical checks for the identities used by the Kenian update.

The tests exercise small exact cases before the corresponding Lean statements.
"""

import torch

torch.manual_seed(0)
torch.set_default_dtype(torch.float64)


def third_slice(f, theta, a, b):
    """Exact 𝒦[a,b,·] = ∇( (∇( (∇f)ᵀ a ))ᵀ b ) via double reverse-mode."""
    theta = theta.detach().requires_grad_(True)
    (g,) = torch.autograd.grad(f(theta), theta, create_graph=True)
    (Ha,) = torch.autograd.grad(g @ a, theta, create_graph=True)
    (Kab,) = torch.autograd.grad(Ha @ b, theta)
    return Kab


def hvp(f, theta, a):
    theta = theta.detach().requires_grad_(True)
    (g,) = torch.autograd.grad(f(theta), theta, create_graph=True)
    (Ha,) = torch.autograd.grad(g @ a, theta)
    return Ha


def test_slice_exactness_polynomial():
    """On an explicit cubic, the autodiff slice equals the symbolic 𝒦[a,b,·]."""
    n = 7
    g0 = torch.randn(n)
    H = torch.randn(n, n)
    H = H + H.T
    T = torch.randn(n, n, n)
    T = (
        T
        + T.permute(0, 2, 1)
        + T.permute(1, 0, 2)
        + T.permute(1, 2, 0)
        + T.permute(2, 0, 1)
        + T.permute(2, 1, 0)
    ) / 6

    def f(th):
        return (
            g0 @ th
            + 0.5 * th @ H @ th
            + (1.0 / 6.0) * torch.einsum("ijk,i,j,k", T, th, th, th)
        )

    theta = torch.randn(n)
    a, b = torch.randn(n), torch.randn(n)
    K_ad = third_slice(f, theta, a, b)
    K_sym = torch.einsum("ijk,j,k->i", T, a, b)
    assert torch.allclose(K_ad, K_sym, atol=1e-9), (K_ad - K_sym).abs().max()
    print("[ok] slice exactness: autodiff 𝒦[a,b,·] == symbolic contraction (cubic)")


def test_slice_on_smooth_network():
    """On a GELU MLP, autodiff slice matches central finite differences of HVPs:
    𝒦[a,b,·] = lim (H(θ+εa) b − H(θ−εa) b)/(2ε)."""
    torch.manual_seed(1)
    W1, b1, W2 = torch.randn(11, 5), torch.randn(11), torch.randn(3, 11)
    x, y = torch.randn(4, 5), torch.tensor([0, 2, 1, 0])

    def pack(f_):
        def f(th):
            w1 = th[:55].reshape(11, 5)
            bb = th[55:66]
            w2 = th[66:].reshape(3, 11)
            z = torch.nn.functional.gelu(x @ w1.T + bb) @ w2.T
            return torch.nn.functional.cross_entropy(z, y)

        return f

    f = pack(None)
    theta = torch.cat([W1.flatten(), b1, W2.flatten()])
    n = theta.numel()
    a, b = torch.randn(n), torch.randn(n)
    K_ad = third_slice(f, theta, a, b)
    eps = 1e-5
    K_fd = (hvp(f, theta + eps * a, b) - hvp(f, theta - eps * a, b)) / (2 * eps)
    err = (K_ad - K_fd).norm() / K_ad.norm()
    assert err < 1e-5, err
    # symmetry in the two contracted arguments
    K_ba = third_slice(f, theta, b, a)
    assert torch.allclose(K_ad, K_ba, atol=1e-8)
    print(
        f"[ok] slice on GELU net matches FD of HVPs (rel err {err:.2e}); symmetric in (a,b)"
    )


def test_cumulant_identity():
    """t ↦ CE(z+tu) has φ''(0)=Var_p(u), φ'''(0)=E_p(u−μ)³  (softmax-CE)."""
    C = 6
    z = torch.randn(C)
    u = torch.randn(C)
    y = 3

    def phi(t):
        return torch.nn.functional.cross_entropy(
            (z + t * u).unsqueeze(0), torch.tensor([y])
        )

    t = torch.zeros((), requires_grad=True)
    val = phi(t)
    d1 = torch.autograd.grad(val, t, create_graph=True)[0]
    d2 = torch.autograd.grad(d1, t, create_graph=True)[0]
    d3 = torch.autograd.grad(d2, t)[0]
    p = torch.softmax(z, 0)
    mu = (p * u).sum()
    var = (p * (u - mu) ** 2).sum()
    m3 = (p * (u - mu) ** 3).sum()
    assert torch.allclose(d2, var, atol=1e-10)
    assert torch.allclose(d3, m3, atol=1e-10)
    # skewness bound |m3| <= osc(u) * var
    osc = u.max() - u.min()
    assert m3.abs() <= osc * var + 1e-12
    print("[ok] cumulant identity: φ''=Var_p(u), φ'''=E_p(u−μ)³; |φ'''| ≤ osc·Var")


def test_chebyshev_residual_orders():
    """Cubic model: Newton residual == ½𝒦[Δ,Δ]; corrected residual is higher order in scale s."""
    torch.manual_seed(2)
    n = 6
    H = torch.randn(n, n)
    H = H @ H.T + n * torch.eye(n)  # SPD
    T = torch.randn(n, n, n)
    T = (
        T
        + T.permute(0, 2, 1)
        + T.permute(1, 0, 2)
        + T.permute(1, 2, 0)
        + T.permute(2, 0, 1)
        + T.permute(2, 1, 0)
    ) / 6
    Hinv = torch.linalg.inv(H)

    def gfield(g0, th):
        return g0 + H @ th + 0.5 * torch.einsum("ijk,j,k->i", T, th, th)

    ratios = []
    for s in [1e-1, 1e-2, 1e-3]:
        g0 = s * torch.randn(n)
        dN = -Hinv @ g0
        rN = gfield(g0, dN)
        assert torch.allclose(
            rN, 0.5 * torch.einsum("ijk,j,k->i", T, dN, dN), atol=1e-12
        )
        dC = dN - 0.5 * Hinv @ torch.einsum("ijk,j,k->i", T, dN, dN)
        rC = gfield(g0, dC)
        ratios.append((rN.norm() / g0.norm(), rC.norm() / g0.norm()))
    # Newton residual ~ s (relative), corrected ~ s² (relative): check decay rates
    (rn1, rc1), (rn2, rc2), (rn3, rc3) = ratios
    assert 5 < rn1 / rn2 < 20 and 5 < rn2 / rn3 < 20  # one extra order
    assert 50 < rc1 / rc2 < 200 and 50 < rc2 / rc3 < 200  # two extra orders
    print(
        "[ok] Chebyshev: Newton res = ½𝒦[Δ,Δ] exactly; correction gains one order "
        f"(decay {rn1 / rn2:.1f}x vs {rc1 / rc2:.1f}x per decade)"
    )


def test_lag_cancellation():
    """EMA tracker of a drifting signal: steady lag −βδ/(1−β); feedforward kills it."""
    beta, delta, Tsteps = 0.999, 1e-3, 20000
    h = 0.0
    v = 0.0
    vff = 0.0
    for t in range(Tsteps):
        h += delta
        v = beta * v + (1 - beta) * h
        vff = beta * (vff + delta) + (1 - beta) * h
    lag_pred = -beta * delta / (1 - beta)
    assert abs((v - h) - lag_pred) < 1e-6, (v - h, lag_pred)
    assert abs(vff - h) < 1e-9
    print(
        f"[ok] lag: EMA err {v - h:.4f} == −βδ/(1−β) {lag_pred:.4f}; feedforward err {vff - h:.1e}"
    )


def test_probe_unbiasedness():
    """E_z[z ⊙ 𝒦[Δ,z,·]] = diag 𝒦[Δ,·,·] for Rademacher z (exact over full enumeration)."""
    torch.manual_seed(3)
    n = 5
    T = torch.randn(n, n, n)
    T = (
        T
        + T.permute(0, 2, 1)
        + T.permute(1, 0, 2)
        + T.permute(1, 2, 0)
        + T.permute(2, 0, 1)
        + T.permute(2, 1, 0)
    ) / 6
    D = torch.randn(n)
    target = torch.einsum("iik->ik", T) @ D  # diag(𝒦[Δ,·,·])_i = Σ_k T_iik Δ_k
    acc = torch.zeros(n)
    for mask in range(2**n):
        z = torch.tensor([1.0 if (mask >> i) & 1 else -1.0 for i in range(n)])
        acc += z * torch.einsum("ijk,j,k->i", T, D, z)
    acc /= 2**n
    assert torch.allclose(acc, target, atol=1e-10)
    print(
        "[ok] probe unbiasedness: E_z[z⊙𝒦[Δ,z,·]] == diag 𝒦[Δ,·,·] (full enumeration)"
    )


if __name__ == "__main__":
    test_slice_exactness_polynomial()
    test_slice_on_smooth_network()
    test_cumulant_identity()
    test_chebyshev_residual_orders()
    test_lag_cancellation()
    test_probe_unbiasedness()
    print("\nAll numerical witnesses hold.")
