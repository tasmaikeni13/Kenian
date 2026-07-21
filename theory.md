# Kenian: a trajectory-sliced third-order update

## Motivation

Let `L(θ)` be a smooth loss, with gradient `g`, Hessian `H`, and third
derivative `𝒦 = ∇³L`. A full third-order tensor has `n³` entries, so it is not
practical to store or approximate directly for a large neural network.

Kenian only evaluates the contraction needed by its update:

```text
κ(Δ) = 𝒦[Δ, Δ, ·]
```

Here `Δ` is the optimizer's previous AdamW base update. This is an exact
autodiff computation for the selected direction, rather than an entrywise,
low-rank, or diagonal approximation of `𝒦`.

## Update

The base update is AdamW:

```text
m_t = β₁m_(t-1) + (1 - β₁)g_t
v_t = β₂v_(t-1) + (1 - β₂)g_t²
Δ₀ = -η m̂_t / (√v̂_t + ε)
```

Every `probe_interval` steps, Kenian computes `κ(Δ₀ / ‖Δ₀‖₂)` with two
reverse-mode contractions and updates an EMA of the result. The applied step
is

```text
Δ = Δ₀ - α κ̂ / P,
```

where `P = √v̂_t + ε` is the AdamW denominator. The global scale `α` is
chosen so that the correction obeys

```text
‖Δ - Δ₀‖_P ≤ correction_cap · ‖Δ₀‖_P.
```

The cap makes the third-order term a refinement, not a replacement for the
base optimizer. The implementation stores `Δ₀`, rather than the corrected
step, as the next probe direction; this avoids feeding a correction back into
the slice estimate.

## Interpretation and limits

For an exact Newton step in a cubic model, the leading residual is
`½𝒦[Δ, Δ, ·]`. This is the source of the Chebyshev-style correction. With an
AdamW diagonal preconditioner and stochastic gradients, that local argument
does not imply a convergence guarantee or a generalization benefit.

The softmax cross-entropy output-space third derivative is the third central
cumulant of the predictive distribution. This gives the correction a useful
interpretation as a directional measure of changing curvature, but it does
not establish when a noisy estimate will help training.

The Lean files in `lean/Kenianproofs/` check the underlying algebraic
identities and bounds. They do not prove that Kenian outperforms AdamW.
