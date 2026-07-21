import Mathlib
set_option linter.style.header false

/-!
# [Descent-Cap] — capped Kenian correction preserves descent

The optimizer takes a base step `Δ` and adds a third-order correction `c`, capped in the
preconditioner norm: `‖c‖ ≤ ρ‖Δ‖`, `ρ < 1`.  Work in the preconditioner inner product
(mathematically: any real inner-product space `E` — for a diagonal preconditioner `P` this
is `⟪x,y⟫ = xᵀPy`, which is an inner product exactly when `P` is positive definite).

Writing the momentum as `m = -Δ` in this geometry (the base step is the preconditioned
negative momentum), the first-order model change along the corrected step `Δ + c` is
`⟪m, Δ + c⟫ = -⟪Δ, Δ + c⟫`, and we show `⟪Δ, Δ + c⟫ ≥ (1-ρ)‖Δ‖² > 0`:
**no capped correction can turn a descent direction into an ascent direction.**
-/

open RealInnerProductSpace

namespace Kenian

variable {E : Type*} [NormedAddCommGroup E] [InnerProductSpace ℝ E]

/-- Core inequality: the corrected step retains at least a `(1-ρ)` fraction of the
base step's alignment with itself. -/
theorem inner_self_add_capped_ge {ρ : ℝ} (Δ c : E) (hcap : ‖c‖ ≤ ρ * ‖Δ‖) :
    (1 - ρ) * ‖Δ‖ ^ 2 ≤ ⟪Δ, Δ + c⟫ := by
  have h1 : ⟪Δ, Δ + c⟫ = ‖Δ‖ ^ 2 + ⟪Δ, c⟫ := by
    rw [inner_add_right, real_inner_self_eq_norm_sq]
  have h2 : |⟪Δ, c⟫| ≤ ‖Δ‖ * ‖c‖ := abs_real_inner_le_norm Δ c
  have h3 : ‖Δ‖ * ‖c‖ ≤ ρ * ‖Δ‖ ^ 2 := by nlinarith [norm_nonneg Δ, norm_nonneg c]
  have h4 : -(ρ * ‖Δ‖ ^ 2) ≤ ⟪Δ, c⟫ := by
    have := (abs_le.mp h2).1
    linarith
  linarith [h1, h4]

/-- Strict descent: if `Δ ≠ 0` and `ρ < 1`, the corrected step has strictly positive
alignment with the base step. -/
theorem inner_self_add_capped_pos {ρ : ℝ} (hρ : ρ < 1) {Δ : E} (hΔ : Δ ≠ 0) (c : E)
    (hcap : ‖c‖ ≤ ρ * ‖Δ‖) : 0 < ⟪Δ, Δ + c⟫ := by
  have hn : 0 < ‖Δ‖ := norm_pos_iff.mpr hΔ
  have : 0 < (1 - ρ) * ‖Δ‖ ^ 2 := by positivity
  linarith [inner_self_add_capped_ge Δ c hcap]

/-- Optimizer form: with momentum `m` and base step `Δ = -m` (in the preconditioner
geometry), the model slope along the corrected step is at most `-(1-ρ)‖m‖²`. -/
theorem model_decrease_of_capped_correction {ρ : ℝ} (m c : E)
    (hcap : ‖c‖ ≤ ρ * ‖m‖) : ⟪m, -m + c⟫ ≤ -((1 - ρ) * ‖m‖ ^ 2) := by
  have h := inner_self_add_capped_ge m c hcap
  have hswap : ⟪m, -m + c⟫ = -⟪m, m + -c⟫ := by
    rw [inner_add_right, inner_add_right, inner_neg_right, inner_neg_right]
    ring
  have hcap' : ‖-c‖ ≤ ρ * ‖m‖ := by rwa [norm_neg]
  have h' := inner_self_add_capped_ge m (-c) hcap'
  linarith [hswap ▸ neg_le_neg h']

end Kenian
