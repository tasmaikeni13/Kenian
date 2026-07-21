import Mathlib
set_option linter.style.header false

/-!
# [Lag-Cancellation] — EMA curvature trackers lag a drifting signal; the Kenian
feedforward term cancels the lag

Adaptive optimizers track a curvature-like signal `h` with an EMA
`v (t+1) = β v t + (1-β) h (t+1)`.  When the tracked signal drifts by `δ` per step
(along the trajectory this drift *is* the Kenian slice, `δ_t = diag 𝒦[Δ_t,·,·]`), the
tracker settles to a steady-state error `-βδ/(1-β)` — e.g. `999·δ` at `β = 0.999`.

Adding a feedforward term `d̂` (the streamed Kenian drift estimate) changes the error
recurrence so that the steady state becomes `β(d̂-δ)/(1-β)`: **the lag is proportional
to the drift-estimate error rather than to the drift itself**, and vanishes entirely
when the estimate is exact, with geometric convergence at rate `β`.
-/

namespace Kenian

/-- Steady-state error of the drift-corrected tracker (pure EMA is `dh = 0`). -/
noncomputable def emaFixedPoint (β δ dh : ℝ) : ℝ := β * (dh - δ) / (1 - β)

/-- Error recurrence: for a signal drifting by `δ` per step and tracker
`v (t+1) = β (v t + d̂) + (1-β) h (t+1)`, the error `e = v - h` satisfies
`e (t+1) = β (e t + d̂ - δ)`. -/
theorem ema_error_recurrence {β δ dh : ℝ} {h v : ℕ → ℝ}
    (hdrift : ∀ t, h (t + 1) = h t + δ)
    (hema : ∀ t, v (t + 1) = β * (v t + dh) + (1 - β) * h (t + 1)) (t : ℕ) :
    v (t + 1) - h (t + 1) = β * ((v t - h t) + (dh - δ)) := by
  rw [hema t, hdrift t]; ring

/-- Closed form for the tracking error: geometric approach to the fixed point. -/
theorem ema_error_closed_form {β δ dh : ℝ} (hβ : β ≠ 1) {h v : ℕ → ℝ}
    (hdrift : ∀ t, h (t + 1) = h t + δ)
    (hema : ∀ t, v (t + 1) = β * (v t + dh) + (1 - β) * h (t + 1)) (t : ℕ) :
    v t - h t = β ^ t * ((v 0 - h 0) - emaFixedPoint β δ dh) + emaFixedPoint β δ dh := by
  induction t with
  | zero => simp
  | succ t ih =>
    have hfix : β * (emaFixedPoint β δ dh + (dh - δ)) = emaFixedPoint β δ dh := by
      unfold emaFixedPoint
      field_simp
      ring
    rw [ema_error_recurrence hdrift hema t, ih]
    calc β * ((β ^ t * ((v 0 - h 0) - emaFixedPoint β δ dh) + emaFixedPoint β δ dh) + (dh - δ))
        = β ^ (t + 1) * ((v 0 - h 0) - emaFixedPoint β δ dh)
            + β * (emaFixedPoint β δ dh + (dh - δ)) := by ring
      _ = _ := by rw [hfix]

/-- **Pure EMA lags**: with no feedforward (`dh = 0`), the tracking error converges to
`-βδ/(1-β)`.  For `β = 0.999` this is `-999 δ`. -/
theorem ema_lag {β δ : ℝ} (hβ : |β| < 1) {h v : ℕ → ℝ}
    (hdrift : ∀ t, h (t + 1) = h t + δ)
    (hema : ∀ t, v (t + 1) = β * (v t + 0) + (1 - β) * h (t + 1)) :
    Filter.Tendsto (fun t => v t - h t) Filter.atTop (nhds (-(β * δ) / (1 - β))) := by
  have hβ1 : β ≠ 1 := by
    intro hh; rw [hh] at hβ; simp at hβ
  have hcf := ema_error_closed_form hβ1 hdrift hema
  have hfp : emaFixedPoint β δ 0 = -(β * δ) / (1 - β) := by
    unfold emaFixedPoint; ring_nf
  have hgeo : Filter.Tendsto (fun t : ℕ => β ^ t * ((v 0 - h 0) - emaFixedPoint β δ 0))
      Filter.atTop (nhds 0) := by
    have := tendsto_pow_atTop_nhds_zero_of_abs_lt_one hβ
    simpa using this.mul_const ((v 0 - h 0) - emaFixedPoint β δ 0)
  have : Filter.Tendsto (fun t : ℕ => β ^ t * ((v 0 - h 0) - emaFixedPoint β δ 0)
      + emaFixedPoint β δ 0) Filter.atTop (nhds (0 + emaFixedPoint β δ 0)) :=
    hgeo.add_const _
  simp only [zero_add, hfp] at this
  refine this.congr fun t => ?_
  rw [hcf t, hfp]

/-- **Exact feedforward cancels the lag**: with `d̂ = δ`, the error contracts
geometrically to zero — no steady-state lag. -/
theorem feedforward_cancels_lag {β δ : ℝ} (hβ1 : β ≠ 1) {h v : ℕ → ℝ}
    (hdrift : ∀ t, h (t + 1) = h t + δ)
    (hema : ∀ t, v (t + 1) = β * (v t + δ) + (1 - β) * h (t + 1)) (t : ℕ) :
    v t - h t = β ^ t * (v 0 - h 0) := by
  have hcf := ema_error_closed_form hβ1 hdrift hema t
  have hfp : emaFixedPoint β δ δ = 0 := by unfold emaFixedPoint; simp
  rw [hcf, hfp]; ring

/-- **Robustness**: an inexact drift estimate `d̂` still wins whenever the estimate error
is smaller than the drift: `|d̂ - δ| ≤ |δ|` implies the feedforward steady state is no
worse than the pure-EMA steady state. -/
theorem feedforward_no_worse {β δ dh : ℝ} (hest : |dh - δ| ≤ |δ|) :
    |emaFixedPoint β δ dh| ≤ |emaFixedPoint β δ 0| := by
  unfold emaFixedPoint
  have hnum : |β * (dh - δ)| ≤ |β * (0 - δ)| := by
    rw [abs_mul, abs_mul]
    have h0 : |(0 : ℝ) - δ| = |δ| := by simp
    rw [h0]
    exact mul_le_mul_of_nonneg_left hest (abs_nonneg β)
  rw [abs_div, abs_div, div_eq_mul_inv, div_eq_mul_inv]
  exact mul_le_mul_of_nonneg_right hnum (inv_nonneg.mpr (abs_nonneg _))

end Kenian
