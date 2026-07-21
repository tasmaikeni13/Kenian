import Mathlib
set_option linter.style.header false
set_option linter.style.longLine false

/-!
# Cubic-model Taylor remainder bound.

Along the step direction, `φ(t) = L(θ + tΔ)`.  The Kenian optimizer's local model is the
cubic Taylor model of `φ`; this file quantifies its error: if the fourth derivative is
bounded by `M` on the step segment, the cubic model misses the true loss by at most
`M/24 · ‖Δ‖⁴` (here in normalized arclength `t ∈ [0,1]`, so the `‖Δ‖⁴` is absorbed).
This is why the *cubic* model — hence one Kenian slice — is the right local object: its
error is one order beyond the quadratic model's `‖Δ‖³`-error, which `newton_residual`
shows is exactly the Kenian slice.
-/

namespace Kenian

open Set

/-- **Cubic-model error**: with `φ` four-times differentiable on the step segment and
`|φ⁗| ≤ M` there, `|φ(1) - [φ(0) + φ'(0) + ½φ''(0) + ⅙φ'''(0)]| ≤ M/24`. -/
theorem cubic_model_error {f : ℝ → ℝ} {M : ℝ}
    (hf : ContDiffOn ℝ 3 f (Icc 0 1))
    (hf' : DifferentiableOn ℝ (iteratedDerivWithin 3 f (Icc 0 1)) (Ioo 0 1))
    (hbound : ∀ x ∈ Ioo (0 : ℝ) 1, |iteratedDerivWithin 4 f (Icc 0 1) x| ≤ M) :
    |f 1 - (f 0 + iteratedDerivWithin 1 f (Icc 0 1) 0
        + iteratedDerivWithin 2 f (Icc 0 1) 0 / 2
        + iteratedDerivWithin 3 f (Icc 0 1) 0 / 6)| ≤ M / 24 := by
  have h01 : (0 : ℝ) ≠ 1 := by norm_num
  have huIcc : uIcc (0 : ℝ) 1 = Icc 0 1 := uIcc_of_le (by norm_num)
  have huIoo : uIoo (0 : ℝ) 1 = Ioo 0 1 := uIoo_of_le (by norm_num)
  obtain ⟨x', hx', hrem⟩ := taylor_mean_remainder_lagrange (n := 3) h01
    (by rw [huIcc]; exact_mod_cast hf) (by rw [huIcc, huIoo]; exact hf')
  rw [huIcc] at hrem
  rw [huIoo] at hx'
  have hexpand : taylorWithinEval f 3 (Icc 0 1) 0 1
      = f 0 + iteratedDerivWithin 1 f (Icc 0 1) 0
        + iteratedDerivWithin 2 f (Icc 0 1) 0 / 2
        + iteratedDerivWithin 3 f (Icc 0 1) 0 / 6 := by
    simp [taylorWithinEval_succ, taylor_within_zero_eval]
    norm_num
    ring
  rw [hexpand] at hrem
  rw [hrem]
  have hb := hbound x' hx'
  have h1 : iteratedDerivWithin (3 + 1) f (Icc 0 1) x' * (1 - 0) ^ (3 + 1)
        / ↑(Nat.factorial (3 + 1))
      = iteratedDerivWithin 4 f (Icc 0 1) x' / 24 := by
    norm_num [Nat.factorial]
  rw [h1, abs_div, abs_of_pos (show (0 : ℝ) < 24 by norm_num),
    div_eq_mul_inv, div_eq_mul_inv]
  exact mul_le_mul_of_nonneg_right hb (by norm_num)

end Kenian
