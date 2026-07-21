import Mathlib
set_option linter.style.header false

/-!
# [Slice-Exactness] — one Kenian slice per step buys one order of local accuracy

The local cubic model of the loss has gradient field
`g(θ) = g₀ + H θ + ½ 𝒦[θ, θ, ·]`.  Here `B : E →L[ℝ] E →L[ℝ] E` plays the role of the
Kenian's action `𝒦[·, ·, ·]` (a continuous bilinear, vector-valued map; symmetry of the
two contracted slots is assumed where needed), and `H` is a continuous linear equivalence.

Formal statements used by the optimizer derivation:

* `newton_residual` — the Newton step `Δ_N = -H⁻¹ g₀` leaves gradient residual exactly
  `½ B[Δ_N, Δ_N]`: the leading error of a second-order method IS the Kenian slice.
* `chebyshev_residual` — correcting with that one slice, `Δ_C = Δ_N - ½ H⁻¹ B[Δ_N, Δ_N]`,
  leaves residual `B[Δ_N, c] + ½ B[c, c]` where `c` is the correction itself.
* `chebyshev_gains_an_order` — hence
  `‖res_C‖ ≤ ½‖B‖²‖H⁻¹‖‖Δ_N‖³ + ⅛‖B‖³‖H⁻¹‖²‖Δ_N‖⁴` (third order in the step),
  versus Newton's exactly-second-order residual `½‖B[Δ_N, Δ_N]‖ ≤ ½‖B‖‖Δ_N‖²`.
-/

namespace Kenian

variable {E : Type*} [NormedAddCommGroup E] [NormedSpace ℝ E]

/-- Gradient field of the local cubic model `L(θ) = L₀ + ⟨g₀,θ⟩ + ½⟨θ,Hθ⟩ + ⅙𝒦[θ,θ,θ]`. -/
noncomputable def gradModel (g₀ : E) (H : E →L[ℝ] E) (B : E →L[ℝ] E →L[ℝ] E) (θ : E) : E :=
  g₀ + H θ + (2⁻¹ : ℝ) • B θ θ

/-- The Newton step of the model. -/
def newtonStep (g₀ : E) (H : E ≃L[ℝ] E) : E := -(H.symm g₀)

/-- The Kenian (Chebyshev-type) correction: precondition the predicted third-order
gradient excess over the step. -/
noncomputable def kenianCorrection (g₀ : E) (H : E ≃L[ℝ] E) (B : E →L[ℝ] E →L[ℝ] E) : E :=
  -((2⁻¹ : ℝ) • H.symm (B (newtonStep g₀ H) (newtonStep g₀ H)))

/-- The corrected (third-order) step. -/
noncomputable def kenianStep (g₀ : E) (H : E ≃L[ℝ] E) (B : E →L[ℝ] E →L[ℝ] E) : E :=
  newtonStep g₀ H + kenianCorrection g₀ H B

/-- **Newton residual identity**: the entire post-Newton gradient is the Kenian slice. -/
theorem newton_residual (g₀ : E) (H : E ≃L[ℝ] E) (B : E →L[ℝ] E →L[ℝ] E) :
    gradModel g₀ (H : E →L[ℝ] E) B (newtonStep g₀ H)
      = (2⁻¹ : ℝ) • B (newtonStep g₀ H) (newtonStep g₀ H) := by
  unfold gradModel newtonStep
  simp

/-- Bilinear expansion of `B` over a sum in both slots. -/
theorem bilin_expand (B : E →L[ℝ] E →L[ℝ] E) (d c : E) :
    B (d + c) (d + c) = B d d + B d c + (B c d + B c c) := by
  rw [map_add]
  simp only [ContinuousLinearMap.add_apply, map_add]
  abel

/-- General residual identity: any step `d + c` with `H d = -g₀` (base Newton direction)
and `H c = -½ B[d,d]` (slice correction) leaves residual `B[d,c] + ½ B[c,c]`. -/
theorem residual_after_corrected_step (g₀ : E) (H : E →L[ℝ] E) (B : E →L[ℝ] E →L[ℝ] E)
    (hsymm : ∀ u v, B u v = B v u) (d c : E)
    (hHd : H d = -g₀) (hHc : H c = -((2⁻¹ : ℝ) • B d d)) :
    gradModel g₀ H B (d + c) = B d c + (2⁻¹ : ℝ) • B c c := by
  unfold gradModel
  rw [map_add, hHd, hHc, bilin_expand B d c, hsymm c d]
  module

/-- **Chebyshev residual identity** for the model's Newton step and Kenian correction. -/
theorem chebyshev_residual (g₀ : E) (H : E ≃L[ℝ] E) (B : E →L[ℝ] E →L[ℝ] E)
    (hsymm : ∀ u v, B u v = B v u) :
    gradModel g₀ (H : E →L[ℝ] E) B (kenianStep g₀ H B)
      = B (newtonStep g₀ H) (kenianCorrection g₀ H B)
        + (2⁻¹ : ℝ) • B (kenianCorrection g₀ H B) (kenianCorrection g₀ H B) := by
  refine residual_after_corrected_step g₀ _ B hsymm _ _ ?_ ?_
  · unfold newtonStep; simp
  · unfold kenianCorrection; simp

/-- Norm bound on the corrected residual in terms of step and correction sizes. -/
theorem chebyshev_residual_bound (g₀ : E) (H : E ≃L[ℝ] E) (B : E →L[ℝ] E →L[ℝ] E)
    (hsymm : ∀ u v, B u v = B v u) :
    ‖gradModel g₀ (H : E →L[ℝ] E) B (kenianStep g₀ H B)‖
      ≤ ‖B‖ * ‖newtonStep g₀ H‖ * ‖kenianCorrection g₀ H B‖
        + 2⁻¹ * (‖B‖ * ‖kenianCorrection g₀ H B‖ ^ 2) := by
  rw [chebyshev_residual g₀ H B hsymm]
  refine (norm_add_le _ _).trans ?_
  have h1 := B.le_opNorm₂ (newtonStep g₀ H) (kenianCorrection g₀ H B)
  have h2 := B.le_opNorm₂ (kenianCorrection g₀ H B) (kenianCorrection g₀ H B)
  have h3 : ‖(2⁻¹ : ℝ) • B (kenianCorrection g₀ H B) (kenianCorrection g₀ H B)‖
      = 2⁻¹ * ‖B (kenianCorrection g₀ H B) (kenianCorrection g₀ H B)‖ := by
    rw [norm_smul]; norm_num
  rw [h3, sq]
  nlinarith [norm_nonneg (B (kenianCorrection g₀ H B) (kenianCorrection g₀ H B))]

/-- The correction is second-order small: `‖c‖ ≤ ½‖H⁻¹‖‖B‖‖Δ_N‖²`. -/
theorem kenianCorrection_bound (g₀ : E) (H : E ≃L[ℝ] E) (B : E →L[ℝ] E →L[ℝ] E) :
    ‖kenianCorrection g₀ H B‖
      ≤ 2⁻¹ * ‖(H.symm : E →L[ℝ] E)‖ * ‖B‖ * ‖newtonStep g₀ H‖ ^ 2 := by
  unfold kenianCorrection
  rw [norm_neg, norm_smul]
  simp only [norm_inv, Real.norm_ofNat]
  have h1 : ‖H.symm (B (newtonStep g₀ H) (newtonStep g₀ H))‖
      ≤ ‖(H.symm : E →L[ℝ] E)‖ * ‖B (newtonStep g₀ H) (newtonStep g₀ H)‖ :=
    (H.symm : E →L[ℝ] E).le_opNorm _
  have h2 : ‖B (newtonStep g₀ H) (newtonStep g₀ H)‖
      ≤ ‖B‖ * ‖newtonStep g₀ H‖ * ‖newtonStep g₀ H‖ := B.le_opNorm₂ _ _
  have hs : (0:ℝ) ≤ ‖(H.symm : E →L[ℝ] E)‖ := norm_nonneg _
  nlinarith [norm_nonneg (B (newtonStep g₀ H) (newtonStep g₀ H)),
    norm_nonneg (newtonStep g₀ H)]

/-- **One order gained** [Slice-Exactness]: the corrected residual is `O(‖Δ_N‖³)`
(with an explicit quartic tail), versus Newton's `O(‖Δ_N‖²)`. -/
theorem chebyshev_gains_an_order (g₀ : E) (H : E ≃L[ℝ] E) (B : E →L[ℝ] E →L[ℝ] E)
    (hsymm : ∀ u v, B u v = B v u) :
    ‖gradModel g₀ (H : E →L[ℝ] E) B (kenianStep g₀ H B)‖
      ≤ 2⁻¹ * ‖B‖ ^ 2 * ‖(H.symm : E →L[ℝ] E)‖ * ‖newtonStep g₀ H‖ ^ 3
        + 8⁻¹ * ‖B‖ ^ 3 * ‖(H.symm : E →L[ℝ] E)‖ ^ 2 * ‖newtonStep g₀ H‖ ^ 4 := by
  have h0 := chebyshev_residual_bound g₀ H B hsymm
  have h1 := kenianCorrection_bound g₀ H B
  have hB : (0:ℝ) ≤ ‖B‖ := norm_nonneg _
  have hH : (0:ℝ) ≤ ‖(H.symm : E →L[ℝ] E)‖ := norm_nonneg _
  have hd : (0:ℝ) ≤ ‖newtonStep g₀ H‖ := norm_nonneg _
  have hc : (0:ℝ) ≤ ‖kenianCorrection g₀ H B‖ := norm_nonneg _
  have key1 : ‖B‖ * ‖newtonStep g₀ H‖ * ‖kenianCorrection g₀ H B‖
      ≤ ‖B‖ * ‖newtonStep g₀ H‖ * (2⁻¹ * ‖(H.symm : E →L[ℝ] E)‖ * ‖B‖ * ‖newtonStep g₀ H‖ ^ 2) := by
    have hbn : (0:ℝ) ≤ ‖B‖ * ‖newtonStep g₀ H‖ := mul_nonneg hB hd
    exact mul_le_mul_of_nonneg_left h1 hbn
  have key2 : ‖kenianCorrection g₀ H B‖ ^ 2
      ≤ (2⁻¹ * ‖(H.symm : E →L[ℝ] E)‖ * ‖B‖ * ‖newtonStep g₀ H‖ ^ 2) ^ 2 := by
    have := mul_le_mul h1 h1 hc (by positivity)
    calc ‖kenianCorrection g₀ H B‖ ^ 2
        = ‖kenianCorrection g₀ H B‖ * ‖kenianCorrection g₀ H B‖ := sq _
      _ ≤ _ := this
      _ = _ := (sq _).symm
  nlinarith [key1, key2, mul_le_mul_of_nonneg_left key2 (mul_nonneg (by norm_num : (0:ℝ) ≤ 2⁻¹) hB)]

end Kenian
