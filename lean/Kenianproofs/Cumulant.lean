import Mathlib
set_option linter.style.header false
set_option linter.style.longLine false

/-!
# [Cumulant-Identity] and [Skewness-Bound] — the output-space Kenian of softmax
cross-entropy is the third cumulant of the predictive distribution, and it is
dominated by logit span × curvature

For logits `z : ι → ℝ`, direction `u : ι → ℝ`, true class `y`, let
`ℓ(t) = log ∑_c exp(z_c + t u_c) - (z_y + t u_y)` (cross-entropy along `u`).

With `w_c(t)` the softmax weights at the shifted logits and `μ(t) = ∑ w_c u_c`:

* `deriv2_ceLoss` : `ℓ''(t) = ∑ w_c (u_c - μ)²`  (variance of `u` under the prediction)
* `deriv3_ceLoss` : `ℓ'''(t) = ∑ w_c (u_c - μ)³`  (third central moment — the skewness
  structure; this is the Kenian of the loss contracted three times with `u`)
* `kenian_slice_dominated` : `|ℓ'''(t)| ≤ (max u - min u) · ℓ''(t)` — the third-order
  term can never exceed the logit oscillation times the second-order (Gauss–Newton)
  quadratic form.  This is the a-priori justification for the capped correction.
-/

namespace Kenian

open Finset Real

variable {ι : Type*} [Fintype ι] [Nonempty ι]

/-- Unnormalized directional moment sums: `S_m(t) = ∑_c u_c^m · e^{z_c + t u_c}`. -/
noncomputable def S (z u : ι → ℝ) (m : ℕ) (t : ℝ) : ℝ :=
  ∑ c, (u c) ^ m * Real.exp (z c + t * u c)

/-- Softmax weights at the shifted logits `z + t u`. -/
noncomputable def w (z u : ι → ℝ) (t : ℝ) (c : ι) : ℝ :=
  Real.exp (z c + t * u c) / S z u 0 t

/-- Predictive mean of `u`. -/
noncomputable def dirMean (z u : ι → ℝ) (t : ℝ) : ℝ := ∑ c, w z u t c * u c

/-- Predictive central moment of `u` of order `m`. -/
noncomputable def centralMoment (z u : ι → ℝ) (m : ℕ) (t : ℝ) : ℝ :=
  ∑ c, w z u t c * (u c - dirMean z u t) ^ m

/-- Softmax cross-entropy along direction `u`. -/
noncomputable def ceLoss (z u : ι → ℝ) (y : ι) (t : ℝ) : ℝ :=
  Real.log (S z u 0 t) - (z y + t * u y)

theorem S0_pos (z u : ι → ℝ) (t : ℝ) : 0 < S z u 0 t := by
  unfold S
  refine Finset.sum_pos (fun c _ => ?_) Finset.univ_nonempty
  simp [Real.exp_pos]

theorem S0_ne (z u : ι → ℝ) (t : ℝ) : S z u 0 t ≠ 0 := ne_of_gt (S0_pos z u t)

theorem hasDerivAt_S (z u : ι → ℝ) (m : ℕ) (t : ℝ) :
    HasDerivAt (S z u m) (S z u (m + 1) t) t := by
  unfold S
  have h : ∀ c ∈ (Finset.univ : Finset ι),
      HasDerivAt (fun s => (u c) ^ m * Real.exp (z c + s * u c))
        ((u c) ^ (m + 1) * Real.exp (z c + t * u c)) t := by
    intro c _
    have hlin : HasDerivAt (fun s : ℝ => z c + s * u c) (u c) t := by
      simpa using (hasDerivAt_mul_const (u c)).const_add (z c)
    have hexp := hlin.exp
    have hmul := hexp.const_mul ((u c) ^ m)
    have heq : (u c) ^ (m + 1) * Real.exp (z c + t * u c)
        = (u c) ^ m * (Real.exp (z c + t * u c) * u c) := by ring
    rw [heq]
    exact hmul
  have hfn : (fun s => ∑ c : ι, (u c) ^ m * Real.exp (z c + s * u c))
      = ∑ c : ι, (fun s => (u c) ^ m * Real.exp (z c + s * u c)) := by
    funext s
    rw [Finset.sum_apply]
  rw [hfn]
  exact HasDerivAt.sum h

/-- Normalized moments: `∑ w u^m = S_m / S_0`. -/
theorem sum_w_pow (z u : ι → ℝ) (t : ℝ) (m : ℕ) :
    ∑ c, w z u t c * (u c) ^ m = S z u m t / S z u 0 t := by
  unfold w S
  rw [Finset.sum_div]
  exact Finset.sum_congr rfl fun c _ => by ring

theorem sum_w (z u : ι → ℝ) (t : ℝ) : ∑ c, w z u t c = 1 := by
  have h := sum_w_pow z u t 0
  simp only [pow_zero, mul_one] at h
  rw [h, div_self (S0_ne z u t)]

theorem w_nonneg (z u : ι → ℝ) (t : ℝ) (c : ι) : 0 ≤ w z u t c :=
  div_nonneg (le_of_lt (Real.exp_pos _)) (le_of_lt (S0_pos z u t))

theorem dirMean_eq (z u : ι → ℝ) (t : ℝ) : dirMean z u t = S z u 1 t / S z u 0 t := by
  have h := sum_w_pow z u t 1
  simp only [pow_one] at h
  unfold dirMean
  exact h

/-- Variance in raw-moment form. -/
theorem centralMoment_two_eq (z u : ι → ℝ) (t : ℝ) :
    centralMoment z u 2 t
      = (∑ c, w z u t c * (u c) ^ 2) - (dirMean z u t) ^ 2 := by
  unfold centralMoment
  have key : ∀ c ∈ (Finset.univ : Finset ι), w z u t c * (u c - dirMean z u t) ^ 2
      = w z u t c * (u c) ^ 2
        - 2 * dirMean z u t * (w z u t c * u c)
        + (dirMean z u t) ^ 2 * w z u t c := fun c _ => by ring
  rw [Finset.sum_congr rfl key, Finset.sum_add_distrib, Finset.sum_sub_distrib,
    ← Finset.mul_sum, ← Finset.mul_sum]
  have hmean : (∑ c, w z u t c * u c) = dirMean z u t := rfl
  rw [hmean, sum_w]
  ring

/-- Third central moment in raw-moment form. -/
theorem centralMoment_three_eq (z u : ι → ℝ) (t : ℝ) :
    centralMoment z u 3 t
      = (∑ c, w z u t c * (u c) ^ 3)
        - 3 * dirMean z u t * (∑ c, w z u t c * (u c) ^ 2)
        + 2 * (dirMean z u t) ^ 3 := by
  unfold centralMoment
  have key : ∀ c ∈ (Finset.univ : Finset ι), w z u t c * (u c - dirMean z u t) ^ 3
      = w z u t c * (u c) ^ 3
        - 3 * dirMean z u t * (w z u t c * (u c) ^ 2)
        + 3 * (dirMean z u t) ^ 2 * (w z u t c * u c)
        - (dirMean z u t) ^ 3 * w z u t c := fun c _ => by ring
  rw [Finset.sum_congr rfl key, Finset.sum_sub_distrib, Finset.sum_add_distrib,
    Finset.sum_sub_distrib, ← Finset.mul_sum, ← Finset.mul_sum, ← Finset.mul_sum]
  have hmean : (∑ c, w z u t c * u c) = dirMean z u t := rfl
  rw [hmean, sum_w]
  ring

/-- **First derivative**: `ℓ'(t) = μ(t) - u_y` (softmax mean minus target). -/
theorem hasDerivAt_ceLoss (z u : ι → ℝ) (y : ι) (t : ℝ) :
    HasDerivAt (ceLoss z u y) (dirMean z u t - u y) t := by
  unfold ceLoss
  have hlog := (hasDerivAt_S z u 0 t).log (S0_ne z u t)
  have hlin : HasDerivAt (fun s : ℝ => z y + s * u y) (u y) t := by
    simpa using (hasDerivAt_mul_const (u y)).const_add (z y)
  have h := hlog.sub hlin
  rw [dirMean_eq]
  exact h

/-- **Second derivative**: the mean's derivative is the variance. -/
theorem hasDerivAt_dirMean (z u : ι → ℝ) (t : ℝ) :
    HasDerivAt (dirMean z u) (centralMoment z u 2 t) t := by
  have hdiv := (hasDerivAt_S z u 1 t).div (hasDerivAt_S z u 0 t) (S0_ne z u t)
  have hfun : dirMean z u = fun s => S z u 1 s / S z u 0 s :=
    funext fun s => dirMean_eq z u s
  have hval : (S z u (1 + 1) t * S z u 0 t - S z u 1 t * S z u (0 + 1) t) / S z u 0 t ^ 2
      = centralMoment z u 2 t := by
    rw [centralMoment_two_eq, sum_w_pow, dirMean_eq]
    have h0 := S0_ne z u t
    show (S z u 2 t * S z u 0 t - S z u 1 t * S z u 1 t) / S z u 0 t ^ 2 = _
    field_simp
  rw [hfun, ← hval]
  exact hdiv

/-- **Third derivative**: the variance's derivative is the third central moment. -/
theorem hasDerivAt_centralMoment_two (z u : ι → ℝ) (t : ℝ) :
    HasDerivAt (centralMoment z u 2) (centralMoment z u 3 t) t := by
  have h0 := hasDerivAt_S z u 0 t
  have h1 := hasDerivAt_S z u 1 t
  have h2 := hasDerivAt_S z u 2 t
  have h0ne := S0_ne z u t
  have hM2 := h2.div h0 h0ne
  have hM1 := h1.div h0 h0ne
  have hM1sq := hM1.pow 2
  have hsub := hM2.sub hM1sq
  have hfun : centralMoment z u 2 = fun s => S z u 2 s / S z u 0 s - (S z u 1 s / S z u 0 s) ^ 2 := by
    funext s
    rw [centralMoment_two_eq, sum_w_pow, dirMean_eq]
  have hval : (S z u (2 + 1) t * S z u 0 t - S z u 2 t * S z u (0 + 1) t) / S z u 0 t ^ 2
        - (2 : ℕ) * (S z u 1 t / S z u 0 t) ^ (2 - 1)
          * ((S z u (1 + 1) t * S z u 0 t - S z u 1 t * S z u (0 + 1) t) / S z u 0 t ^ 2)
      = centralMoment z u 3 t := by
    rw [centralMoment_three_eq, sum_w_pow, sum_w_pow, dirMean_eq]
    have h0 := S0_ne z u t
    show (S z u 3 t * S z u 0 t - S z u 2 t * S z u 1 t) / S z u 0 t ^ 2
        - (2 : ℕ) * (S z u 1 t / S z u 0 t) ^ 1
          * ((S z u 2 t * S z u 0 t - S z u 1 t * S z u 1 t) / S z u 0 t ^ 2) = _
    push_cast
    field_simp
    ring
  rw [hfun, ← hval]
  exact hsub

/-- `ℓ' = μ - u_y` as functions. -/
theorem deriv_ceLoss (z u : ι → ℝ) (y : ι) :
    deriv (ceLoss z u y) = fun t => dirMean z u t - u y :=
  funext fun t => (hasDerivAt_ceLoss z u y t).deriv

/-- **[Cumulant-Identity], order 2**: `ℓ''(t) = Var_{w(t)}(u)`. -/
theorem deriv2_ceLoss (z u : ι → ℝ) (y : ι) (t : ℝ) :
    deriv (deriv (ceLoss z u y)) t = centralMoment z u 2 t := by
  rw [deriv_ceLoss]
  exact ((hasDerivAt_dirMean z u t).sub_const (u y)).deriv

/-- **[Cumulant-Identity], order 3**: `ℓ'''(t) = E_{w(t)}[(u - μ)³]` — the Kenian of
softmax-CE contracted three times with `u` is the predictive third central moment. -/
theorem deriv3_ceLoss (z u : ι → ℝ) (y : ι) (t : ℝ) :
    deriv (deriv (deriv (ceLoss z u y))) t = centralMoment z u 3 t := by
  rw [deriv_ceLoss]
  have hfun : deriv (fun s => dirMean z u s - u y) = centralMoment z u 2 := by
    funext s
    exact ((hasDerivAt_dirMean z u s).sub_const (u y)).deriv
  rw [hfun]
  exact (hasDerivAt_centralMoment_two z u t).deriv

/-- The predictive mean stays inside the range of `u`. -/
theorem dirMean_mem (z u : ι → ℝ) (t : ℝ) {a b : ℝ} (ha : ∀ c, a ≤ u c)
    (hb : ∀ c, u c ≤ b) : a ≤ dirMean z u t ∧ dirMean z u t ≤ b := by
  constructor
  · calc a = ∑ c, w z u t c * a := by rw [← Finset.sum_mul, sum_w, one_mul]
      _ ≤ ∑ c, w z u t c * u c :=
        Finset.sum_le_sum fun c _ => mul_le_mul_of_nonneg_left (ha c) (w_nonneg z u t c)
  · calc dirMean z u t = ∑ c, w z u t c * u c := rfl
      _ ≤ ∑ c, w z u t c * b :=
        Finset.sum_le_sum fun c _ => mul_le_mul_of_nonneg_left (hb c) (w_nonneg z u t c)
      _ = b := by rw [← Finset.sum_mul, sum_w, one_mul]

/-- **[Skewness-Bound]**: `|E (u-μ)³| ≤ (b - a) · E (u-μ)²` whenever `u ∈ [a, b]`. -/
theorem abs_centralMoment_three_le (z u : ι → ℝ) (t : ℝ) {a b : ℝ}
    (ha : ∀ c, a ≤ u c) (hb : ∀ c, u c ≤ b) :
    |centralMoment z u 3 t| ≤ (b - a) * centralMoment z u 2 t := by
  obtain ⟨hma, hmb⟩ := dirMean_mem z u t ha hb
  unfold centralMoment
  calc |∑ c, w z u t c * (u c - dirMean z u t) ^ 3|
      ≤ ∑ c, |w z u t c * (u c - dirMean z u t) ^ 3| := Finset.abs_sum_le_sum_abs _ _
    _ ≤ ∑ c, (b - a) * (w z u t c * (u c - dirMean z u t) ^ 2) := by
        refine Finset.sum_le_sum fun c _ => ?_
        have hw := w_nonneg z u t c
        have habs : |u c - dirMean z u t| ≤ b - a := by
          rw [abs_le]
          constructor
          · linarith [ha c, hmb]
          · linarith [hb c, hma]
        have h3 : |w z u t c * (u c - dirMean z u t) ^ 3|
            = w z u t c * ((u c - dirMean z u t) ^ 2 * |u c - dirMean z u t|) := by
          rw [abs_mul, abs_of_nonneg hw, abs_pow, pow_succ, sq_abs]
        rw [h3]
        have hstep : (u c - dirMean z u t) ^ 2 * |u c - dirMean z u t|
            ≤ (u c - dirMean z u t) ^ 2 * (b - a) :=
          mul_le_mul_of_nonneg_left habs (sq_nonneg _)
        calc w z u t c * ((u c - dirMean z u t) ^ 2 * |u c - dirMean z u t|)
            ≤ w z u t c * ((u c - dirMean z u t) ^ 2 * (b - a)) :=
              mul_le_mul_of_nonneg_left hstep hw
          _ = (b - a) * (w z u t c * (u c - dirMean z u t) ^ 2) := by ring
    _ = (b - a) * ∑ c, w z u t c * (u c - dirMean z u t) ^ 2 := by rw [← Finset.mul_sum]

/-- **Headline corollary**: for softmax-CE, the third-order directional term is dominated
by the logit oscillation times the curvature: `|ℓ'''| ≤ osc(u) · ℓ''`.  A capped
correction therefore suffices; the cubic term cannot run away from the quadratic one. -/
theorem kenian_slice_dominated (z u : ι → ℝ) (y : ι) (t : ℝ) {a b : ℝ}
    (ha : ∀ c, a ≤ u c) (hb : ∀ c, u c ≤ b) :
    |deriv (deriv (deriv (ceLoss z u y))) t|
      ≤ (b - a) * deriv (deriv (ceLoss z u y)) t := by
  rw [deriv3_ceLoss, deriv2_ceLoss]
  exact abs_centralMoment_three_le z u t ha hb

end Kenian
