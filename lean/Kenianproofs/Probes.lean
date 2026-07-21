import Mathlib
set_option linter.style.header false
set_option linter.style.longLine false

/-!
# [Unbiased-Probes] — Rademacher probes recover exact diagonal slices

The optimizer estimates the diagonal curvature `diag H` and its Kenian drift
`diag 𝒦[Δ,·,·]` with a single Rademacher probe `z`:

* `E_z[z ⊙ (H z)]     = diag H`                    (Hutchinson, order 2)
* `E_z[z ⊙ 𝒦[Δ,z,·]]  = diag 𝒦[Δ,·,·]`             (Kenian drift, order 3)

We formalize the expectation as the exact average over all `2^n` sign vectors
(`σ : ι → Bool`, sign `s σ i = ±1`).  The key combinatorial fact is
`∑_σ s σ i * s σ k = 2^n · [i = k]`, proven by an involution (flip the `k`-th sign)
for `i ≠ k`.
-/

namespace Kenian

open Finset

variable {ι : Type*} [Fintype ι] [DecidableEq ι]

/-- Sign of a Boolean: `true ↦ 1`, `false ↦ -1`. -/
def sgn (b : Bool) : ℝ := if b then 1 else -1

@[simp] theorem sgn_not (b : Bool) : sgn (!b) = -sgn b := by
  cases b <;> simp [sgn]

@[simp] theorem sgn_mul_self (b : Bool) : sgn b * sgn b = 1 := by
  cases b <;> norm_num [sgn]

/-- Flipping one coordinate is an involution of sign-vector space. -/
theorem flip_involutive (k : ι) :
    Function.Involutive (fun σ : ι → Bool => Function.update σ k (!σ k)) := by
  intro σ
  funext j
  rcases eq_or_ne j k with rfl | hj
  · simp
  · simp [Function.update_of_ne hj]

/-- **Second-moment orthogonality**: `∑_σ s σ i · s σ k = 0` for `i ≠ k`. -/
theorem sum_sgn_mul_sgn_of_ne {i k : ι} (hik : i ≠ k) :
    ∑ σ : ι → Bool, sgn (σ i) * sgn (σ k) = 0 := by
  have hswap : ∑ σ : ι → Bool,
        sgn (Function.update σ k (!σ k) i) * sgn (Function.update σ k (!σ k) k)
      = ∑ σ : ι → Bool, sgn (σ i) * sgn (σ k) :=
    Fintype.sum_bijective _ (flip_involutive k).bijective _ _ (fun σ => rfl)
  have hterm : ∀ σ : ι → Bool,
      sgn (Function.update σ k (!σ k) i) * sgn (Function.update σ k (!σ k) k)
        = -(sgn (σ i) * sgn (σ k)) := by
    intro σ
    rw [Function.update_of_ne hik, Function.update_self, sgn_not]
    ring
  have hneg : ∑ σ : ι → Bool, sgn (σ i) * sgn (σ k)
      = -∑ σ : ι → Bool, sgn (σ i) * sgn (σ k) := by
    conv_lhs => rw [← hswap]
    rw [Finset.sum_congr rfl fun σ _ => hterm σ, Finset.sum_neg_distrib]
  linarith

/-- Second-moment identity: `∑_σ s σ i · s σ k = 2^n · [i = k]`. -/
theorem sum_sgn_mul_sgn (i k : ι) :
    ∑ σ : ι → Bool, sgn (σ i) * sgn (σ k)
      = if i = k then (2 : ℝ) ^ Fintype.card ι else 0 := by
  rcases eq_or_ne i k with rfl | hik
  · simp only [if_pos rfl]
    have : ∀ σ : ι → Bool, sgn (σ i) * sgn (σ i) = 1 := fun σ => sgn_mul_self _
    rw [Finset.sum_congr rfl fun σ _ => this σ, Finset.sum_const, Finset.card_univ]
    simp [Fintype.card_fun]
  · rw [if_neg hik]
    exact sum_sgn_mul_sgn_of_ne hik

/-- **Hutchinson unbiasedness (order 2)**: averaging `z_i (M z)_i` over all sign vectors
recovers `M_ii` exactly. -/
theorem hutchinson_unbiased (M : ι → ι → ℝ) (i : ι) :
    (∑ σ : ι → Bool, sgn (σ i) * ∑ k, M i k * sgn (σ k))
      = (2 : ℝ) ^ Fintype.card ι * M i i := by
  have expand : ∀ σ : ι → Bool, sgn (σ i) * ∑ k, M i k * sgn (σ k)
      = ∑ k, M i k * (sgn (σ i) * sgn (σ k)) := by
    intro σ
    rw [Finset.mul_sum]
    exact Finset.sum_congr rfl fun k _ => by ring
  rw [Finset.sum_congr rfl fun σ _ => expand σ, Finset.sum_comm]
  have inner : ∀ k, ∑ σ : ι → Bool, M i k * (sgn (σ i) * sgn (σ k))
      = M i k * (if i = k then (2 : ℝ) ^ Fintype.card ι else 0) := by
    intro k
    rw [← Finset.mul_sum, sum_sgn_mul_sgn]
  rw [Finset.sum_congr rfl fun k _ => inner k]
  simp only [mul_ite, mul_zero]
  rw [Finset.sum_ite_eq]
  simp [mul_comm]

/-- **Kenian probe unbiasedness (order 3)**: averaging `z_i · 𝒦[Δ, z, ·]_i` over all sign
vectors recovers the diagonal drift `∑_j T i j i Δ_j` exactly. -/
theorem kenian_probe_unbiased (T : ι → ι → ι → ℝ) (D : ι → ℝ) (i : ι) :
    (∑ σ : ι → Bool, sgn (σ i) * ∑ j, ∑ k, T i j k * D j * sgn (σ k))
      = (2 : ℝ) ^ Fintype.card ι * ∑ j, T i j i * D j := by
  have expand : ∀ σ : ι → Bool, sgn (σ i) * ∑ j, ∑ k, T i j k * D j * sgn (σ k)
      = ∑ j, ∑ k, T i j k * D j * (sgn (σ i) * sgn (σ k)) := by
    intro σ
    rw [Finset.mul_sum]
    refine Finset.sum_congr rfl fun j _ => ?_
    rw [Finset.mul_sum]
    exact Finset.sum_congr rfl fun k _ => by ring
  rw [Finset.sum_congr rfl fun σ _ => expand σ, Finset.sum_comm]
  have inner : ∀ j, ∑ σ : ι → Bool, ∑ k, T i j k * D j * (sgn (σ i) * sgn (σ k))
      = (2 : ℝ) ^ Fintype.card ι * (T i j i * D j) := by
    intro j
    rw [Finset.sum_comm]
    have inner2 : ∀ k, ∑ σ : ι → Bool, T i j k * D j * (sgn (σ i) * sgn (σ k))
        = T i j k * D j * (if i = k then (2 : ℝ) ^ Fintype.card ι else 0) := by
      intro k
      rw [← Finset.mul_sum, sum_sgn_mul_sgn]
    rw [Finset.sum_congr rfl fun k _ => inner2 k]
    simp only [mul_ite, mul_zero]
    rw [Finset.sum_ite_eq]
    simp [mul_comm]
  rw [Finset.sum_congr rfl fun j _ => inner j, ← Finset.mul_sum]

/-- With the Kenian's full symmetry, `∑_j T i j i Δ_j` is exactly the diagonal of the
directional Kenian: `diag 𝒦[Δ,·,·]_i = ∑_k 𝒦_{i i k} Δ_k`. -/
theorem kenian_probe_target_eq_diag (T : ι → ι → ι → ℝ)
    (hsymm : ∀ a b c, T a b c = T a c b) (D : ι → ℝ) (i : ι) :
    ∑ j, T i j i * D j = ∑ k, T i i k * D k :=
  Finset.sum_congr rfl fun j _ => by rw [hsymm i j i]

end Kenian
