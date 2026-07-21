# Kenian experiment report

## Summary

Kenian tests whether an optimizer can use exact, directional third-order
information without materializing a third-derivative tensor. It augments an
AdamW update with a small correction derived from `∇³L[Δ, Δ, ·]`, measured
along the optimizer's own base update.

This is a research result with a mixed outcome: the reported vision experiment
improved validation loss, while the reported language-model experiment was
consistently worse than AdamW. Kenian should therefore be viewed as a
well-tested direction for further work, not as a drop-in replacement.

## Method

The optimizer computes an AdamW base step and periodically takes two
reverse-mode derivatives to obtain a directional third-order slice. An EMA of
that slice is preconditioned by the AdamW denominator and globally capped.
The cap ensures that the correction remains a fixed small fraction of the
base step in the preconditioned norm. See [theory.md](theory.md) for the
derivation and assumptions.

The code uses a PyTorch reference backend plus optional Triton and custom CUDA
implementations for the elementwise update. Tests check AdamW correspondence
when the third-order term is disabled, the global cap, the exact slice on a
small smooth network, numerical identities, and CUDA backend equivalence.

## Experimental setup

Both benchmark runs used three seeds and 40,000 training steps.

| Domain | Model | Dataset | Parameters |
|---|---|---|---:|
| Vision | ViT with GELU activations | CIFAR-100 | 37.9M |
| Language | Decoder-only transformer with GELU | WikiText-103 | 39.0M |

AdamW and Kenian used the same models, schedules, and data pipeline. The
third-order probe ran every ten steps, with a correction cap of 0.1.

## Results

| Benchmark | AdamW validation loss | Kenian validation loss | Outcome |
|---|---:|---:|---|
| CIFAR-100 / ViT | 3.135 ± 0.023 | **3.067 ± 0.013** | Kenian improved loss |
| WikiText-103 / transformer | **3.182 ± 0.003** | 3.212 ± 0.006 | Kenian was worse |

On vision, the third-order slice remained stable and the small correction was
associated with lower validation loss. On language modeling, the slice was
near zero for much of training, yet the capped update still introduced a
persistent perturbation. The most plausible interpretation is that probe
noise outweighed any useful curvature signal in that regime.

## Limitations and next steps

The experiments cover only two workloads, use a small number of seeds, and
add meaningful probe overhead. The formal work verifies local identities and
safety bounds, not stochastic convergence or broad empirical superiority.

The clearest next step is a signal-quality gate: apply the correction only
when the directional slice is sufficiently large relative to its estimated
noise. Additional evaluation should focus on workloads with changing
curvature, longer training horizons, and larger seed counts.
