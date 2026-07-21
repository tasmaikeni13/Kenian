# Kenian formalization

This Lean 4 project contains the algebraic results that motivate the Kenian
update. The modules cover Taylor and Chebyshev identities, descent under a
capped correction, EMA lag, randomized probes, and softmax cross-entropy
cumulants.

Build the project with:

```bash
lake build
```

The proofs establish the stated mathematical identities and bounds. They do
not make claims about empirical performance on neural-network benchmarks.
