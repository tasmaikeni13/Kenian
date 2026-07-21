"""JIT loader and backend for the custom CUDA Kenian kernels."""

from __future__ import annotations

import os

_EXT = None


def _load_extension():
    global _EXT
    if _EXT is None:
        from torch.utils.cpp_extension import load

        source_dir = os.path.dirname(os.path.abspath(__file__))
        source_path = os.path.join(source_dir, "..", "kernels", "kenian_ext.cu")
        _EXT = load(
            name="kenian_ext",
            sources=[source_path],
            verbose=False,
            extra_cuda_cflags=["-O3"],
        )
    return _EXT


class CudaBackend:
    name = "cuda"

    def __init__(self):
        self.extension = _load_extension()

    def phase_a_adamw(
        self,
        gradient,
        exp_avg,
        exp_avg_sq,
        learning_rate,
        beta1,
        beta2,
        epsilon,
        bias_correction1,
        bias_correction2,
    ):
        return self.extension.phase_a_adamw(
            gradient.contiguous(),
            exp_avg,
            exp_avg_sq,
            learning_rate,
            beta1,
            beta2,
            epsilon,
            bias_correction1,
            bias_correction2,
        )

    def partial_norms(self, base_step, denominator, slice_ema):
        out = self.extension.partial_norms(
            base_step.contiguous(),
            denominator.contiguous(),
            slice_ema.contiguous() if slice_ema is not None else None,
        )
        return float(out[0]), float(out[1]), float(out[2])

    def phase_b(
        self,
        parameter,
        base_step,
        denominator,
        slice_ema,
        correction_scale,
        learning_rate,
        weight_decay,
    ):
        return self.extension.phase_b(
            parameter,
            base_step.contiguous(),
            denominator.contiguous(),
            slice_ema.contiguous() if slice_ema is not None else None,
            correction_scale,
            learning_rate,
            weight_decay,
        )
