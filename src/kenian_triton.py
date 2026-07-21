"""Triton kernels for Kenian's fused elementwise update path.

The kernels mirror :class:`kenian_backends.TorchBackend`; the CUDA test suite
checks each primitive and the complete optimizer update against that reference.
"""

from __future__ import annotations

import math
import torch
import triton
import triton.language as tl


@triton.jit
def _phase_a_adamw_kernel(
    g_ptr,
    m_ptr,
    v_ptr,
    base_ptr,
    denom_ptr,
    n,
    b1,
    b2,
    one_m_b1,
    one_m_b2,
    eps,
    inv_sqrt_bc2,
    neg_lr_over_bc1,
    BLOCK: tl.constexpr,
):
    pid = tl.program_id(0)
    off = pid * BLOCK + tl.arange(0, BLOCK)
    mask = off < n
    g = tl.load(g_ptr + off, mask=mask, other=0.0)
    m = tl.load(m_ptr + off, mask=mask, other=0.0)
    v = tl.load(v_ptr + off, mask=mask, other=0.0)
    m = b1 * m + one_m_b1 * g
    v = b2 * v + one_m_b2 * g * g
    denom = tl.sqrt(v) * inv_sqrt_bc2 + eps
    base = neg_lr_over_bc1 * m / denom
    tl.store(m_ptr + off, m, mask=mask)
    tl.store(v_ptr + off, v, mask=mask)
    tl.store(base_ptr + off, base, mask=mask)
    tl.store(denom_ptr + off, denom, mask=mask)


@triton.jit
def _partial_norms_kernel(
    base_ptr,
    denom_ptr,
    slice_ptr,
    base_p_ptr,
    base_l2_ptr,
    slice_p_ptr,
    n,
    HAS_SLICE: tl.constexpr,
    BLOCK: tl.constexpr,
):
    pid = tl.program_id(0)
    off = pid * BLOCK + tl.arange(0, BLOCK)
    mask = off < n
    base = tl.load(base_ptr + off, mask=mask, other=0.0)
    denom = tl.load(denom_ptr + off, mask=mask, other=1.0)
    b2 = base * base
    tl.store(base_p_ptr + pid, tl.sum(b2 * denom, axis=0))
    tl.store(base_l2_ptr + pid, tl.sum(b2, axis=0))
    if HAS_SLICE:
        slice_value = tl.load(slice_ptr + off, mask=mask, other=0.0)
        tl.store(
            slice_p_ptr + pid, tl.sum(0.25 * slice_value * slice_value / denom, axis=0)
        )


@triton.jit
def _phase_b_kernel(
    parameter_ptr,
    base_ptr,
    denom_ptr,
    slice_ptr,
    update_ptr,
    n,
    correction_scale,
    decay_factor,
    HAS_SLICE: tl.constexpr,
    BLOCK: tl.constexpr,
):
    pid = tl.program_id(0)
    off = pid * BLOCK + tl.arange(0, BLOCK)
    mask = off < n
    base = tl.load(base_ptr + off, mask=mask, other=0.0)
    if HAS_SLICE:
        denom = tl.load(denom_ptr + off, mask=mask, other=1.0)
        slice_value = tl.load(slice_ptr + off, mask=mask, other=0.0)
        upd = base - correction_scale * slice_value / denom
    else:
        upd = base
    parameter = tl.load(parameter_ptr + off, mask=mask, other=0.0)
    parameter = parameter * decay_factor + upd
    tl.store(parameter_ptr + off, parameter, mask=mask)
    tl.store(update_ptr + off, upd, mask=mask)


_BLOCK = 1024


def _grid(numel):
    return (triton.cdiv(numel, _BLOCK),)


class TritonBackend:
    name = "triton"

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
        gradient = gradient.reshape(-1)
        exp_avg = exp_avg.reshape(-1)
        exp_avg_sq = exp_avg_sq.reshape(-1)
        numel = gradient.numel()
        base_step = torch.empty_like(gradient)
        denominator = torch.empty_like(gradient)
        _phase_a_adamw_kernel[_grid(numel)](
            gradient,
            exp_avg,
            exp_avg_sq,
            base_step,
            denominator,
            numel,
            beta1,
            beta2,
            1.0 - beta1,
            1.0 - beta2,
            epsilon,
            1.0 / math.sqrt(bias_correction2),
            -learning_rate / bias_correction1,
            BLOCK=_BLOCK,
        )
        return base_step, denominator

    def partial_norms(self, base_step, denominator, slice_ema):
        base_step = base_step.reshape(-1)
        denominator = denominator.reshape(-1)
        numel = base_step.numel()
        blocks = _grid(numel)[0]
        base_p = torch.empty(blocks, device=base_step.device, dtype=base_step.dtype)
        base_l2 = torch.empty(blocks, device=base_step.device, dtype=base_step.dtype)
        slice_p = torch.zeros(blocks, device=base_step.device, dtype=base_step.dtype)
        has_slice = slice_ema is not None
        slice_value = slice_ema.reshape(-1) if has_slice else base_step
        _partial_norms_kernel[(blocks,)](
            base_step,
            denominator,
            slice_value,
            base_p,
            base_l2,
            slice_p,
            numel,
            HAS_SLICE=has_slice,
            BLOCK=_BLOCK,
        )
        return float(base_p.sum()), float(base_l2.sum()), float(slice_p.sum())

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
        parameter_flat = parameter.reshape(-1)
        base_step = base_step.reshape(-1)
        denominator = denominator.reshape(-1)
        numel = parameter_flat.numel()
        update = torch.empty_like(base_step)
        has_slice = slice_ema is not None
        slice_value = slice_ema.reshape(-1) if has_slice else base_step
        _phase_b_kernel[_grid(numel)](
            parameter_flat,
            base_step,
            denominator,
            slice_value,
            update,
            numel,
            correction_scale,
            1.0 - learning_rate * weight_decay,
            HAS_SLICE=has_slice,
            BLOCK=_BLOCK,
        )
        return update.reshape(parameter.shape)
