"""Backend implementations for Kenian's elementwise update path."""

from __future__ import annotations

import math
from typing import Any

import torch


class TorchBackend:
    """Reference implementation used for CPU execution and backend validation."""

    name = "torch"

    def phase_a_adamw(
        self,
        gradient: torch.Tensor,
        exp_avg: torch.Tensor,
        exp_avg_sq: torch.Tensor,
        learning_rate: float,
        beta1: float,
        beta2: float,
        epsilon: float,
        bias_correction1: float,
        bias_correction2: float,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Update AdamW moments and return its preconditioned base step."""
        exp_avg.mul_(beta1).add_(gradient, alpha=1.0 - beta1)
        exp_avg_sq.mul_(beta2).addcmul_(gradient, gradient, value=1.0 - beta2)
        denominator = exp_avg_sq.sqrt().div_(math.sqrt(bias_correction2)).add_(epsilon)
        base_step = exp_avg.div(denominator).mul_(-learning_rate / bias_correction1)
        return base_step, denominator

    def partial_norms(
        self,
        base_step: torch.Tensor,
        denominator: torch.Tensor,
        slice_ema: torch.Tensor | None,
    ) -> tuple[float, float, float]:
        """Return reduction terms used to globally cap the correction."""
        base_l2_norm_sq = float(base_step.square().sum())
        base_p_norm_sq = float((base_step.square() * denominator).sum())
        slice_p_norm_sq = 0.0
        if slice_ema is not None:
            slice_p_norm_sq = float((slice_ema.square() / denominator).sum()) * 0.25
        return base_p_norm_sq, base_l2_norm_sq, slice_p_norm_sq

    def phase_b(
        self,
        parameter: torch.Tensor,
        base_step: torch.Tensor,
        denominator: torch.Tensor,
        slice_ema: torch.Tensor | None,
        correction_scale: float,
        learning_rate: float,
        weight_decay: float,
    ) -> torch.Tensor:
        """Apply decoupled weight decay and the complete parameter update."""
        correction = 0.0
        if slice_ema is not None:
            correction = correction_scale * slice_ema / denominator
        update = base_step - correction
        if weight_decay:
            parameter.mul_(1.0 - learning_rate * weight_decay)
        parameter.add_(update)
        return update.detach().clone()


_BACKEND_CACHE: dict[str, Any] = {}


def get_backend(name: str) -> Any:
    """Return a cached update backend by name."""
    backend_name = name.lower()
    if backend_name in _BACKEND_CACHE:
        return _BACKEND_CACHE[backend_name]
    if backend_name == "torch":
        backend = TorchBackend()
    elif backend_name == "triton":
        from kenian_triton import TritonBackend

        backend = TritonBackend()
    elif backend_name == "cuda":
        from kenian_cuda import CudaBackend

        backend = CudaBackend()
    else:
        raise ValueError(
            "unknown backend {!r}; use torch, triton, or cuda".format(name)
        )
    _BACKEND_CACHE[backend_name] = backend
    return backend
