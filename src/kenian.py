"""Kenian, an AdamW optimizer with a trajectory-sliced third-order correction.

Kenian does not materialize the third-derivative tensor.  On a configurable
schedule, it evaluates the exact slice ``∇³L[Δ, Δ, ·]`` along the previous
base update and maintains an exponential moving average of that slice.  The
slice supplies a small, globally capped Chebyshev-style correction.

The training loop owns gradient computation because a scheduled probe needs a
second reverse-mode pass:

    optimizer.zero_grad(set_to_none=True)
    optimizer.backward_and_prepare(loss)
    optimizer.step()
"""

from __future__ import annotations

import math
from typing import Any, Iterator

import torch

from kenian_backends import get_backend


class Kenian(torch.optim.Optimizer):
    """AdamW with a capped, trajectory-sliced third-order correction.

    Set ``third_order_off=True`` to obtain an AdamW-equivalent update.  This
    is useful for ablations and for checking backend implementations.
    """

    def __init__(
        self,
        params: Any,
        lr: float = 3e-4,
        betas: tuple[float, float] = (0.9, 0.999),
        beta3: float = 0.95,
        eps: float = 1e-8,
        weight_decay: float = 0.0,
        correction_cap: float = 0.1,
        probe_interval: int = 10,
        third_order_off: bool = False,
        backend: str = "torch",
    ) -> None:
        if lr < 0:
            raise ValueError("learning rate must be non-negative")
        if eps < 0:
            raise ValueError("epsilon must be non-negative")
        if not 0 <= correction_cap:
            raise ValueError("correction_cap must be non-negative")
        if probe_interval < 1:
            raise ValueError("probe_interval must be at least 1")
        if not all(0 <= beta < 1 for beta in (*betas, beta3)):
            raise ValueError("all beta values must be in [0, 1)")

        defaults = {
            "lr": lr,
            "betas": betas,
            "beta3": beta3,
            "eps": eps,
            "weight_decay": weight_decay,
            "correction_cap": correction_cap,
        }
        super().__init__(params, defaults)
        self.probe_interval = probe_interval
        self.third_order_off = third_order_off
        self.backend = get_backend(backend)
        self._step_count = 0
        self._probe_count = 0
        self.diagnostics = {
            "correction_norm": 0.0,
            "correction_ratio": 0.0,
            "cap_applied": 0.0,
            "probe_ms": 0.0,
            "slice_norm": 0.0,
        }

    def _parameters(self) -> Iterator[tuple[dict[str, Any], torch.Tensor]]:
        """Yield trainable parameters together with their parameter group."""
        for group in self.param_groups:
            for parameter in group["params"]:
                if parameter.requires_grad:
                    yield group, parameter

    def state_dict(self) -> dict[str, Any]:
        """Include counters that are not part of PyTorch's parameter state."""
        optimizer_state = super().state_dict()
        optimizer_state["kenian_metadata"] = {
            "step_count": self._step_count,
            "probe_count": self._probe_count,
        }
        return optimizer_state

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        """Restore optimizer and Kenian-specific state without mutating the input."""
        optimizer_state = dict(state_dict)
        metadata = optimizer_state.pop("kenian_metadata", None)
        super().load_state_dict(optimizer_state)
        if metadata is not None:
            self._step_count = metadata["step_count"]
            self._probe_count = metadata["probe_count"]

    def needs_probe(self) -> bool:
        """Return whether the next call to ``backward_and_prepare`` probes ∇³L."""
        return not self.third_order_off and self._step_count % self.probe_interval == 0

    def backward_and_prepare(
        self,
        loss: torch.Tensor,
        probe_loss: torch.Tensor | None = None,
        scaler: Any = None,
    ) -> None:
        """Backpropagate ``loss`` and collect a third-order slice when scheduled.

        ``probe_loss`` may be a smaller full-precision batch.  When a scaler is
        supplied, it is intentionally applied only to the main loss.
        """
        if not self.needs_probe():
            self._backward(loss, scaler)
            return

        started_at = self._start_probe_timer(loss)
        parameters = [parameter for _, parameter in self._parameters()]
        if probe_loss is None:
            graph_gradients = torch.autograd.grad(
                loss,
                parameters,
                create_graph=True,
            )
            for parameter, gradient in zip(parameters, graph_gradients):
                parameter.grad = gradient.detach().clone()
        else:
            self._backward(loss, scaler)
            graph_gradients = torch.autograd.grad(
                probe_loss,
                parameters,
                create_graph=True,
            )

        self._probe_slice(parameters, graph_gradients)
        self._probe_count += 1
        self._stop_probe_timer(started_at)

    @staticmethod
    def _backward(loss: torch.Tensor, scaler: Any) -> None:
        (scaler.scale(loss) if scaler is not None else loss).backward()

    @staticmethod
    def _start_probe_timer(
        loss: torch.Tensor,
    ) -> tuple[torch.cuda.Event, torch.cuda.Event] | None:
        if not loss.is_cuda:
            return None
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        return start, end

    def _stop_probe_timer(
        self,
        timer: tuple[torch.cuda.Event, torch.cuda.Event] | None,
    ) -> None:
        if timer is None:
            return
        start, end = timer
        end.record()
        torch.cuda.synchronize()
        self.diagnostics["probe_ms"] = start.elapsed_time(end)

    def _previous_directions(
        self,
        parameters: list[torch.Tensor],
    ) -> list[torch.Tensor]:
        directions = []
        for parameter in parameters:
            direction = self.state[parameter].get("base_step")
            if direction is None:
                direction = parameter.grad
            directions.append(
                direction if direction is not None else torch.zeros_like(parameter)
            )
        return directions

    def _probe_slice(
        self,
        parameters: list[torch.Tensor],
        graph_gradients: tuple[torch.Tensor, ...],
    ) -> None:
        """Update the EMA of ``∇³L[u, u, ·]`` for the normalized base step."""
        directions = self._previous_directions(parameters)
        direction_norm_sq = sum(
            float(direction.square().sum()) for direction in directions
        )
        inverse_norm = 1.0 / max(math.sqrt(direction_norm_sq), 1e-30)

        first_contraction = (
            sum(
                (gradient * direction).sum()
                for gradient, direction in zip(graph_gradients, directions)
            )
            * inverse_norm
        )
        hessian_direction = torch.autograd.grad(
            first_contraction,
            parameters,
            create_graph=True,
        )
        second_contraction = (
            sum(
                (hessian * direction).sum()
                for hessian, direction in zip(hessian_direction, directions)
            )
            * inverse_norm
        )
        third_order_slice = torch.autograd.grad(second_contraction, parameters)

        slice_norm_sq = 0.0
        for parameter, slice_value in zip(parameters, third_order_slice):
            state = self.state[parameter]
            if "slice_ema" not in state:
                state["slice_ema"] = torch.zeros_like(parameter)
                state["slice_steps"] = 0
            beta3 = self.param_groups[0]["beta3"]
            state["slice_ema"].mul_(beta3).add_(
                slice_value.detach(),
                alpha=1.0 - beta3,
            )
            state["slice_steps"] += 1
            slice_norm_sq += float(state["slice_ema"].square().sum())
        self.diagnostics["slice_norm"] = math.sqrt(slice_norm_sq)

    @torch.no_grad()
    def step(self, closure: Any = None) -> torch.Tensor | None:
        """Apply an AdamW base update and its capped third-order correction."""
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        self._step_count += 1
        step = self._step_count
        updates = []
        base_p_norm_sq = 0.0
        base_l2_norm_sq = 0.0
        slice_p_norm_sq = 0.0

        for group, parameter in self._parameters():
            if parameter.grad is None:
                continue
            base_step, denominator = self._adamw_step(group, parameter, step)
            slice_ema = None
            if not self.third_order_off:
                slice_ema = self.state[parameter].get("slice_ema")
            partials = self.backend.partial_norms(
                base_step,
                denominator,
                slice_ema,
            )
            base_p_norm_sq += partials[0]
            base_l2_norm_sq += partials[1]
            slice_p_norm_sq += partials[2]
            updates.append((group, parameter, base_step, denominator, slice_ema))

        correction_scale = self._correction_scale(
            base_p_norm_sq,
            base_l2_norm_sq,
            slice_p_norm_sq,
        )
        for group, parameter, base_step, denominator, slice_ema in updates:
            correction = slice_ema if correction_scale else None
            self.backend.phase_b(
                parameter,
                base_step,
                denominator,
                correction,
                correction_scale,
                group["lr"],
                group["weight_decay"],
            )
            if not self.third_order_off:
                self.state[parameter]["base_step"] = (
                    base_step.detach().reshape(parameter.shape).clone()
                )
        return loss

    def _correction_scale(
        self,
        base_p_norm_sq: float,
        base_l2_norm_sq: float,
        slice_p_norm_sq: float,
    ) -> float:
        """Return the global correction scale and refresh diagnostics."""
        self.diagnostics["cap_applied"] = 0.0
        self.diagnostics["correction_ratio"] = 0.0
        self.diagnostics["correction_norm"] = 0.0
        if slice_p_norm_sq <= 0.0:
            return 0.0

        base_p_norm = math.sqrt(base_p_norm_sq)
        uncapped_norm = math.sqrt(slice_p_norm_sq) * base_l2_norm_sq
        cap = self.param_groups[0]["correction_cap"] * base_p_norm
        correction_factor = min(1.0, cap / max(uncapped_norm, 1e-30))
        if correction_factor < 1.0:
            self.diagnostics["cap_applied"] = 1.0

        scale = 0.5 * base_l2_norm_sq * correction_factor
        correction_norm = math.sqrt(slice_p_norm_sq) * scale
        self.diagnostics["correction_norm"] = correction_norm
        self.diagnostics["correction_ratio"] = correction_norm / max(base_p_norm, 1e-30)
        return scale

    def _adamw_step(
        self,
        group: dict[str, Any],
        parameter: torch.Tensor,
        step: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Update AdamW moments and return the base step and denominator."""
        state = self.state[parameter]
        beta1, beta2 = group["betas"]
        if "exp_avg" not in state:
            state["exp_avg"] = torch.zeros_like(parameter)
            state["exp_avg_sq"] = torch.zeros_like(parameter)
        bias_correction1 = 1.0 - beta1**step
        bias_correction2 = 1.0 - beta2**step
        return self.backend.phase_a_adamw(
            parameter.grad,
            state["exp_avg"],
            state["exp_avg_sq"],
            group["lr"],
            beta1,
            beta2,
            group["eps"],
            bias_correction1,
            bias_correction2,
        )
