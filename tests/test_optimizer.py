"""Unit tests for the Kenian optimizer on the PyTorch reference backend."""

from __future__ import annotations

import math
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from kenian import Kenian


torch.set_default_dtype(torch.float64)


def _tiny_problem(
    seed: int = 0,
) -> tuple[torch.Tensor, torch.Tensor, torch.nn.Parameter]:
    torch.manual_seed(seed)
    matrix = torch.randn(12, 8)
    target = torch.randn(12)
    weights = torch.nn.Parameter(torch.randn(8))
    return matrix, target, weights


def test_matches_adamw_when_third_order_is_disabled() -> None:
    """The ablation mode must be step-for-step equivalent to AdamW."""
    for weight_decay in (0.0, 0.05):
        matrix, target, kenian_weights = _tiny_problem(1)
        adamw_weights = torch.nn.Parameter(kenian_weights.detach().clone())
        kenian = Kenian(
            [kenian_weights],
            lr=1e-2,
            weight_decay=weight_decay,
            third_order_off=True,
        )
        adamw = torch.optim.AdamW(
            [adamw_weights],
            lr=1e-2,
            weight_decay=weight_decay,
        )
        for _ in range(50):
            kenian.zero_grad(set_to_none=True)
            kenian_loss = ((matrix @ kenian_weights - target) ** 2).mean()
            kenian.backward_and_prepare(kenian_loss)
            kenian.step()

            adamw.zero_grad(set_to_none=True)
            adamw_loss = ((matrix @ adamw_weights - target) ** 2).mean()
            adamw_loss.backward()
            adamw.step()

        max_difference = (kenian_weights - adamw_weights).abs().max().item()
        assert max_difference < 1e-10


def test_correction_obeys_the_global_cap() -> None:
    """The correction P-norm must remain within the configured fraction."""
    torch.manual_seed(2)
    model = torch.nn.Sequential(
        torch.nn.Linear(6, 10),
        torch.nn.GELU(),
        torch.nn.Linear(10, 3),
    )
    inputs = torch.randn(16, 6)
    targets = torch.randint(0, 3, (16,))
    correction_cap = 0.5
    optimizer = Kenian(
        model.parameters(),
        lr=5e-3,
        correction_cap=correction_cap,
        probe_interval=1,
        beta3=0.0,
    )
    max_ratio = 0.0
    for _ in range(30):
        optimizer.zero_grad(set_to_none=True)
        loss = torch.nn.functional.cross_entropy(model(inputs), targets)
        optimizer.backward_and_prepare(loss)
        optimizer.step()
        max_ratio = max(max_ratio, optimizer.diagnostics["correction_ratio"])
    assert max_ratio <= correction_cap + 1e-6


def test_probe_matches_an_exact_third_order_slice() -> None:
    """A one-sample EMA equals an independently calculated slice."""
    torch.manual_seed(3)
    model = torch.nn.Sequential(
        torch.nn.Linear(5, 7),
        torch.nn.GELU(),
        torch.nn.Linear(7, 4),
    )
    inputs = torch.randn(8, 5)
    targets = torch.randint(0, 4, (8,))
    parameters = list(model.parameters())
    optimizer = Kenian(parameters, lr=1e-2, probe_interval=1, beta3=0.0)

    optimizer.zero_grad(set_to_none=True)
    torch.nn.functional.cross_entropy(model(inputs), targets).backward()
    optimizer.step()

    directions = [optimizer.state[parameter]["base_step"] for parameter in parameters]
    inverse_norm = 1.0 / math.sqrt(
        sum(float(direction.square().sum()) for direction in directions)
    )
    unit_directions = [direction * inverse_norm for direction in directions]
    loss = torch.nn.functional.cross_entropy(model(inputs), targets)
    gradients = torch.autograd.grad(loss, parameters, create_graph=True)
    hessian_vector = torch.autograd.grad(
        sum(
            (gradient * direction).sum()
            for gradient, direction in zip(gradients, unit_directions)
        ),
        parameters,
        create_graph=True,
    )
    expected_slice = torch.autograd.grad(
        sum(
            (hessian * direction).sum()
            for hessian, direction in zip(hessian_vector, unit_directions)
        ),
        parameters,
    )

    optimizer.zero_grad(set_to_none=True)
    probe_loss = torch.nn.functional.cross_entropy(model(inputs), targets)
    optimizer.backward_and_prepare(probe_loss)
    max_error = max(
        (optimizer.state[parameter]["slice_ema"] - expected).abs().max().item()
        for parameter, expected in zip(parameters, expected_slice)
    )
    scale = max(expected.abs().max().item() for expected in expected_slice)
    assert max_error / scale < 1e-8


def test_reduces_a_convex_loss() -> None:
    """The complete optimizer should make steady progress on a simple objective."""
    matrix, target, weights = _tiny_problem(4)
    optimizer = Kenian([weights], lr=5e-3, probe_interval=2, correction_cap=0.5)
    initial_loss = None
    for _ in range(200):
        optimizer.zero_grad(set_to_none=True)
        loss = ((matrix @ weights - target) ** 2).mean()
        optimizer.backward_and_prepare(loss)
        optimizer.step()
        initial_loss = loss.item() if initial_loss is None else initial_loss
    final_loss = ((matrix @ weights - target) ** 2).mean().item()
    assert initial_loss is not None
    assert math.isfinite(final_loss)
    assert final_loss < 0.5 * initial_loss
