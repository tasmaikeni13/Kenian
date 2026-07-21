"""CUDA backend checks against the PyTorch reference implementation."""

from __future__ import annotations

import sys
from importlib.util import find_spec
from pathlib import Path

import pytest
import torch
from torch.utils.cpp_extension import CUDA_HOME

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from kenian import Kenian
from kenian_backends import TorchBackend, get_backend


pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="CUDA backends require an NVIDIA GPU",
)

BACKENDS = pytest.mark.parametrize(
    "backend_name",
    [
        pytest.param(
            "triton",
            marks=pytest.mark.skipif(
                find_spec("triton") is None,
                reason="Triton is not installed",
            ),
        ),
        pytest.param(
            "cuda",
            marks=pytest.mark.skipif(
                CUDA_HOME is None,
                reason="a CUDA compiler is not available",
            ),
        ),
    ],
)
SIZES = [1, 31, 256, 1_000, 1_024, 4_097, 250_001]


def _tolerance(backend_name: str, dtype: torch.dtype) -> tuple[float, float]:
    if dtype == torch.float32:
        return 1e-5, 1e-6
    if backend_name == "triton":
        return 5e-6, 1e-9
    return 1e-10, 1e-11


def _random_state(
    size: int,
    dtype: torch.dtype,
    seed: int,
) -> tuple[torch.Tensor, ...]:
    generator = torch.Generator(device="cuda").manual_seed(seed)
    gradient = torch.randn(size, device="cuda", dtype=dtype, generator=generator)
    exp_avg = torch.randn(size, device="cuda", dtype=dtype, generator=generator) * 0.1
    exp_avg_sq = (
        torch.rand(size, device="cuda", dtype=dtype, generator=generator) * 0.5 + 1e-3
    )
    slice_ema = torch.randn(size, device="cuda", dtype=dtype, generator=generator) * 0.3
    parameter = torch.randn(size, device="cuda", dtype=dtype, generator=generator)
    return gradient, exp_avg, exp_avg_sq, slice_ema, parameter


def _assert_close(
    name: str,
    actual: torch.Tensor,
    expected: torch.Tensor,
    dtype: torch.dtype,
    backend_name: str,
) -> None:
    relative_tolerance, absolute_tolerance = _tolerance(backend_name, dtype)
    actual = actual.double()
    expected = expected.double()
    scale = expected.abs().max().item() + 1e-30
    error = (actual - expected).abs().max().item()
    assert error <= absolute_tolerance + relative_tolerance * scale, (
        f"{name} [{dtype}]: maximum error {error:.3e} at scale {scale:.3e}"
    )


@BACKENDS
def test_adamw_phase_matches_torch(backend_name: str) -> None:
    backend = get_backend(backend_name)
    reference = TorchBackend()
    arguments = (3e-4, 0.9, 0.999, 1e-8, 0.271, 0.0952)
    for dtype in (torch.float32, torch.float64):
        for size in SIZES:
            gradient, exp_avg, exp_avg_sq, _, _ = _random_state(size, dtype, 100 + size)
            actual_avg, actual_avg_sq = exp_avg.clone(), exp_avg_sq.clone()
            expected_avg, expected_avg_sq = exp_avg.clone(), exp_avg_sq.clone()
            expected_step, expected_denom = reference.phase_a_adamw(
                gradient.clone(), expected_avg, expected_avg_sq, *arguments
            )
            actual_step, actual_denom = backend.phase_a_adamw(
                gradient.clone(), actual_avg, actual_avg_sq, *arguments
            )
            _assert_close("base step", actual_step, expected_step, dtype, backend_name)
            _assert_close(
                "denominator", actual_denom, expected_denom, dtype, backend_name
            )
            _assert_close("first moment", actual_avg, expected_avg, dtype, backend_name)
            _assert_close(
                "second moment", actual_avg_sq, expected_avg_sq, dtype, backend_name
            )


@BACKENDS
def test_norm_reductions_match_torch(backend_name: str) -> None:
    backend = get_backend(backend_name)
    reference = TorchBackend()
    for dtype in (torch.float32, torch.float64):
        for size in SIZES:
            _, _, _, slice_ema, _ = _random_state(size, dtype, 200 + size)
            base_step = torch.randn(size, device="cuda", dtype=dtype)
            denominator = torch.rand(size, device="cuda", dtype=dtype) + 1e-2
            for slice_value in (None, slice_ema):
                expected = reference.partial_norms(base_step, denominator, slice_value)
                actual = backend.partial_norms(base_step, denominator, slice_value)
                tolerance = (
                    2e-4
                    if dtype == torch.float32
                    else (5e-6 if backend_name == "triton" else 1e-9)
                )
                for expected_value, actual_value in zip(expected, actual):
                    relative_error = abs(expected_value - actual_value) / (
                        abs(expected_value) + 1e-30
                    )
                    assert relative_error <= tolerance


@BACKENDS
def test_parameter_update_matches_torch(backend_name: str) -> None:
    backend = get_backend(backend_name)
    reference = TorchBackend()
    for dtype in (torch.float32, torch.float64):
        for size in SIZES:
            _, _, _, slice_ema, parameter = _random_state(size, dtype, 300 + size)
            base_step = torch.randn(size, device="cuda", dtype=dtype) * 1e-3
            denominator = torch.rand(size, device="cuda", dtype=dtype) + 1e-2
            for slice_value, correction_scale in ((None, 0.0), (slice_ema, 1.7e-3)):
                expected_parameter = parameter.clone()
                actual_parameter = parameter.clone()
                expected_update = reference.phase_b(
                    expected_parameter,
                    base_step.clone(),
                    denominator,
                    slice_value,
                    correction_scale,
                    3e-4,
                    0.05,
                )
                actual_update = backend.phase_b(
                    actual_parameter,
                    base_step.clone(),
                    denominator,
                    slice_value,
                    correction_scale,
                    3e-4,
                    0.05,
                )
                _assert_close(
                    "parameter",
                    actual_parameter,
                    expected_parameter,
                    dtype,
                    backend_name,
                )
                _assert_close(
                    "update", actual_update, expected_update, dtype, backend_name
                )


@BACKENDS
def test_complete_optimizer_matches_torch(backend_name: str) -> None:
    """The complete fused update should track the reference implementation."""
    torch.manual_seed(7)
    for dtype in (torch.float32, torch.float64):
        reference_model = torch.nn.Sequential(
            torch.nn.Linear(32, 48),
            torch.nn.GELU(),
            torch.nn.Linear(48, 10),
        ).to("cuda", dtype)
        backend_model = torch.nn.Sequential(
            torch.nn.Linear(32, 48),
            torch.nn.GELU(),
            torch.nn.Linear(48, 10),
        ).to("cuda", dtype)
        backend_model.load_state_dict(reference_model.state_dict())
        inputs = torch.randn(64, 32, device="cuda", dtype=dtype)
        targets = torch.randint(0, 10, (64,), device="cuda")
        reference_optimizer = Kenian(
            reference_model.parameters(),
            lr=1e-3,
            correction_cap=0.5,
            probe_interval=3,
            weight_decay=0.05,
            beta3=0.9,
            backend="torch",
        )
        backend_optimizer = Kenian(
            backend_model.parameters(),
            lr=1e-3,
            correction_cap=0.5,
            probe_interval=3,
            weight_decay=0.05,
            beta3=0.9,
            backend=backend_name,
        )
        for _ in range(40):
            reference_optimizer.zero_grad(set_to_none=True)
            reference_loss = torch.nn.functional.cross_entropy(
                reference_model(inputs), targets
            )
            reference_optimizer.backward_and_prepare(reference_loss)
            reference_optimizer.step()

            backend_optimizer.zero_grad(set_to_none=True)
            backend_loss = torch.nn.functional.cross_entropy(
                backend_model(inputs), targets
            )
            backend_optimizer.backward_and_prepare(backend_loss)
            backend_optimizer.step()

        tolerance = (
            3e-4
            if dtype == torch.float32
            else (1e-4 if backend_name == "triton" else 1e-9)
        )
        maximum_error = max(
            (reference_parameter - backend_parameter).abs().max().item()
            for reference_parameter, backend_parameter in zip(
                reference_model.parameters(),
                backend_model.parameters(),
            )
        )
        assert maximum_error <= tolerance
