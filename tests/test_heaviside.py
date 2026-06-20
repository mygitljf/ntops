import pytest
import torch

import ntops
from tests.skippers import skip_if_cuda_not_available


def _assert_close(output, reference, rtol=1e-3, atol=1e-3):
    assert output.shape == reference.shape
    assert output.dtype == reference.dtype
    if reference.dtype.is_floating_point:
        assert torch.allclose(output, reference, rtol=rtol, atol=atol, equal_nan=True)
    else:
        assert torch.equal(output, reference)


@skip_if_cuda_not_available
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32, torch.float64])
@pytest.mark.parametrize("shape", [(1,), (7,), (3, 5), (2, 3, 4), (8, 1)])
def test_heaviside_float_shapes(shape, dtype):
    input = torch.randn(shape, dtype=dtype, device="cuda")
    # Force some exact zeros to exercise the values branch.
    input.view(-1)[: max(1, input.numel() // 4)] = 0
    values = torch.randn(shape, dtype=dtype, device="cuda")
    output = ntops.torch.heaviside(input, values)
    reference = torch.heaviside(input, values)
    _assert_close(output, reference)


@skip_if_cuda_not_available
@pytest.mark.parametrize("dtype", [torch.int16, torch.int32, torch.int64])
def test_heaviside_integer(dtype):
    input = torch.tensor([-3, 0, 2, 0, 5], dtype=dtype, device="cuda")
    values = torch.tensor([7, 9, 1, 4, 2], dtype=dtype, device="cuda")
    output = ntops.torch.heaviside(input, values)
    reference = torch.heaviside(input, values)
    _assert_close(output, reference)


@skip_if_cuda_not_available
def test_heaviside_special_values():
    input = torch.tensor([0.0, -0.0, float("inf"), -float("inf"), float("nan")], device="cuda")
    values = torch.tensor([5.0, 6.0, 7.0, 8.0, 9.0], device="cuda")
    output = ntops.torch.heaviside(input, values)
    reference = torch.heaviside(input, values)
    _assert_close(output, reference, rtol=1e-4, atol=1e-4)


@skip_if_cuda_not_available
def test_heaviside_scalar_values_broadcast():
    input = torch.tensor([-1.0, 0.0, 1.0, 0.0], device="cuda")
    values = torch.tensor(0.5, device="cuda")
    output = ntops.torch.heaviside(input, values)
    reference = torch.heaviside(input, values)
    _assert_close(output, reference, rtol=1e-4, atol=1e-4)


@skip_if_cuda_not_available
def test_heaviside_non_contiguous_and_out():
    input = torch.randn((5, 7), dtype=torch.float32, device="cuda").t()
    values = torch.randn((5, 7), dtype=torch.float32, device="cuda").t()
    out = torch.empty_like(input)
    result = ntops.torch.heaviside(input, values, out=out)
    reference = torch.heaviside(input, values)
    assert result is out
    _assert_close(out, reference, rtol=1e-4, atol=1e-4)
