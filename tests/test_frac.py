import pytest
import torch

import ntops
from tests.skippers import skip_if_cuda_not_available


def _reference(input):
    if not input.dtype.is_floating_point:
        input = input.to(torch.float32)
    return torch.frac(input)


def _assert_close(output, reference, rtol=2e-3, atol=2e-3):
    assert output.shape == reference.shape
    assert output.dtype == reference.dtype
    assert torch.allclose(output, reference, rtol=rtol, atol=atol, equal_nan=True)


@skip_if_cuda_not_available
@pytest.mark.parametrize("dtype", [torch.float16, torch.float32, torch.float64, torch.bfloat16])
@pytest.mark.parametrize("shape", [(0,), (1,), (7,), (3, 5), (2, 3, 4), (1, 4, 1)])
def test_frac_float_shapes(shape, dtype):
    input = torch.randn(shape, dtype=dtype, device="cuda") * 8 - 4
    output = ntops.torch.frac(input)
    reference = _reference(input)
    _assert_close(output, reference)


@skip_if_cuda_not_available
@pytest.mark.parametrize("dtype", [torch.int16, torch.int32, torch.int64, torch.bool])
@pytest.mark.parametrize("shape", [(3,), (2, 3)])
def test_frac_integer_promotes(dtype, shape):
    if dtype == torch.bool:
        input = torch.randint(0, 2, shape, dtype=dtype, device="cuda")
    else:
        input = torch.randint(-8, 8, shape, dtype=dtype, device="cuda")
    output = ntops.torch.frac(input)
    reference = _reference(input)
    _assert_close(output, reference)


@skip_if_cuda_not_available
def test_frac_special_values():
    input = torch.tensor(
        [0.0, -0.0, float("inf"), -float("inf"), float("nan"), -1.75, 1.75],
        dtype=torch.float32,
        device="cuda",
    )
    output = ntops.torch.frac(input)
    reference = torch.frac(input)
    _assert_close(output, reference)


@skip_if_cuda_not_available
def test_frac_non_contiguous_and_out():
    input = (torch.randn((4, 5), dtype=torch.float32, device="cuda") * 4).t()
    out = torch.empty_like(input)
    result = ntops.torch.frac(input, out=out)
    reference = torch.frac(input)
    assert result is out
    _assert_close(out, reference)


@skip_if_cuda_not_available
def test_frac_3d_permute_non_contiguous_and_out():
    input = (torch.randn((2, 3, 4), dtype=torch.float32, device="cuda") * 4).permute(2, 0, 1)
    out = torch.empty_like(input)
    result = ntops.torch.frac(input, out=out)
    reference = torch.frac(input)
    assert result is out
    _assert_close(out, reference)


@skip_if_cuda_not_available
def test_frac_scalar():
    input = torch.tensor(-1.25, dtype=torch.float32, device="cuda")
    output = ntops.torch.frac(input)
    reference = torch.frac(input)
    _assert_close(output, reference)


@skip_if_cuda_not_available
def test_frac_resizes_out_like_torch():
    input = torch.randn((2, 3), dtype=torch.float32, device="cuda")
    out = torch.empty((1,), dtype=torch.float32, device="cuda")
    with pytest.warns(UserWarning):
        result = ntops.torch.frac(input, out=out)
    reference = torch.empty((1,), dtype=torch.float32, device="cuda")
    with pytest.warns(UserWarning):
        torch.frac(input, out=reference)
    assert result is out
    _assert_close(out, reference)


@skip_if_cuda_not_available
def test_frac_rejects_integer_out():
    input = torch.randn((4,), dtype=torch.float32, device="cuda")
    out = torch.empty((4,), dtype=torch.int32, device="cuda")
    with pytest.raises(RuntimeError):
        ntops.torch.frac(input, out=out)


@skip_if_cuda_not_available
def test_frac_large_magnitude_float64():
    input = torch.tensor(
        [1e15 + 0.5, 123456789.25, -987654321.75, 2.5, -0.5],
        dtype=torch.float64,
        device="cuda",
    )
    output = ntops.torch.frac(input)
    reference = torch.frac(input)
    _assert_close(output, reference, rtol=1e-9, atol=1e-9)


@skip_if_cuda_not_available
def test_frac_large_magnitude_float32():
    input = torch.tensor(
        [8388610.5, 1000000.25, -2000000.75, 0.5],
        dtype=torch.float32,
        device="cuda",
    )
    output = ntops.torch.frac(input)
    reference = torch.frac(input)
    _assert_close(output, reference, rtol=1e-5, atol=1e-5)
