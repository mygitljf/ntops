import pytest
import torch

import ntops
from tests.skippers import skip_if_cuda_not_available


def _assert_close(output, reference):
    assert output.shape == reference.shape
    assert output.dtype == reference.dtype
    assert torch.allclose(output, reference, rtol=0, atol=0, equal_nan=True)


def _torch_nextafter_reference(input, other):
    if input.device.type == "cuda" and input.dtype == torch.float16:
        return torch.nextafter(input.cpu(), other.cpu()).to(input.device)
    return torch.nextafter(input, other)


@skip_if_cuda_not_available
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32, torch.float64])
@pytest.mark.parametrize(
    "input_shape, other_shape",
    [((0,), (0,)), ((9,), (9,)), ((4, 1), (1, 7)), ((2, 3, 4), (1, 3, 1)), ((4, 1), (3,))],
)
def test_nextafter_float_shapes(dtype, input_shape, other_shape):
    input = torch.randn(input_shape, dtype=dtype, device="cuda")
    other = torch.randn(other_shape, dtype=dtype, device="cuda")
    output = ntops.torch.nextafter(input, other)
    reference = _torch_nextafter_reference(input, other)
    _assert_close(output, reference)


@skip_if_cuda_not_available
@pytest.mark.parametrize("dtype", [torch.bool, torch.int16, torch.int32, torch.int64])
def test_nextafter_rejects_integer_inputs(dtype):
    input = torch.tensor([0, 1, -2], dtype=dtype, device="cuda")
    other = torch.tensor([1, 2, -3], dtype=dtype, device="cuda")
    with pytest.raises(NotImplementedError):
        ntops.torch.nextafter(input, other)


@skip_if_cuda_not_available
def test_nextafter_special_values():
    input = torch.tensor([0.0, -0.0, 1.0, -1.0, float("inf"), -float("inf"), float("nan")], device="cuda")
    other = torch.tensor([1.0, -1.0, 2.0, -2.0, 0.0, 0.0, 1.0], device="cuda")
    output = ntops.torch.nextafter(input, other)
    reference = _torch_nextafter_reference(input, other)
    _assert_close(output, reference)


@skip_if_cuda_not_available
def test_nextafter_non_contiguous_and_out():
    input = torch.randn((5, 7), dtype=torch.float32, device="cuda").t()
    other = torch.randn((5, 7), dtype=torch.float32, device="cuda").t()
    out = torch.empty_like(input)
    result = ntops.torch.nextafter(input, other, out=out)
    reference = _torch_nextafter_reference(input, other)
    assert result is out
    _assert_close(out, reference)


@skip_if_cuda_not_available
def test_nextafter_3d_permute_non_contiguous_and_out():
    input = torch.randn((3, 5, 7), dtype=torch.float32, device="cuda").permute(2, 0, 1)
    other = torch.randn((3, 5, 7), dtype=torch.float32, device="cuda").permute(2, 0, 1)
    out = torch.empty_like(input)
    result = ntops.torch.nextafter(input, other, out=out)
    reference = _torch_nextafter_reference(input, other)
    assert result is out
    _assert_close(out, reference)


@skip_if_cuda_not_available
def test_nextafter_scalar():
    input = torch.tensor(1.0, dtype=torch.float32, device="cuda")
    other = torch.tensor(2.0, dtype=torch.float32, device="cuda")
    output = ntops.torch.nextafter(input, other)
    reference = _torch_nextafter_reference(input, other)
    _assert_close(output, reference)


@skip_if_cuda_not_available
def test_nextafter_resizes_out_like_torch():
    input = torch.randn((2, 3), dtype=torch.float32, device="cuda")
    other = torch.randn((2, 3), dtype=torch.float32, device="cuda")
    out = torch.empty((1,), dtype=torch.float32, device="cuda")
    with pytest.warns(UserWarning):
        result = ntops.torch.nextafter(input, other, out=out)
    reference = _torch_nextafter_reference(input, other)
    assert result is out
    _assert_close(out, reference)


@skip_if_cuda_not_available
def test_nextafter_rejects_integer_out_for_float_result():
    input = torch.randn((2,), dtype=torch.float32, device="cuda")
    other = torch.randn((2,), dtype=torch.float32, device="cuda")
    out = torch.empty((2,), dtype=torch.int32, device="cuda")
    with pytest.raises(RuntimeError):
        ntops.torch.nextafter(input, other, out=out)
