import pytest
import torch

import ntops
from tests.skippers import skip_if_cuda_not_available


def _assert_equal(output, reference):
    assert output.shape == reference.shape
    assert output.dtype == reference.dtype
    assert torch.equal(output, reference)


@skip_if_cuda_not_available
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.int32, torch.int64])
@pytest.mark.parametrize(
    "shape,dims",
    [
        ((7,), [0]),
        ((3, 5), [0]),
        ((3, 5), [1]),
        ((3, 5), [0, 1]),
        ((2, 3, 4), [0, 2]),
        ((2, 3, 4), [1]),
        ((4, 4), [-1]),
    ],
)
def test_flip(shape, dims, dtype):
    if dtype.is_floating_point:
        input = torch.randn(shape, dtype=dtype, device="cuda")
    else:
        input = torch.randint(-100, 100, shape, dtype=dtype, device="cuda")
    output = ntops.torch.flip(input, dims)
    reference = torch.flip(input, dims)
    _assert_equal(output, reference)


@skip_if_cuda_not_available
def test_flip_non_contiguous():
    input = torch.randn((5, 7), dtype=torch.float32, device="cuda").t()
    output = ntops.torch.flip(input, [0])
    reference = torch.flip(input, [0])
    _assert_equal(output, reference)


@skip_if_cuda_not_available
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.int32])
@pytest.mark.parametrize("shape", [(3, 5), (2, 3, 4), (4, 4, 4, 4)])
def test_fliplr(shape, dtype):
    if dtype.is_floating_point:
        input = torch.randn(shape, dtype=dtype, device="cuda")
    else:
        input = torch.randint(-100, 100, shape, dtype=dtype, device="cuda")
    output = ntops.torch.fliplr(input)
    reference = torch.fliplr(input)
    _assert_equal(output, reference)
