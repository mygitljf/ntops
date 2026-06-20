import pytest
import torch
import torch.nn.functional as F

import ntops
from tests.skippers import skip_if_cuda_not_available


def _assert_equal(output, reference):
    assert output.shape == reference.shape
    assert output.dtype == reference.dtype
    assert torch.equal(output, reference)


@skip_if_cuda_not_available
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.float64, torch.int32])
@pytest.mark.parametrize(
    "shape,factor",
    [
        ((1, 1, 8, 8), 2),
        ((2, 3, 8, 8), 2),
        ((1, 1, 9, 9), 3),
        ((4, 2, 16, 16), 4),
        ((2, 3, 12, 8), 2),
    ],
)
def test_pixel_unshuffle(shape, factor, dtype):
    if dtype.is_floating_point:
        input = torch.randn(shape, dtype=dtype, device="cuda")
    else:
        input = torch.randint(-100, 100, shape, dtype=dtype, device="cuda")
    output = ntops.torch.pixel_unshuffle(input, factor)
    reference = F.pixel_unshuffle(input, factor)
    _assert_equal(output, reference)


@skip_if_cuda_not_available
def test_pixel_unshuffle_non_contiguous():
    input = torch.randn((2, 3, 8, 16), dtype=torch.float32, device="cuda").transpose(-1, -2)
    output = ntops.torch.pixel_unshuffle(input, 2)
    reference = F.pixel_unshuffle(input, 2)
    _assert_equal(output, reference)
