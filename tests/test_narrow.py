import pytest
import torch

import ntops
from tests.skippers import skip_if_cuda_not_available


def _assert_equal(output, reference):
    assert output.shape == reference.shape
    assert output.dtype == reference.dtype
    assert torch.equal(output, reference)


@skip_if_cuda_not_available
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.int64])
@pytest.mark.parametrize(
    "shape,dim,start,length",
    [
        ((4, 6), 1, 1, 3),
        ((4, 6), 0, 0, 2),
        ((8,), 0, 2, 4),
        ((2, 3, 4), 2, 1, 2),
        ((4, 6), -1, 2, 3),
        ((6, 6), 1, 0, 6),
    ],
)
def test_narrow(shape, dim, start, length, dtype):
    if dtype.is_floating_point:
        input = torch.randn(shape, dtype=dtype, device="cuda")
    else:
        input = torch.randint(-100, 100, shape, dtype=dtype, device="cuda")
    output = ntops.torch.narrow(input, dim, start, length)
    reference = torch.narrow(input, dim, start, length)
    _assert_equal(output, reference)


@skip_if_cuda_not_available
def test_narrow_non_contiguous():
    input = torch.randn((5, 7), dtype=torch.float32, device="cuda").t()
    output = ntops.torch.narrow(input, 0, 1, 3)
    reference = torch.narrow(input, 0, 1, 3)
    _assert_equal(output, reference)
