import pytest
import torch

import ntops
from tests.skippers import skip_if_cuda_not_available


def _assert_equal(output, reference):
    assert output.shape == reference.shape
    assert output.dtype == reference.dtype
    assert torch.equal(output, reference)


@skip_if_cuda_not_available
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.int32, torch.int64, torch.bool])
@pytest.mark.parametrize("shape", [(8, 16), (7,), (4, 3, 6), (32, 32)])
def test_count_nonzero_all(shape, dtype):
    if dtype == torch.bool:
        input = torch.randint(0, 2, shape, device="cuda").bool()
    elif dtype.is_floating_point:
        input = (torch.randn(shape, device="cuda") > 0).to(dtype)
    else:
        input = torch.randint(0, 3, shape, dtype=dtype, device="cuda")
    output = ntops.torch.count_nonzero(input)
    reference = torch.count_nonzero(input)
    _assert_equal(output, reference)


@skip_if_cuda_not_available
@pytest.mark.parametrize("dim", [0, 1, -1, (0, 1)])
def test_count_nonzero_dim(dim):
    input = torch.randint(0, 2, (8, 16, 4), dtype=torch.int32, device="cuda")
    output = ntops.torch.count_nonzero(input, dim=dim)
    reference = torch.count_nonzero(input, dim=dim)
    _assert_equal(output, reference)


@skip_if_cuda_not_available
def test_count_nonzero_non_contiguous():
    input = torch.randint(0, 2, (5, 7), dtype=torch.int32, device="cuda").t()
    output = ntops.torch.count_nonzero(input, dim=0)
    reference = torch.count_nonzero(input, dim=0)
    _assert_equal(output, reference)
