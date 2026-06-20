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
@pytest.mark.parametrize("n", [4, 5, 8, 1, 0])
@pytest.mark.parametrize("r", [2, 3])
def test_combinations(n, r, dtype):
    if dtype.is_floating_point:
        input = torch.randn(n, dtype=dtype, device="cuda")
    else:
        input = torch.arange(n, dtype=dtype, device="cuda")
    output = ntops.torch.combinations(input, r=r)
    reference = torch.combinations(input, r=r)
    _assert_equal(output, reference)


@skip_if_cuda_not_available
@pytest.mark.parametrize("r", [2, 3])
def test_combinations_with_replacement(r):
    input = torch.arange(5, dtype=torch.int64, device="cuda")
    output = ntops.torch.combinations(input, r=r, with_replacement=True)
    reference = torch.combinations(input, r=r, with_replacement=True)
    _assert_equal(output, reference)


@skip_if_cuda_not_available
def test_combinations_default_r():
    input = torch.arange(6, dtype=torch.int64, device="cuda")
    output = ntops.torch.combinations(input)
    reference = torch.combinations(input)
    _assert_equal(output, reference)
