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
@pytest.mark.parametrize("shape", [(3, 5), (2, 3, 4), (4, 4, 4, 4), (16, 64)])
def test_fliplr(shape, dtype):
    if dtype.is_floating_point:
        input = torch.randn(shape, dtype=dtype, device="cuda")
    else:
        input = torch.randint(-100, 100, shape, dtype=dtype, device="cuda")

    output = ntops.torch.fliplr(input)
    reference = torch.fliplr(input)

    _assert_equal(output, reference)


@skip_if_cuda_not_available
def test_fliplr_non_contiguous():
    input = torch.randn((5, 7), dtype=torch.float32, device="cuda").t()
    output = ntops.torch.fliplr(input)
    reference = torch.fliplr(input)

    _assert_equal(output, reference)


@skip_if_cuda_not_available
def test_fliplr_requires_rank_at_least_two():
    input = torch.randn((7,), dtype=torch.float32, device="cuda")

    with pytest.raises(RuntimeError, match="at least 2 dimensions"):
        ntops.torch.fliplr(input)
