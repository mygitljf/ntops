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
    "shape,dim,start,end,step",
    [
        ((4, 4), 1, 1, 3, 1),
        ((4, 4), 0, 0, 2, 1),
        ((8, 6), 1, 0, 6, 2),
        ((2, 3, 4), 2, 1, 4, 1),
        ((10,), 0, 2, 8, 2),
        ((6, 6), 1, None, None, 1),
    ],
)
def test_slice_scatter(shape, dim, start, end, step, dtype):
    if dtype.is_floating_point:
        input = torch.randn(shape, dtype=dtype, device="cuda")
    else:
        input = torch.randint(-100, 100, shape, dtype=dtype, device="cuda")
    indices = torch.arange(
        0 if start is None else start,
        input.size(dim) if end is None else end,
        step,
        device="cuda",
    )
    src = input.index_select(dim, indices).clone().add_(1)
    output = ntops.torch.slice_scatter(input, src, dim=dim, start=start, end=end, step=step)
    reference = torch.slice_scatter(input, src, dim=dim, start=start, end=end, step=step)
    _assert_equal(output, reference)
