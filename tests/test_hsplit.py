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
    "shape,ios",
    [
        ((8,), 2),
        ((6,), 3),
        ((4, 8), 2),
        ((4, 9), 3),
        ((2, 8, 4), 4),
        ((10,), [2, 5]),
        ((4, 10), [3, 7]),
        ((4, 10), [2, 4, 8]),
    ],
)
def test_hsplit(shape, ios, dtype):
    if dtype.is_floating_point:
        input = torch.randn(shape, dtype=dtype, device="cuda")
    else:
        input = torch.randint(-100, 100, shape, dtype=dtype, device="cuda")
    outputs = ntops.torch.hsplit(input, ios)
    references = torch.hsplit(input, ios)
    assert len(outputs) == len(references)
    for o, r in zip(outputs, references):
        _assert_equal(o, r)
