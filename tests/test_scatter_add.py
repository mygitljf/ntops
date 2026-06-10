import pytest
import torch

import ntops
from tests.skippers import skip_if_cuda_not_available


def _assert_equal(output, reference):
    assert output.shape == reference.shape
    assert output.dtype == reference.dtype
    if reference.dtype.is_floating_point:
        assert torch.allclose(output, reference, rtol=2e-3, atol=2e-3, equal_nan=True)
    else:
        assert torch.equal(output, reference)


def _make_case(shape, dim, dtype):
    input = torch.randn(shape, dtype=dtype, device="cuda")
    src = torch.randn(shape, dtype=dtype, device="cuda")
    if input.numel() == 0:
        index = torch.empty(shape, dtype=torch.long, device="cuda")
    else:
        index = torch.randint(0, shape[dim], shape, dtype=torch.long, device="cuda")
    return input, dim, index, src


@skip_if_cuda_not_available
@pytest.mark.parametrize("dtype", [torch.float16, torch.float32, torch.float64])
@pytest.mark.parametrize(
    "shape, dim",
    [((0,), 0), ((1,), 0), ((7,), 0), ((3, 5), 0), ((3, 5), 1), ((2, 3, 4), 2)],
)
def test_scatter_add_shapes(shape, dim, dtype):
    input, dim, index, src = _make_case(shape, dim, dtype)
    output = ntops.torch.scatter_add(input, dim, index, src)
    reference = torch.scatter_add(input, dim, index, src)
    _assert_equal(output, reference)


@skip_if_cuda_not_available
@pytest.mark.parametrize("dtype", [torch.float32, torch.int32, torch.int64])
def test_scatter_add_duplicate_indices(dtype):
    input = torch.zeros((2, 4), dtype=dtype, device="cuda")
    index = torch.tensor([[0, 1, 1, 3], [2, 2, 0, 1]], dtype=torch.long, device="cuda")
    if dtype.is_floating_point:
        src = torch.arange(8, dtype=dtype, device="cuda").reshape(2, 4)
    else:
        src = torch.arange(8, dtype=dtype, device="cuda").reshape(2, 4)
    output = ntops.torch.scatter_add(input, 1, index, src)
    reference = torch.scatter_add(input, 1, index, src)
    _assert_equal(output, reference)


@skip_if_cuda_not_available
def test_scatter_add_negative_dim_and_out():
    input, _, index, src = _make_case((2, 3, 4), -1, torch.float32)
    out = torch.empty_like(input)
    result = ntops.torch.scatter_add(input, -1, index, src, out=out)
    reference = torch.scatter_add(input, -1, index, src)
    assert result is out
    _assert_equal(out, reference)


@skip_if_cuda_not_available
def test_scatter_add_non_contiguous_inputs():
    input = torch.randn((4, 3), dtype=torch.float32, device="cuda").t()
    src = torch.randn((4, 3), dtype=torch.float32, device="cuda").t()
    index = torch.randint(0, input.shape[1], input.shape, dtype=torch.long, device="cuda")
    output = ntops.torch.scatter_add(input, 1, index, src)
    reference = torch.scatter_add(input, 1, index, src)
    _assert_equal(output, reference)


@skip_if_cuda_not_available
def test_scatter_add_resizes_out_like_torch():
    input, dim, index, src = _make_case((2, 3), 1, torch.float32)
    out = torch.empty((1,), dtype=torch.float32, device="cuda")
    with pytest.warns(UserWarning):
        result = ntops.torch.scatter_add(input, dim, index, src, out=out)
    reference = torch.empty((1,), dtype=torch.float32, device="cuda")
    with pytest.warns(UserWarning):
        torch.scatter_add(input, dim, index, src, out=reference)
    assert result is out
    _assert_equal(out, reference)


@skip_if_cuda_not_available
def test_scatter_add_rejects_index_dtype():
    input = torch.zeros((2, 3), dtype=torch.float32, device="cuda")
    src = torch.ones_like(input)
    index = torch.zeros((2, 3), dtype=torch.int32, device="cuda")
    with pytest.raises(RuntimeError):
        ntops.torch.scatter_add(input, 1, index, src)


@skip_if_cuda_not_available
def test_scatter_add_rejects_bad_dim():
    input = torch.zeros((2, 3), dtype=torch.float32, device="cuda")
    src = torch.ones_like(input)
    index = torch.zeros((2, 3), dtype=torch.long, device="cuda")
    with pytest.raises(IndexError):
        ntops.torch.scatter_add(input, 2, index, src)
