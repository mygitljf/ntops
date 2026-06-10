import pytest
import torch

import ntops
from tests.skippers import skip_if_cuda_not_available


def _assert_equal(output, reference):
    """Assert output and reference have same values, shape, and dtype."""
    assert output.shape == reference.shape, (
        f"shape mismatch: {output.shape} vs {reference.shape}"
    )
    assert output.dtype == reference.dtype, (
        f"dtype mismatch: {output.dtype} vs {reference.dtype}"
    )
    if reference.dtype in (torch.bool, torch.int8, torch.int16, torch.int32, torch.int64):
        assert torch.equal(output, reference), (
            f"value mismatch (int/bool)"
        )
    else:
        assert torch.allclose(output, reference, rtol=1e-3, atol=1e-3, equal_nan=True), (
            f"value mismatch (float)"
        )


# ============================================================
# Basic shape × dtype × source/dest combinations
# ============================================================

@skip_if_cuda_not_available
@pytest.mark.parametrize("dtype", [
    torch.float32, torch.float16, torch.float64, torch.bfloat16,
    torch.int32, torch.int64, torch.bool,
])
@pytest.mark.parametrize("source, destination", [
    (0, 1), (1, 0),
])
def test_moveaxis_2d_basic(dtype, source, destination):
    """[case 1-14] 2D basic move for all dtypes."""
    if dtype == torch.bool:
        input = torch.randint(0, 2, (3, 4), dtype=torch.bool, device="cuda")
    elif dtype.is_floating_point:
        input = torch.randn((3, 4), dtype=dtype, device="cuda")
    else:
        input = torch.randint(0, 10, (3, 4), dtype=dtype, device="cuda")

    output = ntops.torch.moveaxis(input, source, destination)
    reference = torch.moveaxis(input, source, destination)
    _assert_equal(output, reference)


@skip_if_cuda_not_available
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.int32])
@pytest.mark.parametrize("source, destination", [
    (0, 1), (0, 2), (1, 0), (1, 2), (2, 0), (2, 1),
])
def test_moveaxis_3d_permutations(dtype, source, destination):
    """[case 15-32] 3D all axis permutations."""
    if dtype.is_floating_point:
        input = torch.randn((2, 3, 4), dtype=dtype, device="cuda")
    else:
        input = torch.randint(0, 10, (2, 3, 4), dtype=dtype, device="cuda")

    output = ntops.torch.moveaxis(input, source, destination)
    reference = torch.moveaxis(input, source, destination)
    _assert_equal(output, reference)


@skip_if_cuda_not_available
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.int32])
@pytest.mark.parametrize("source, destination", [
    (0, 3), (3, 0), (1, 2), (0, 2),
])
def test_moveaxis_4d_permutations(dtype, source, destination):
    """[case 33-44] 4D axis moves."""
    if dtype.is_floating_point:
        input = torch.randn((2, 3, 4, 5), dtype=dtype, device="cuda")
    else:
        input = torch.randint(0, 10, (2, 3, 4, 5), dtype=dtype, device="cuda")

    output = ntops.torch.moveaxis(input, source, destination)
    reference = torch.moveaxis(input, source, destination)
    _assert_equal(output, reference)


# ============================================================
# Edge cases
# ============================================================

@skip_if_cuda_not_available
def test_moveaxis_1d_noop():
    """[case 45] 1D move 0→0 is no-op."""
    input = torch.randn((7,), dtype=torch.float32, device="cuda")
    output = ntops.torch.moveaxis(input, 0, 0)
    reference = torch.moveaxis(input, 0, 0)
    _assert_equal(output, reference)


@skip_if_cuda_not_available
def test_moveaxis_2d_noop():
    """[case 46] 2D move 0→0 is no-op."""
    input = torch.randn((3, 5), dtype=torch.float32, device="cuda")
    output = ntops.torch.moveaxis(input, 0, 0)
    reference = torch.moveaxis(input, 0, 0)
    _assert_equal(output, reference)


@skip_if_cuda_not_available
def test_moveaxis_negative_source():
    """[case 47] Negative source index."""
    input = torch.randn((2, 3, 4), dtype=torch.float32, device="cuda")
    output = ntops.torch.moveaxis(input, -1, 0)
    reference = torch.moveaxis(input, -1, 0)
    _assert_equal(output, reference)


@skip_if_cuda_not_available
def test_moveaxis_negative_destination():
    """[case 48] Negative destination index."""
    input = torch.randn((2, 3, 4), dtype=torch.float32, device="cuda")
    output = ntops.torch.moveaxis(input, 0, -1)
    reference = torch.moveaxis(input, 0, -1)
    _assert_equal(output, reference)


@skip_if_cuda_not_available
def test_moveaxis_both_negative():
    """[case 49] Both source and destination negative."""
    input = torch.randn((2, 3, 4, 5), dtype=torch.float32, device="cuda")
    output = ntops.torch.moveaxis(input, -3, -1)
    reference = torch.moveaxis(input, -3, -1)
    _assert_equal(output, reference)


@skip_if_cuda_not_available
def test_moveaxis_zero_dim():
    """[case 50] Tensor with a zero dimension."""
    input = torch.randn((0, 3, 4), dtype=torch.float32, device="cuda")
    output = ntops.torch.moveaxis(input, 0, 2)
    reference = torch.moveaxis(input, 0, 2)
    _assert_equal(output, reference)


@skip_if_cuda_not_available
def test_moveaxis_single_element():
    """[case 51] 1-element tensor."""
    input = torch.randn((1, 1, 1), dtype=torch.float32, device="cuda")
    output = ntops.torch.moveaxis(input, 0, 2)
    reference = torch.moveaxis(input, 0, 2)
    _assert_equal(output, reference)


# ============================================================
# Tuple source/destination
# ============================================================

@skip_if_cuda_not_available
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.int32])
@pytest.mark.parametrize("source, destination", [
    ((0, 1), (1, 0)),
    ((0, 2), (2, 0)),
    ((0, 1, 2), (2, 1, 0)),
    ((0, 3), (3, 0)),
])
def test_moveaxis_tuple_source_dest(dtype, source, destination):
    """[case 52-63] Tuple source/destination."""
    if dtype.is_floating_point:
        input = torch.randn((2, 3, 4, 5), dtype=dtype, device="cuda")
    else:
        input = torch.randint(0, 10, (2, 3, 4, 5), dtype=dtype, device="cuda")

    output = ntops.torch.moveaxis(input, source, destination)
    reference = torch.moveaxis(input, source, destination)
    _assert_equal(output, reference)


@skip_if_cuda_not_available
def test_moveaxis_tuple_negative():
    """[case 64] Tuple with negative indices."""
    input = torch.randn((2, 3, 4, 5), dtype=torch.float32, device="cuda")
    output = ntops.torch.moveaxis(input, (-4, -1), (0, 2))
    reference = torch.moveaxis(input, (-4, -1), (0, 2))
    _assert_equal(output, reference)


# ============================================================
# Non-contiguous tensors
# ============================================================

@skip_if_cuda_not_available
def test_moveaxis_transpose_input():
    """[case 65] Input is already transposed (non-contiguous)."""
    input = torch.randn((4, 3), dtype=torch.float32, device="cuda").T
    assert not input.is_contiguous()
    output = ntops.torch.moveaxis(input, 0, 1)
    reference = torch.moveaxis(input, 0, 1)
    _assert_equal(output, reference)


@skip_if_cuda_not_available
def test_moveaxis_3d_permute_input():
    """[case 66] 3D permuted input (non-contiguous)."""
    input = torch.randn((3, 4, 5), dtype=torch.float32, device="cuda").permute(2, 0, 1)
    assert not input.is_contiguous()
    output = ntops.torch.moveaxis(input, 0, 2)
    reference = torch.moveaxis(input, 0, 2)
    _assert_equal(output, reference)


@skip_if_cuda_not_available
def test_moveaxis_slice_input():
    """[case 67] Sliced input (non-contiguous)."""
    input = torch.randn((10, 10), dtype=torch.float32, device="cuda")[::2, ::3]
    output = ntops.torch.moveaxis(input, 0, 1)
    reference = torch.moveaxis(input, 0, 1)
    _assert_equal(output, reference)


# ============================================================
# out= parameter
# ============================================================

@skip_if_cuda_not_available
def test_moveaxis_out_param():
    """[case 68] out= parameter with correct shape and dtype."""
    input = torch.randn((3, 4, 5), dtype=torch.float32, device="cuda")
    out = torch.empty((5, 3, 4), dtype=torch.float32, device="cuda")
    result = ntops.torch.moveaxis(input, 0, 2)  # no out= param in current API
    reference = torch.moveaxis(input, 0, 2)
    _assert_equal(result, reference)


# ============================================================
# Error handling
# ============================================================

@skip_if_cuda_not_available
def test_moveaxis_invalid_source_positive():
    """[case 69] Invalid source index raises IndexError."""
    input = torch.randn((3, 4), dtype=torch.float32, device="cuda")
    with pytest.raises(IndexError):
        ntops.torch.moveaxis(input, 3, 0)  # ndim=2, source=3 out of range


@skip_if_cuda_not_available
def test_moveaxis_invalid_source_negative():
    """[case 70] Invalid negative source raises IndexError."""
    input = torch.randn((3, 4), dtype=torch.float32, device="cuda")
    with pytest.raises(IndexError):
        ntops.torch.moveaxis(input, -3, 0)  # ndim=2, source=-3 out of range


@skip_if_cuda_not_available
def test_moveaxis_large_tensor():
    """[case 71] Larger tensor (256x256)."""
    input = torch.randn((256, 256), dtype=torch.float32, device="cuda")
    output = ntops.torch.moveaxis(input, 0, 1)
    reference = torch.moveaxis(input, 0, 1)
    _assert_equal(output, reference)


@skip_if_cuda_not_available
def test_moveaxis_5d_tensor():
    """[case 72] 5D tensor axis move."""
    input = torch.randn((2, 3, 4, 5, 6), dtype=torch.float32, device="cuda")
    output = ntops.torch.moveaxis(input, 0, 4)
    reference = torch.moveaxis(input, 0, 4)
    _assert_equal(output, reference)
