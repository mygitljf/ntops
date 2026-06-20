import pytest
import torch

import ntops
from tests.skippers import skip_if_cuda_not_available


def _assert_close(output, reference, rtol=1e-3, atol=1e-3):
    assert output.shape == reference.shape
    assert output.dtype == reference.dtype
    if reference.dtype.is_floating_point:
        assert torch.allclose(output, reference, rtol=rtol, atol=atol, equal_nan=True)
    else:
        assert torch.equal(output, reference)


# ============================================================
# Basic shape × dtype tests
# ============================================================

@skip_if_cuda_not_available
@pytest.mark.parametrize("shape", [(2, 2), (2, 6), (3, 12), (4, 24)])
def test_unflatten_2d_basic(shape):
    """[Case 1-4] 2D contiguous tensors, unflatten dim=1 into (2, ...)."""
    input = torch.randn(shape, dtype=torch.float32, device="cuda")
    sizes = (2, shape[1] // 2)

    output = ntops.torch.unflatten(input, 1, sizes)
    reference = torch.unflatten(input, 1, sizes)

    _assert_close(output, reference)


@skip_if_cuda_not_available
@pytest.mark.parametrize("shape", [(2, 3, 4), (3, 4, 6), (4, 5, 8)])
def test_unflatten_3d_basic(shape):
    """[Case 5-7] 3D contiguous tensors, unflatten dim=1."""
    input = torch.randn(shape, dtype=torch.float32, device="cuda")
    sizes = (shape[1], 1)  # reshape to add a dimension

    output = ntops.torch.unflatten(input, 1, sizes)
    reference = torch.unflatten(input, 1, sizes)

    _assert_close(output, reference)


# ============================================================
# dtype tests
# ============================================================

@skip_if_cuda_not_available
@pytest.mark.parametrize("dtype", [torch.float16, torch.float32, torch.float64])
def test_unflatten_float_dtypes(dtype):
    """[Case 8-10] Float dtypes: f16, f32, f64."""
    input = torch.randn(7, 12, dtype=dtype, device="cuda")
    sizes = (3, 4)

    output = ntops.torch.unflatten(input, 1, sizes)
    reference = torch.unflatten(input, 1, sizes)

    _assert_close(output, reference)


@skip_if_cuda_not_available
@pytest.mark.parametrize("dtype", [torch.int32, torch.int64])
def test_unflatten_int_dtypes(dtype):
    """[Case 11-12] Integer dtypes: int32, int64."""
    input = torch.randint(0, 100, (5, 10), dtype=dtype, device="cuda")
    sizes = (2, 5)

    output = ntops.torch.unflatten(input, 1, sizes)
    reference = torch.unflatten(input, 1, sizes)

    _assert_close(output, reference)


@skip_if_cuda_not_available
def test_unflatten_bool_dtype():
    """[Case 13] Bool dtype."""
    input = torch.tensor([[True, False, True, False]], device="cuda").repeat(3, 1)
    sizes = (2, 2)

    output = ntops.torch.unflatten(input, 1, sizes)
    reference = torch.unflatten(input, 1, sizes)

    _assert_close(output, reference)


# ============================================================
# Different unflatten patterns
# ============================================================

@skip_if_cuda_not_available
def test_unflatten_3d_multiple_sizes():
    """[Case 14] 3D, unflatten dim=2 into (2, 2, 2) - 3 sizes."""
    input = torch.randn(2, 3, 8, dtype=torch.float32, device="cuda")

    output = ntops.torch.unflatten(input, 2, (2, 2, 2))
    reference = torch.unflatten(input, 2, (2, 2, 2))

    _assert_close(output, reference)


@skip_if_cuda_not_available
def test_unflatten_3d_dim0():
    """[Case 15] Unflatten dim=0."""
    input = torch.randn(8, 3, 4, dtype=torch.float32, device="cuda")

    output = ntops.torch.unflatten(input, 0, (2, 4))
    reference = torch.unflatten(input, 0, (2, 4))

    _assert_close(output, reference)


@skip_if_cuda_not_available
def test_unflatten_3d_dim2():
    """[Case 16] Unflatten dim=2 (last dim)."""
    input = torch.randn(2, 3, 8, dtype=torch.float32, device="cuda")

    output = ntops.torch.unflatten(input, 2, (2, 4))
    reference = torch.unflatten(input, 2, (2, 4))

    _assert_close(output, reference)


# ============================================================
# Negative dim tests
# ============================================================

@skip_if_cuda_not_available
def test_unflatten_negative_dim():
    """[Case 17] Negative dim=-1 (unflatten last dim)."""
    input = torch.randn(2, 3, 8, dtype=torch.float32, device="cuda")

    output = ntops.torch.unflatten(input, -1, (2, 4))
    reference = torch.unflatten(input, -1, (2, 4))

    _assert_close(output, reference)


@skip_if_cuda_not_available
def test_unflatten_negative_dim2():
    """[Case 18] Negative dim=-2 on 3D tensor."""
    input = torch.randn(2, 6, 4, dtype=torch.float32, device="cuda")

    output = ntops.torch.unflatten(input, -2, (2, 3))
    reference = torch.unflatten(input, -2, (2, 3))

    _assert_close(output, reference)


# ============================================================
# Non-contiguous input tests
# ============================================================

@skip_if_cuda_not_available
def test_unflatten_non_contiguous_transpose():
    """[Case 19] Transposed 2D tensor, unflatten dim=0."""
    base = torch.randn((6, 5), dtype=torch.float32, device="cuda")
    input = base.t()  # shape (5, 6), non-contiguous
    assert not input.is_contiguous()

    output = ntops.torch.unflatten(input, 1, (2, 3))
    reference = torch.unflatten(input, 1, (2, 3))

    _assert_close(output, reference)


@skip_if_cuda_not_available
def test_unflatten_non_contiguous_transpose_dim0():
    """[Case 20] Transposed 2D tensor, unflatten dim=0."""
    base = torch.randn((5, 8), dtype=torch.float32, device="cuda")
    input = base.t()  # shape (8, 5), non-contiguous
    assert not input.is_contiguous()

    output = ntops.torch.unflatten(input, 0, (2, 4))
    reference = torch.unflatten(input, 0, (2, 4))

    _assert_close(output, reference)


@skip_if_cuda_not_available
def test_unflatten_non_contiguous_permute_3d():
    """[Case 21] Permuted 3D tensor, unflatten dim=1."""
    input = torch.randn((3, 5, 7), dtype=torch.float32, device="cuda").permute(2, 0, 1)
    assert not input.is_contiguous()

    output = ntops.torch.unflatten(input, 1, (1, 3))
    reference = torch.unflatten(input, 1, (1, 3))

    _assert_close(output, reference)


# ============================================================
# Special values
# ============================================================

@skip_if_cuda_not_available
def test_unflatten_special_values():
    """[Case 22] Special float values: 0, -0, inf, -inf, nan."""
    input = torch.tensor(
        [[0.0, -0.0, float("inf"), -float("inf"), float("nan"), 1.0]],
        dtype=torch.float32, device="cuda",
    ).repeat(3, 1)

    output = ntops.torch.unflatten(input, 1, (2, 3))
    reference = torch.unflatten(input, 1, (2, 3))

    _assert_close(output, reference)


@skip_if_cuda_not_available
def test_unflatten_large():
    """[Case 23] Large 4D tensor, unflatten middle dim."""
    input = torch.randn(2, 16, 32, 64, dtype=torch.float32, device="cuda")

    output = ntops.torch.unflatten(input, 1, (2, 8))
    reference = torch.unflatten(input, 1, (2, 8))

    _assert_close(output, reference)


@skip_if_cuda_not_available
def test_unflatten_bfloat16():
    """[Case 24] bfloat16 dtype."""
    input = torch.randn(3, 6, dtype=torch.bfloat16, device="cuda")

    output = ntops.torch.unflatten(input, 1, (2, 3))
    reference = torch.unflatten(input, 1, (2, 3))

    _assert_close(output, reference)


@skip_if_cuda_not_available
def test_unflatten_int8():
    """[Case 25] int8 dtype."""
    input = torch.randint(0, 10, (4, 6), dtype=torch.int8, device="cuda")

    output = ntops.torch.unflatten(input, 1, (2, 3))
    reference = torch.unflatten(input, 1, (2, 3))

    _assert_close(output, reference)


@skip_if_cuda_not_available
def test_unflatten_uint8():
    """[Case 26] uint8 dtype."""
    input = torch.randint(0, 10, (4, 6), dtype=torch.uint8, device="cuda")

    output = ntops.torch.unflatten(input, 1, (2, 3))
    reference = torch.unflatten(input, 1, (2, 3))

    _assert_close(output, reference)


@skip_if_cuda_not_available
def test_unflatten_int16():
    """[Case 27] int16 dtype."""
    input = torch.randint(0, 10, (4, 6), dtype=torch.int16, device="cuda")

    output = ntops.torch.unflatten(input, 1, (2, 3))
    reference = torch.unflatten(input, 1, (2, 3))

    _assert_close(output, reference)


# ============================================================
# Edge cases
# ============================================================

@skip_if_cuda_not_available
def test_unflatten_single_dim():
    """[Case 28] Unflatten with single-element sizes tuple."""
    input = torch.randn(2, 6, 4, dtype=torch.float32, device="cuda")

    output = ntops.torch.unflatten(input, 1, (6,))
    reference = torch.unflatten(input, 1, (6,))

    _assert_close(output, reference)


@skip_if_cuda_not_available
def test_unflatten_1d_input():
    """[Case 29] 1D input tensor."""
    input = torch.randn(12, dtype=torch.float32, device="cuda")

    output = ntops.torch.unflatten(input, 0, (3, 4))
    reference = torch.unflatten(input, 0, (3, 4))

    _assert_close(output, reference)


@skip_if_cuda_not_available
def test_unflatten_4d_tensor():
    """[Case 30] 4D tensor, unflatten dim=2."""
    input = torch.randn(2, 3, 8, 5, dtype=torch.float32, device="cuda")

    output = ntops.torch.unflatten(input, 2, (2, 4))
    reference = torch.unflatten(input, 2, (2, 4))

    _assert_close(output, reference)


@skip_if_cuda_not_available
def test_unflatten_many_dims():
    """[Case 31] Unflatten into 4 sub-dimensions."""
    input = torch.randn(2, 24, 3, dtype=torch.float32, device="cuda")

    output = ntops.torch.unflatten(input, 1, (2, 3, 4))
    reference = torch.unflatten(input, 1, (2, 3, 4))

    _assert_close(output, reference)
