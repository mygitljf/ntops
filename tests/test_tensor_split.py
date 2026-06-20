import pytest
import torch

import ntops
from tests.skippers import skip_if_cuda_not_available


def _assert_tensor_split_equal(outputs, references):
    """Compare tuples of tensors from tensor_split."""
    assert len(outputs) == len(references)
    for out, ref in zip(outputs, references):
        assert out.shape == ref.shape, f"shape mismatch: {out.shape} vs {ref.shape}"
        assert out.dtype == ref.dtype, f"dtype mismatch: {out.dtype} vs {ref.dtype}"
        if ref.dtype.is_floating_point:
            assert torch.allclose(out, ref, rtol=2e-3, atol=2e-3, equal_nan=True)
        else:
            assert torch.equal(out, ref), f"value mismatch for dtype {out.dtype}"


def _storage_ptr(tensor):
    return tensor.untyped_storage().data_ptr()


# === Integer split mode tests ===

@skip_if_cuda_not_available
@pytest.mark.parametrize("dtype", [torch.float16, torch.float32, torch.float64])
@pytest.mark.parametrize("shape", [(0,), (1,), (7,), (10,), (3, 5), (2, 3, 4)])
def test_tensor_split_int_float_shapes(shape, dtype):
    """[1-18] Float dtypes x multiple shapes with integer split."""
    input = torch.randn(shape, dtype=dtype, device="cuda")
    sections = 2 if shape[0] >= 2 else 1

    output = ntops.torch.tensor_split(input, sections)
    reference = torch.tensor_split(input, sections)

    _assert_tensor_split_equal(output, reference)


@skip_if_cuda_not_available
@pytest.mark.parametrize("dtype", [torch.int32, torch.int64])
@pytest.mark.parametrize("shape", [(0,), (1,), (8,), (4, 3)])
def test_tensor_split_int_integer_dtypes(shape, dtype):
    """[19-26] Integer dtypes with integer split."""
    input = torch.randint(-100, 100, shape, dtype=dtype, device="cuda")
    sections = 2 if shape[0] >= 2 else 1

    output = ntops.torch.tensor_split(input, sections)
    reference = torch.tensor_split(input, sections)

    _assert_tensor_split_equal(output, reference)


@skip_if_cuda_not_available
def test_tensor_split_int_special_values():
    """[27] Special float values: 0, -0, inf, -inf, nan."""
    input = torch.tensor(
        [0.0, -0.0, float("inf"), -float("inf"), float("nan")],
        dtype=torch.float32,
        device="cuda",
    )

    output = ntops.torch.tensor_split(input, 2)
    reference = torch.tensor_split(input, 2)

    _assert_tensor_split_equal(output, reference)


@skip_if_cuda_not_available
def test_tensor_split_bool_dtype():
    """[28] Bool dtype support."""
    input = torch.tensor([True, False, True, False, True, False], device="cuda")

    output = ntops.torch.tensor_split(input, 2)
    reference = torch.tensor_split(input, 2)

    _assert_tensor_split_equal(output, reference)


@skip_if_cuda_not_available
def test_tensor_split_uint8_dtype():
    """[29] uint8 dtype support."""
    input = torch.tensor([1, 2, 3, 4, 5, 6], dtype=torch.uint8, device="cuda")

    output = ntops.torch.tensor_split(input, 2)
    reference = torch.tensor_split(input, 2)

    _assert_tensor_split_equal(output, reference)


@skip_if_cuda_not_available
def test_tensor_split_int16_dtype():
    """[30] int16 dtype support."""
    input = torch.tensor([1, 2, 3, 4, 5, 6], dtype=torch.int16, device="cuda")

    output = ntops.torch.tensor_split(input, 2)
    reference = torch.tensor_split(input, 2)

    _assert_tensor_split_equal(output, reference)


@skip_if_cuda_not_available
def test_tensor_split_uneven_int_split():
    """[31] Uneven integer split (7 split into 3)."""
    input = torch.arange(7, device="cuda", dtype=torch.float32)

    output = ntops.torch.tensor_split(input, 3)
    reference = torch.tensor_split(input, 3)

    _assert_tensor_split_equal(output, reference)


@skip_if_cuda_not_available
def test_tensor_split_uneven_int_split_2():
    """[32] Uneven integer split (10 split into 3)."""
    input = torch.arange(10, device="cuda", dtype=torch.float32)

    output = ntops.torch.tensor_split(input, 3)
    reference = torch.tensor_split(input, 3)

    _assert_tensor_split_equal(output, reference)


# === Index split mode tests ===

@skip_if_cuda_not_available
def test_tensor_split_index_basic():
    """[33] Index-based split with [3, 6] on 10 elements."""
    input = torch.arange(10, device="cuda", dtype=torch.float32)

    output = ntops.torch.tensor_split(input, [3, 6])
    reference = torch.tensor_split(input, [3, 6])

    _assert_tensor_split_equal(output, reference)


@skip_if_cuda_not_available
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.int32, torch.int64])
def test_tensor_split_index_2d(dtype):
    """[34-37] Index-based split on 2D tensor along dim=1."""
    input = torch.arange(24, device="cuda", dtype=dtype).reshape(4, 6)

    output = ntops.torch.tensor_split(input, [2, 4], dim=1)
    reference = torch.tensor_split(input, [2, 4], dim=1)

    _assert_tensor_split_equal(output, reference)


@skip_if_cuda_not_available
def test_tensor_split_index_single():
    """[38] Single index split (effectively 2 parts)."""
    input = torch.arange(10, device="cuda", dtype=torch.float32)

    output = ntops.torch.tensor_split(input, [4])
    reference = torch.tensor_split(input, [4])

    _assert_tensor_split_equal(output, reference)


@skip_if_cuda_not_available
def test_tensor_split_index_empty():
    """[39] Empty indices list returns single tensor."""
    input = torch.arange(10, device="cuda", dtype=torch.float32)

    output = ntops.torch.tensor_split(input, [])
    reference = torch.tensor_split(input, [])

    _assert_tensor_split_equal(output, reference)


@skip_if_cuda_not_available
@pytest.mark.parametrize("indices", ([2, 2], [3, 1], [8], [-8], [6, 3], [0], [10], [11]))
def test_tensor_split_index_matches_pytorch_edge_semantics(indices):
    """Index split matches PyTorch for duplicate, unsorted, and out-of-bound indices."""
    input = torch.arange(5, device="cuda", dtype=torch.float32)

    output = ntops.torch.tensor_split(input, indices)
    reference = torch.tensor_split(input, indices)

    _assert_tensor_split_equal(output, reference)


# === Dimension tests ===

@skip_if_cuda_not_available
def test_tensor_split_dim1():
    """[40] Split along dim=1."""
    input = torch.arange(24, device="cuda", dtype=torch.float32).reshape(3, 8)

    output = ntops.torch.tensor_split(input, 2, dim=1)
    reference = torch.tensor_split(input, 2, dim=1)

    _assert_tensor_split_equal(output, reference)


@skip_if_cuda_not_available
def test_tensor_split_dim2():
    """[41] Split along dim=2 (3D tensor)."""
    input = torch.randn((2, 3, 4), dtype=torch.float32, device="cuda")

    output = ntops.torch.tensor_split(input, 2, dim=2)
    reference = torch.tensor_split(input, 2, dim=2)

    _assert_tensor_split_equal(output, reference)


@skip_if_cuda_not_available
def test_tensor_split_negative_dim():
    """[42] Split along negative dimension dim=-1."""
    input = torch.arange(24, device="cuda", dtype=torch.float32).reshape(3, 8)

    output = ntops.torch.tensor_split(input, 2, dim=-1)
    reference = torch.tensor_split(input, 2, dim=-1)

    _assert_tensor_split_equal(output, reference)


@skip_if_cuda_not_available
def test_tensor_split_negative_dim_3d():
    """[43] Split along negative dimension dim=-2 on 3D."""
    input = torch.randn((2, 3, 4), dtype=torch.float32, device="cuda")

    output = ntops.torch.tensor_split(input, 3, dim=-2)
    reference = torch.tensor_split(input, 3, dim=-2)

    _assert_tensor_split_equal(output, reference)


# === Non-contiguous input tests ===

@skip_if_cuda_not_available
def test_tensor_split_non_contiguous_transpose():
    """[44] Split non-contiguous transposed tensor."""
    base = torch.randn((5, 7), dtype=torch.float32, device="cuda")
    input = base.t()  # non-contiguous

    output = ntops.torch.tensor_split(input, 2, dim=0)
    reference = torch.tensor_split(input, 2, dim=0)

    _assert_tensor_split_equal(output, reference)


@skip_if_cuda_not_available
def test_tensor_split_non_contiguous_permute_3d():
    """[45] Split non-contiguous 3D permuted tensor."""
    input = torch.randn((3, 5, 7), dtype=torch.float32, device="cuda").permute(2, 0, 1)

    output = ntops.torch.tensor_split(input, 3, dim=1)
    reference = torch.tensor_split(input, 3, dim=1)

    _assert_tensor_split_equal(output, reference)


@skip_if_cuda_not_available
def test_tensor_split_non_contiguous_transpose_dim1():
    """[46] Split non-contiguous transposed tensor along dim=1."""
    base = torch.randn((4, 8), dtype=torch.float32, device="cuda")
    input = base.t()

    output = ntops.torch.tensor_split(input, 2, dim=1)
    reference = torch.tensor_split(input, 2, dim=1)

    _assert_tensor_split_equal(output, reference)


@skip_if_cuda_not_available
def test_tensor_split_non_contiguous_outputs_are_views():
    """Pure-kernel tensor_split materializes non-contiguous sections."""
    base = torch.arange(12, dtype=torch.float32, device="cuda").reshape(3, 4)
    input = base.t()

    output = ntops.torch.tensor_split(input, 2, dim=0)
    reference = torch.tensor_split(input, 2, dim=0)

    _assert_tensor_split_equal(output, reference)
    for out in output:
        assert out.is_contiguous()
        assert _storage_ptr(out) != _storage_ptr(input)


# === Edge case tests ===

@skip_if_cuda_not_available
def test_tensor_split_split_into_one():
    """[47] Split into 1 section returns single-element tuple."""
    input = torch.arange(10, device="cuda", dtype=torch.float32)

    output = ntops.torch.tensor_split(input, 1)
    reference = torch.tensor_split(input, 1)

    _assert_tensor_split_equal(output, reference)


@skip_if_cuda_not_available
def test_tensor_split_split_each_element():
    """[48] Split into n sections for n-element tensor (each element its own)."""
    input = torch.arange(5, device="cuda", dtype=torch.float32)

    output = ntops.torch.tensor_split(input, 5)
    reference = torch.tensor_split(input, 5)

    _assert_tensor_split_equal(output, reference)


@skip_if_cuda_not_available
def test_tensor_split_negative_indices():
    """[49] Negative indices in index-based split."""
    input = torch.arange(10, device="cuda", dtype=torch.float32)

    output = ntops.torch.tensor_split(input, [-1])
    reference = torch.tensor_split(input, [-1])

    _assert_tensor_split_equal(output, reference)


@skip_if_cuda_not_available
def test_tensor_split_large_1d():
    """[50] Large 1D tensor split."""
    input = torch.randn((1 << 20,), dtype=torch.float32, device="cuda")

    output = ntops.torch.tensor_split(input, 4)
    reference = torch.tensor_split(input, 4)

    _assert_tensor_split_equal(output, reference)


@skip_if_cuda_not_available
def test_tensor_split_4d_tensor():
    """[51] 4D tensor split along dim=2."""
    input = torch.randn((2, 3, 4, 5), dtype=torch.float32, device="cuda")

    output = ntops.torch.tensor_split(input, 2, dim=2)
    reference = torch.tensor_split(input, 2, dim=2)

    _assert_tensor_split_equal(output, reference)


@skip_if_cuda_not_available
def test_tensor_split_bfloat16():
    """[52] bfloat16 dtype support."""
    input = torch.arange(10, device="cuda", dtype=torch.bfloat16)

    output = ntops.torch.tensor_split(input, 2)
    reference = torch.tensor_split(input, 2)

    _assert_tensor_split_equal(output, reference)


@skip_if_cuda_not_available
def test_tensor_split_index_tuple():
    """[53] Indices as tuple instead of list."""
    input = torch.arange(10, device="cuda", dtype=torch.float32)

    output = ntops.torch.tensor_split(input, (3, 7))
    reference = torch.tensor_split(input, (3, 7))

    _assert_tensor_split_equal(output, reference)
