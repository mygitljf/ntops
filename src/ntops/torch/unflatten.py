import functools

import ntops
from ntops.torch import _vendor_triton
from ntops.torch.utils import _cached_make

try:
    import torch as _torch
    _torch_Tensor = _torch.Tensor
except ImportError:
    _torch = None
    _torch_Tensor = ()


@functools.cache
def _unflatten_metadata(shape, stride, dim, sizes):
    ndim = len(shape)
    if dim < -ndim or dim >= ndim:
        raise IndexError(
            "Dimension out of range (expected to be in range of "
            f"[{-ndim}, {ndim - 1}], but got {dim})"
        )
    if dim < 0:
        dim += ndim

    old_dim_size = shape[dim]
    inferred_idx = None
    normalized = list(sizes)
    known_product = 1

    for i, size in enumerate(sizes):
        if size == -1:
            if inferred_idx is not None:
                raise RuntimeError("only one dimension can be inferred")
            inferred_idx = i
        elif size < 0:
            raise RuntimeError("invalid shape dimension")
        else:
            known_product *= size

    if inferred_idx is not None:
        if known_product == 0 or old_dim_size % known_product != 0:
            raise RuntimeError(
                f"unflatten: Provided sizes {sizes} do not multiply up to "
                f"the size of dim {dim} ({old_dim_size})"
            )
        normalized[inferred_idx] = old_dim_size // known_product
    elif known_product != old_dim_size:
        raise RuntimeError(
            f"unflatten: Provided sizes {sizes} do not multiply up to "
            f"the size of dim {dim} ({old_dim_size})"
        )

    n = len(normalized)
    split_strides = [0] * n
    tail_product = 1
    base_stride = stride[dim]
    for i in range(n - 1, -1, -1):
        split_strides[i] = base_stride * tail_product
        tail_product *= normalized[i]

    new_shape = shape[:dim] + tuple(normalized) + shape[dim + 1:]
    new_stride = stride[:dim] + tuple(split_strides) + stride[dim + 1:]
    return new_shape, new_stride


@functools.cache
def _get_kernel(ndim):
    return _cached_make(
        ntops.kernels.unflatten.premake,
        ndim,
        block_size=ntops.kernels.unflatten.BLOCK_SIZE,
        max_num_configs=1,
    )


def unflatten(input, dim, sizes):
    new_shape, new_stride = _unflatten_metadata(
        tuple(input.shape), tuple(input.stride()), dim, tuple(sizes)
    )
    view = input.as_strided(new_shape, new_stride)
    if _torch is not None and type(input) is _torch_Tensor:
        if (
            _vendor_triton.is_metax_device(input)
            and not input.is_contiguous()
            and input.dtype == _torch.float16
        ):
            materialized_input = _vendor_triton.materialize_metax_f16_permute3d(input)
            if materialized_input is None:
                materialized_input = _vendor_triton.materialize_to_contiguous(input)
            if materialized_input is not None:
                contiguous_shape, contiguous_stride = _unflatten_metadata(
                    tuple(materialized_input.shape), materialized_input.stride(), dim, tuple(sizes)
                )
                return materialized_input.as_strided(contiguous_shape, contiguous_stride)
        output = _torch.empty(view.shape, dtype=view.dtype, device=view.device)
        if view.numel() == 0:
            return output
        if _vendor_triton.materialize_fast_path(view, output):
            return output
        _get_kernel(view.ndim)(view, output)
        return output
    return view
