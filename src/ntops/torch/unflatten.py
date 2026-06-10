import functools
import operator

try:
    import torch as _torch
    _torch_unflatten = _torch.unflatten
    _torch_Tensor = _torch.Tensor
except ImportError:
    _torch_unflatten = None
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


def unflatten(input, dim, sizes):
    if _torch_unflatten is not None and type(input) is _torch_Tensor:
        return _torch_unflatten(input, dim, sizes)

    new_shape, new_stride = _unflatten_metadata(
        tuple(input.shape), input.stride(), dim, tuple(sizes)
    )
    return input.as_strided(new_shape, new_stride)
