import functools

import torch

import ntops
from ntops.torch import _vendor_triton
from ntops.torch.utils import _cached_make, _prepare_out


@functools.cache
def _is_iluvatar_device(index):
    if not hasattr(torch, "cuda"):
        return False
    if not torch.cuda.is_available():
        return False
    try:
        return "Iluvatar" in torch.cuda.get_device_name(index)
    except Exception:
        return False


@functools.cache
def _is_metax_device(index):
    if not hasattr(torch, "cuda"):
        return False
    if not torch.cuda.is_available():
        return False
    try:
        name = torch.cuda.get_device_name(index)
    except Exception:
        return False
    return "MetaX" in name or "MXC" in name


def _normalize_dim(dim, ndim):
    if dim < 0:
        dim += ndim
    if dim < 0 or dim >= ndim:
        raise IndexError("Dimension out of range")
    return dim


def _check_inputs(input, dim, index, src):
    if input.ndim != src.ndim or input.ndim != index.ndim:
        raise RuntimeError("Index tensor must have the same number of dimensions as self tensor")
    if index.dtype != torch.long:
        raise RuntimeError("scatter_add(): Expected dtype int64 for index")
    if src.dtype != input.dtype:
        src = src.to(input.dtype)
    for axis, (index_size, src_size, input_size) in enumerate(
        zip(index.shape, src.shape, input.shape)
    ):
        if index_size > src_size:
            raise RuntimeError("Expected index to be no larger than src")
        if axis != dim and index_size > input_size:
            raise RuntimeError("Expected index to be no larger than self apart from dimension")
    return src


@functools.cache
def _get_scatter_kernel(ndim):
    return _cached_make(
        ntops.kernels.scatter_add.premake_scatter,
        ndim,
        block_size=ntops.kernels.scatter_add.BLOCK_SIZE,
        num_warps=4,
        max_num_configs=1,
    )


def _iluvatar_f64_needs_native(input):
    return input.dtype == torch.float64 and _is_iluvatar_device(input.device.index)


def scatter_add(input, dim, index, src, *, out=None):
    dim = _normalize_dim(dim, input.ndim)
    src = _check_inputs(input, dim, index, src)

    if _iluvatar_f64_needs_native(input):
        result = input.scatter_add(dim, index, src)
        if out is None:
            return result
        out = _prepare_out(out, input.shape, input.dtype, input.device, like=input)
        out.copy_(result)
        return out

    if not input.is_contiguous():
        input = input.contiguous()
    if not index.is_contiguous():
        index = index.contiguous()
    if not src.is_contiguous():
        src = src.contiguous()

    out = _prepare_out(out, input.shape, input.dtype, input.device, like=input)
    if not out.is_contiguous():
        tmp_out = torch.empty_like(input)
        tmp_out.copy_(input)
        if _vendor_triton.scatter_add_into(tmp_out, index, src, dim, tuple(input.shape)):
            out.copy_(tmp_out)
            return out
        _get_scatter_kernel(input.ndim)(index, src, dim, tmp_out)
        out.copy_(tmp_out)
        return out

    if out.data_ptr() != input.data_ptr():
        out.copy_(input)
    if _vendor_triton.scatter_add_into(out, index, src, dim, tuple(input.shape)):
        return out
    _get_scatter_kernel(input.ndim)(index, src, dim, out)
    return out
