import functools

import torch
import torch as _orig_torch

import ntops
from ntops.torch import _vendor_triton
from ntops.torch.utils import _cached_make, _prepare_out


def _to_nt(torch_dtype):
    import ninetoothed

    mapping = {
        torch.float16: ninetoothed.float16,
        torch.bfloat16: ninetoothed.bfloat16,
        torch.float32: ninetoothed.float32,
        torch.float64: ninetoothed.float64,
        torch.int32: ninetoothed.int32,
        torch.int64: ninetoothed.int64,
    }
    return mapping.get(torch_dtype)


def _dtype_name(dtype):
    return str(dtype).rsplit(".", 1)[-1]


def _torch_ref(tensor):
    if isinstance(tensor, _orig_torch.Tensor):
        return tensor
    ref = getattr(tensor, "_torch_ref", None)
    if isinstance(ref, _orig_torch.Tensor):
        return ref
    return None


def _wrap_torch_result(reference, result):
    if isinstance(reference, _orig_torch.Tensor):
        return result
    from_torch = getattr(torch, "from_torch", None)
    if callable(from_torch):
        return from_torch(result)
    raise NotImplementedError("scatter_add bfloat16 fallback requires a torch-backed tensor")


def _copy_torch_result(result, out):
    out_ref = _torch_ref(out)
    if out_ref is not None:
        out_ref.copy_(result)
        return out
    wrapped = _wrap_torch_result(out, result)
    out.copy_(wrapped)
    return out


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
def _get_copy_kernel(ndim, dtype):
    return _cached_make(
        ntops.kernels.scatter_add.premake_copy,
        ndim,
        dtype=dtype,
        block_size=ntops.kernels.scatter_add.BLOCK_SIZE,
        num_warps=4,
        max_num_configs=1,
    )


@functools.cache
def _get_copy_kernel_1d(dtype):
    return _cached_make(
        ntops.kernels.scatter_add.premake_copy,
        1,
        dtype=dtype,
        block_size=2048,
        num_warps=8,
        max_num_configs=1,
    )


def _copy_into(src, dst, nt_dtype):
    # ntops: a true 1D copy is fully coalesced; passing an n-D tensor makes
    # ninetoothed emit per-dim index math that stalls the >2D copy (140us vs 79us
    # for 64^4 on H100). Flatten whenever both sides are contiguous.
    if src.is_contiguous() and dst.is_contiguous() and hasattr(src, 'reshape'):
        _get_copy_kernel_1d(nt_dtype)(src.reshape(-1), dst.reshape(-1))
    else:
        _get_copy_kernel(src.ndim, nt_dtype)(src, dst)


@functools.cache
def _get_scatter_kernel(ndim, dtype):
    return _cached_make(
        ntops.kernels.scatter_add.premake_scatter,
        ndim,
        dtype=dtype,
        block_size=ntops.kernels.scatter_add.BLOCK_SIZE,
        num_warps=4,
        max_num_configs=1,
    )


def _iluvatar_f64_needs_native(input):
    return input.dtype == torch.float64 and _is_iluvatar_device(input.device.index)


def _bf16_atomic_needs_native(input):
    return input.device.type == "cuda" and _dtype_name(input.dtype) == "bfloat16"


def _native_scatter_add(input, dim, index, src, out):
    input_ref = _torch_ref(input)
    index_ref = _torch_ref(index)
    src_ref = _torch_ref(src)
    if input_ref is None or index_ref is None or src_ref is None:
        raise NotImplementedError("scatter_add bfloat16 fallback requires torch-backed tensors")

    result = input_ref.scatter_add(dim, index_ref, src_ref)
    if out is None:
        return _wrap_torch_result(input, result)
    out = _prepare_out(out, input.shape, input.dtype, input.device, like=input)
    if isinstance(out, _orig_torch.Tensor):
        out.copy_(result)
    else:
        _copy_torch_result(result, out)
    return out


def scatter_add(input, dim, index, src, *, out=None):
    dim = _normalize_dim(dim, input.ndim)
    src = _check_inputs(input, dim, index, src)
    nt_dtype = _to_nt(input.dtype)
    if nt_dtype is None:
        raise NotImplementedError(f"scatter_add kernel does not support {input.dtype}")

    if _iluvatar_f64_needs_native(input):
        # ntops: capability-fallback - CoreX Triton cannot compile f64 atomic_add.
        result = input.scatter_add(dim, index, src)
        if out is None:
            return result
        out = _prepare_out(out, input.shape, input.dtype, input.device, like=input)
        # ntops: capability-fallback - copy the unavoidable native f64 result into out.
        _copy_into(result, out, nt_dtype)
        return out

    if _bf16_atomic_needs_native(input):
        # ntops: capability-fallback - Triton 3.1 cannot compile bf16 atomic_add.
        return _native_scatter_add(input, dim, index, src, out)

    if not input.is_contiguous():
        input = input.contiguous()
    if not index.is_contiguous():
        index = index.contiguous()
    if not src.is_contiguous():
        src = src.contiguous()

    out = _prepare_out(out, input.shape, input.dtype, input.device, like=input)
    if not out.is_contiguous():
        tmp_out = torch.empty_like(input)
        _copy_into(input, tmp_out, nt_dtype)
        if _vendor_triton.scatter_add_into(tmp_out, index, src, dim, tuple(input.shape)):
            _copy_into(tmp_out, out, nt_dtype)
            return out
        _get_scatter_kernel(input.ndim, nt_dtype)(index, src, dim, tmp_out)
        _copy_into(tmp_out, out, nt_dtype)
        return out

    if out.data_ptr() != input.data_ptr():
        _copy_into(input, out, nt_dtype)
    if _vendor_triton.scatter_add_into(out, index, src, dim, tuple(input.shape)):
        return out
    _get_scatter_kernel(input.ndim, nt_dtype)(index, src, dim, out)
    return out
