import functools
import warnings

import ninetoothed
import torch

import ntops


class _CachedMakeDefaultConfig:
    def __init__(self, num_warps=None, num_stages=None, max_num_configs=None):
        self.num_warps = num_warps

        self.num_stages = num_stages

        self.max_num_configs = max_num_configs


_cached_make_default_config = _CachedMakeDefaultConfig()


def get_default_num_warps():
    return _cached_make_default_config.num_warps


def set_default_num_warps(num_warps):
    _cached_make_default_config.num_warps = num_warps


def get_default_num_stages():
    return _cached_make_default_config.num_stages


def set_default_num_stages(num_stages):
    _cached_make_default_config.num_stages = num_stages


def get_default_max_num_configs():
    return _cached_make_default_config.max_num_configs


def set_default_max_num_configs(max_num_configs):
    _cached_make_default_config.max_num_configs = max_num_configs


@functools.cache
def _cached_make(
    premake, *args, num_warps=None, num_stages=None, max_num_configs=None, **keywords
):
    if num_warps is None:
        num_warps = _cached_make_default_config.num_warps

    if num_stages is None:
        num_stages = _cached_make_default_config.num_stages

    if max_num_configs is None:
        max_num_configs = _cached_make_default_config.max_num_configs

    return ninetoothed.make(
        *premake(*args, **keywords),
        num_warps=num_warps,
        num_stages=num_stages,
        max_num_configs=max_num_configs,
    )


def _reshape_tensor(tensor, shape):
    reshape = getattr(tensor, "reshape", None)
    if callable(reshape):
        return reshape(shape)
    resolved = list(shape)
    if -1 in resolved:
        idx = resolved.index(-1)
        known = 1
        for i, s in enumerate(resolved):
            if i != idx:
                known *= int(s)
        resolved[idx] = tensor.numel() // known
    return tensor.view(tuple(resolved))


def _clone(tensor):
    """Clone a tensor, namespace-independent. Falls back to view for infinicore."""
    clone_fn = getattr(tensor, "clone", None)
    if callable(clone_fn):
        return clone_fn()
    zeros_fn = getattr(tensor, "new_zeros", None)
    if callable(zeros_fn):
        return tensor + zeros_fn(())
    return tensor.view(tensor.shape)


def _is_contiguous(tensor):
    is_contiguous = getattr(tensor, "is_contiguous", None)
    if callable(is_contiguous):
        return is_contiguous()
    return bool(is_contiguous)


def _strides(tensor):
    stride = getattr(tensor, "stride", None)
    if callable(stride):
        return tuple(stride())
    strides = getattr(tensor, "strides", None)
    if strides is not None:
        return tuple(strides)
    return None


def _permute_tensor(tensor, dims):
    permute = getattr(tensor, "permute", None)
    if callable(permute):
        return permute(dims)
    raise TypeError("tensor does not support permute")


def _physical_contiguous_permutation(tensors):
    if not tensors:
        return None

    ndim = tensors[0].ndim
    shape = tuple(tensors[0].shape)
    if ndim <= 1 or any(tensor.ndim != ndim or tuple(tensor.shape) != shape for tensor in tensors):
        return None

    strides = _strides(tensors[0])
    if strides is None or _is_contiguous(tensors[0]):
        return None

    dims = tuple(sorted(range(ndim), key=lambda dim: strides[dim], reverse=True))
    if dims == tuple(range(ndim)):
        return None

    try:
        if not _is_contiguous(_permute_tensor(tensors[0], dims)):
            return None
        if not all(_is_contiguous(_permute_tensor(tensor, dims)) for tensor in tensors[1:]):
            return None
    except TypeError:
        return None

    return dims


def _flatten_kernel_tensors(*tensors):
    kernel_tensors = tuple(
        _reshape_tensor(tensor, (1,)) if tensor.ndim == 0 else tensor
        for tensor in tensors
    )
    if all(tensor.ndim == 1 and _is_contiguous(tensor) for tensor in kernel_tensors):
        return kernel_tensors

    physical_order = _physical_contiguous_permutation(kernel_tensors)
    if physical_order is not None:
        kernel_tensors = tuple(_permute_tensor(tensor, physical_order) for tensor in kernel_tensors)

    if all(tensor.ndim > 0 and _is_contiguous(tensor) for tensor in kernel_tensors):
        return tuple(_reshape_tensor(tensor, (tensor.numel(),)) for tensor in kernel_tensors)
    return kernel_tensors


def _check_out_dtype(result_dtype, out):
    if out is None:
        return

    try:
        can_cast = torch.can_cast(result_dtype, out.dtype) if hasattr(torch, "can_cast") else True
    except TypeError:
        can_cast = result_dtype == out.dtype

    if not can_cast:
        raise RuntimeError(
            f"result type {result_dtype} can't be cast to the desired output type {out.dtype}"
        )


def _prepare_out(out, shape, dtype, device, like=None):
    _check_out_dtype(dtype, out)
    shape = tuple(shape)

    if out is None:
        if like is not None and tuple(like.shape) == shape and like.dtype == dtype:
            try:
                return torch.empty_like(like)
            except TypeError:
                import infinicore

                return infinicore.empty_like(like, dtype=dtype, device=device)
        try:
            return torch.empty(shape, dtype=dtype, device=device)
        except TypeError:
            import infinicore

            return infinicore.empty(list(shape), dtype=dtype, device=device)

    if tuple(out.shape) != tuple(shape):
        warnings.warn(
            (
                f"An output with one or more elements was resized since it had shape "
                f"{tuple(out.shape)}, which does not match the required output shape "
                f"{tuple(shape)}."
            ),
            UserWarning,
            stacklevel=2,
        )
        out.resize_(shape)

    return out


def _dtype_name(dtype):
    """Return the short name of a dtype, namespace-independent.

    Works for torch.float32, infinicore.float32, numpy.float32, etc.
    Returns the last segment after the last dot, lowercased.
    """
    return str(dtype).split(".")[-1].lower()


def _is_dtype(dtype, name):
    """Check if dtype has the given short name (namespace-independent)."""
    return _dtype_name(dtype) == name.lower()


def _is_dtype_any(dtype, names):
    """Check if dtype matches any of the given short names."""
    return _dtype_name(dtype) in {n.lower() for n in names}


def _dtype_is_floating_point(dtype):
    """Check if dtype is floating-point, namespace-independent."""
    if hasattr(dtype, "is_floating_point"):
        return bool(dtype.is_floating_point)
    return _dtype_name(dtype) in ("float16", "float32", "float64", "bfloat16", "half", "float", "double")


def _cast_dtype(tensor, dtype):
    """Cast tensor to dtype only if different, namespace-independent."""
    if _dtype_name(getattr(tensor, "dtype", None)) == _dtype_name(dtype):
        return tensor
    try:
        return tensor.to(dtype=dtype)
    except (TypeError, AttributeError):
        import torch as _torch
        name = _dtype_name(dtype)
        real_dtype = getattr(_torch, name, None)
        if real_dtype is not None:
            return tensor.to(dtype=real_dtype)
        raise


def _to_nt(torch_dtype):
    """Map dtype to ninetoothed dtype, namespace-independent."""
    import ninetoothed

    name = _dtype_name(torch_dtype)
    mapping = {
        "float16": ninetoothed.float16,
        "bfloat16": ninetoothed.bfloat16,
        "float32": ninetoothed.float32,
        "float64": ninetoothed.float64,
        "int16": ninetoothed.int16,
        "int32": ninetoothed.int32,
        "int64": ninetoothed.int64,
    }
    return mapping.get(name)


@functools.cache
def _device_name_for_index(index):
    if not hasattr(torch, "cuda") or not torch.cuda.is_available():
        return ""
    try:
        return torch.cuda.get_device_name(index)
    except Exception:
        return ""


def _is_corex_compat_device(device):
    if getattr(device, "type", None) != "cuda":
        return False
    index = device.index
    if index is None:
        index = torch.cuda.current_device()
    name = _device_name_for_index(index)
    return "Iluvatar" in name or "MetaX" in name or "MXC" in name


def _torch_binary_fallback(op_name, input, other, out):
    raise RuntimeError(
        f"{op_name}: PyTorch runtime fallback is disabled for ntops operators"
    )


def _get_matmul_input_precision():
    if torch.get_float32_matmul_precision() == "highest":
        return ntops.kernels.mm.InputPrecisionVariant.IEEE

    return ntops.kernels.mm.InputPrecisionVariant.TF32
