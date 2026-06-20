import torch

import ntops
from ntops.torch import _vendor_triton
from ntops.torch._vendor_triton import is_iluvatar_device
from ntops.torch.utils import (
    _cached_make,
    _cast_dtype,
    _dtype_is_floating_point,
    _flatten_kernel_tensors,
    _prepare_out,
)


_NUM_WARPS = 4
_NUM_STAGES = 2


def _broadcast(input, other):
    if hasattr(torch, "broadcast_tensors"):
        return torch.broadcast_tensors(input, other)
    return input, other


def _prepare_inputs(input, other):
    result_dtype = torch.result_type(input, other) if hasattr(torch, "result_type") else input.dtype
    if not _dtype_is_floating_point(result_dtype):
        raise NotImplementedError("nextafter is only implemented for floating point inputs")
    return _cast_dtype(input, result_dtype), _cast_dtype(other, result_dtype), result_dtype


def _compile_errors():
    errors = [FileNotFoundError]
    try:
        from triton.compiler.errors import CompilationError

        errors.append(CompilationError)
    except ImportError:
        pass
    return tuple(errors)


# B 类兜底：记录半精 int16-bitcast kernel 编译失败的 dtype，
# 这些 dtype 改走 fp32 round-trip NineToothed kernel（仍是 kernel 派发）。
_HALF_BITCAST_BROKEN = set()


def nextafter(input, other, *, out=None):
    is_broadcast = tuple(input.shape) != tuple(other.shape)
    input, other = _broadcast(input, other)
    input, other, result_dtype = _prepare_inputs(input, other)

    if is_broadcast:
        out = _prepare_out(out, input.shape, result_dtype, input.device, like=input)
        if _vendor_triton.nextafter_broadcast_2d(input, other, out):
            return out
        # broadcast_tensors returns non-contiguous expanded views; materialize
        # them so the flattened kernel views are contiguous.
        input = input.contiguous()
        other = other.contiguous()

    out = _prepare_out(out, input.shape, result_dtype, input.device, like=input)
    in_view, other_view, out_view = _flatten_kernel_tensors(input, other, out)

    if _vendor_triton.nextafter_1d(in_view, other_view, out_view):
        return out

    if _vendor_triton.nextafter_half_iluvatar_1d(in_view, other_view, out_view):
        return out

    is_half = result_dtype in (torch.float16, torch.bfloat16)
    f32_roundtrip = (
        is_iluvatar_device(input)
        or (is_half and result_dtype in _HALF_BITCAST_BROKEN)
    ) and is_half

    block_size = ntops.kernels.nextafter.BLOCK_SIZE
    if result_dtype == torch.float64:
        block_size = 512

    def _make(use_f32_roundtrip):
        return _cached_make(
            ntops.kernels.nextafter.premake,
            in_view.ndim,
            dtype=_to_nt(result_dtype),
            iluvatar_half=use_f32_roundtrip,
            block_size=block_size,
            num_warps=_NUM_WARPS,
            num_stages=_NUM_STAGES,
            max_num_configs=1,
        )

    kernel = _make(f32_roundtrip)

    if is_half and not f32_roundtrip:
        try:
            kernel(in_view, other_view, out_view)
            return out
        except _compile_errors():
            # B 类兜底：半精 int16 bitcast kernel 在部分 Triton 上编译失败，
            # 退回 fp32 round-trip kernel 变体。
            _HALF_BITCAST_BROKEN.add(result_dtype)
            kernel = _make(True)

    kernel(in_view, other_view, out_view)
    return out


def _to_nt(torch_dtype):
    import ninetoothed

    mapping = {
        torch.float16: ninetoothed.float16,
        torch.bfloat16: ninetoothed.bfloat16,
        torch.float32: ninetoothed.float32,
        torch.float64: ninetoothed.float64,
    }
    return mapping.get(torch_dtype)
