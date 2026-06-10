import torch

import ntops
from ntops.torch._vendor_triton import is_iluvatar_device, is_metax_device
from ntops.torch.utils import _cached_make, _flatten_kernel_tensors, _prepare_out


def _broadcast(input, other):
    if hasattr(torch, "broadcast_tensors"):
        return torch.broadcast_tensors(input, other)
    return input, other


def _prepare_inputs(input, other):
    result_dtype = torch.result_type(input, other) if hasattr(torch, "result_type") else input.dtype
    if not result_dtype.is_floating_point:
        result_dtype = torch.float32
    return input.to(result_dtype), other.to(result_dtype), result_dtype


def _prefer_native(input, result_dtype, is_broadcast):
    if is_iluvatar_device(input):
        return result_dtype == torch.float64 or is_broadcast
    if is_metax_device(input):
        return (
            result_dtype in (torch.float64, torch.float16, torch.bfloat16)
            or is_broadcast
            or not input.is_contiguous()
        )
    return False


def copysign(input, other, *, out=None):
    is_broadcast = tuple(input.shape) != tuple(other.shape)
    input, other = _broadcast(input, other)
    input, other, result_dtype = _prepare_inputs(input, other)

    if _prefer_native(input, result_dtype, is_broadcast):
        if out is None:
            return torch.copysign(input, other)
        out = _prepare_out(out, input.shape, result_dtype, input.device, like=input)
        torch.copysign(input, other, out=out)
        return out

    out = _prepare_out(out, input.shape, result_dtype, input.device, like=input)
    in_view, other_view, out_view = _flatten_kernel_tensors(input, other, out)

    iluvatar_half = (
        is_iluvatar_device(input)
        and result_dtype in (torch.float16, torch.bfloat16)
    )

    block_size = ntops.kernels.copysign.BLOCK_SIZE
    if result_dtype == torch.float64:
        block_size = 512

    kernel = _cached_make(
        ntops.kernels.copysign.premake,
        in_view.ndim,
        dtype=_to_nt(result_dtype),
        iluvatar_half=iluvatar_half,
        block_size=block_size,
        num_warps=4,
        max_num_configs=1,
    )

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
