import torch

import ntops
from ntops.torch import _vendor_triton
from ntops.torch.utils import _cached_make, _flatten_kernel_tensors, _prepare_out


_NUM_WARPS = 4
_NUM_STAGES = 1


def _promote_input(input):
    if hasattr(torch, "is_floating_point") and not torch.is_floating_point(input):
        return input.to(torch.float32)
    return input


def _route_native(input, out):
    if out is None:
        return torch.rad2deg(input)
    out = _prepare_out(out, input.shape, input.dtype, input.device, like=input)
    torch.rad2deg(input, out=out)
    return out


def _prefer_native(input):
    if _vendor_triton.is_iluvatar_device(input):
        return input.dtype == torch.float64
    if _vendor_triton.is_metax_device(input):
        return (
            input.dtype in (torch.float64, torch.float16, torch.bfloat16)
            or not input.is_contiguous()
        )
    return False


def rad2deg(input, *, out=None):
    input = _promote_input(input)

    if _prefer_native(input):
        return _route_native(input, out)

    out = _prepare_out(out, input.shape, input.dtype, input.device, like=input)
    in_view, out_view = _flatten_kernel_tensors(input, out)

    if _vendor_triton.rad2deg_1d(in_view, out_view):
        return out

    block_size = ntops.kernels.rad2deg.BLOCK_SIZE
    if input.dtype == torch.float64:
        block_size = 512

    kernel = _cached_make(
        ntops.kernels.rad2deg.premake,
        in_view.ndim,
        dtype=_to_nt(input.dtype),
        block_size=block_size,
        num_warps=_NUM_WARPS,
        num_stages=_NUM_STAGES,
        max_num_configs=1,
    )

    kernel(in_view, out_view)

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
