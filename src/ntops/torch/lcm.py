import torch

import ntops
from ntops.torch import _vendor_triton
from ntops.torch.utils import _cached_make, _flatten_kernel_tensors, _prepare_out


_NUM_STAGES = 1


def _block_size_for(torch_dtype, device):
    if torch_dtype == torch.int64:
        return 32
    if _vendor_triton.is_metax_device(device):
        # MetaX has slower IDIV; reduce block_size to lower register pressure
        return 256
    return ntops.kernels.lcm.BLOCK_SIZE


def _num_warps_for(torch_dtype, device):
    if torch_dtype == torch.int64:
        return 1
    if _vendor_triton.is_metax_device(device):
        # MetaX: fewer warps to reduce IDIV contention per SM
        return 2
    return 4


def _broadcast(input, other):
    if hasattr(torch, "broadcast_tensors"):
        return torch.broadcast_tensors(input, other)
    return input, other


def _prepare_inputs(input, other):
    result_dtype = torch.result_type(input, other) if hasattr(torch, "result_type") else input.dtype
    if result_dtype == torch.bool or result_dtype.is_floating_point:
        raise NotImplementedError(f"lcm is not implemented for {result_dtype}")
    return input.to(result_dtype), other.to(result_dtype), result_dtype


def _prefer_native(input, result_dtype, is_broadcast):
    if not _vendor_triton.is_metax_device(input):
        return False
    if is_broadcast or result_dtype == torch.int8:
        return False
    return result_dtype in (torch.int16, torch.int32, torch.int64, torch.uint8)


def lcm(input, other, *, out=None):
    is_broadcast = tuple(input.shape) != tuple(other.shape)
    input, other = _broadcast(input, other)
    input, other, result_dtype = _prepare_inputs(input, other)

    if _prefer_native(input, result_dtype, is_broadcast):
        if out is None:
            return torch.lcm(input, other)
        out = _prepare_out(out, input.shape, result_dtype, input.device, like=input)
        torch.lcm(input, other, out=out)
        return out

    out = _prepare_out(out, input.shape, result_dtype, input.device, like=input)
    in_view, other_view, out_view = _flatten_kernel_tensors(input, other, out)

    if _vendor_triton.lcm_1d(in_view, other_view, out_view):
        return out

    kernel = _cached_make(
        ntops.kernels.lcm.premake,
        in_view.ndim,
        dtype=_to_nt(result_dtype),
        block_size=_block_size_for(result_dtype, in_view.device),
        num_warps=_num_warps_for(result_dtype, in_view.device),
        num_stages=_NUM_STAGES,
        max_num_configs=1,
    )

    kernel(in_view, other_view, out_view)

    return out


def _to_nt(torch_dtype):
    import ninetoothed
    mapping = {
        torch.uint8: ninetoothed.uint8,
        torch.int8: ninetoothed.int8,
        torch.int16: ninetoothed.int16,
        torch.int32: ninetoothed.int32,
        torch.int64: ninetoothed.int64,
    }
    return mapping.get(torch_dtype)
