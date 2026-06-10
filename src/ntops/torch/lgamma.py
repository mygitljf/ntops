import torch

import ntops
from ntops.torch import _vendor_triton
from ntops.torch.utils import _cached_make, _flatten_kernel_tensors, _prepare_out


_LARGE_NUMEL_THRESHOLD = 2_000_000


def lgamma(input, *, out=None):
    input = _promote_input(input)
    out = _prepare_out(out, input.shape, input.dtype, input.device, like=input)
    in_view, out_view = _flatten_kernel_tensors(input, out)

    if _vendor_triton.lgamma_1d(in_view, out_view):
        return out

    block_size = ntops.kernels.lgamma.BLOCK_SIZE
    if input.dtype == torch.float64:
        block_size = 512

    if input.numel() >= _LARGE_NUMEL_THRESHOLD:
        kernel = _cached_make(
            ntops.kernels.lgamma.premake,
            in_view.ndim,
            dtype=_to_nt(input.dtype),
            block_size=block_size,
            num_warps=4,
            num_stages=5,
            max_num_configs=1,
        )
    else:
        kernel = _cached_make(
            ntops.kernels.lgamma.premake,
            in_view.ndim,
            dtype=_to_nt(input.dtype),
            max_num_configs=1,
        )

    kernel(in_view, out_view)

    return out


def _promote_input(input):
    if hasattr(torch, "is_floating_point") and not torch.is_floating_point(input):
        return input.to(torch.float32)
    return input


def _to_nt(torch_dtype):
    import ninetoothed

    mapping = {
        torch.float16: ninetoothed.float16,
        torch.bfloat16: ninetoothed.bfloat16,
        torch.float32: ninetoothed.float32,
        torch.float64: ninetoothed.float64,
    }
    return mapping.get(torch_dtype)
