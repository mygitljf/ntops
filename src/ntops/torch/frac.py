import functools

import torch

import ntops
from ntops.torch import _vendor_triton
from ntops.torch.utils import _cached_make, _flatten_kernel_tensors, _prepare_out


def _promote_unary_input(input):
    if hasattr(torch, "is_floating_point") and not torch.is_floating_point(input):
        return input.to(torch.float32)
    return input


@functools.cache
def _get_kernel_1d(half=False, bfloat16=False):
    return _cached_make(
        ntops.kernels.frac.premake,
        1,
        half,
        bfloat16,
        block_size=ntops.kernels.frac.BLOCK_SIZE,
        num_warps=4,
        max_num_configs=1,
    )


def frac(input, *, out=None):
    input = _promote_unary_input(input)
    half = input.dtype == torch.float16
    bfloat16 = input.dtype == torch.bfloat16

    # ntops: do NOT pre-materialize a contiguous copy. _prepare_out(like=input) keeps
    # input's physical layout so _flatten_kernel_tensors runs one coalesced 1D pass.
    out = _prepare_out(out, input.shape, input.dtype, input.device, like=input)
    kernel_input, kernel_out = _flatten_kernel_tensors(input, out)

    if _vendor_triton.frac_1d(kernel_input, kernel_out):
        return out

    kernel = _cached_make(
        ntops.kernels.frac.premake,
        kernel_input.ndim,
        half,
        bfloat16,
        block_size=ntops.kernels.frac.BLOCK_SIZE,
        num_warps=4,
        max_num_configs=1,
    )
    kernel(kernel_input, kernel_out)
    return out
