import functools

import torch

import ntops
from ntops.torch import _vendor_triton
from ntops.torch.utils import _cached_make, _flatten_kernel_tensors, _prepare_out


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


def _prefer_native(input):
    if input.dtype == torch.float64 and _is_iluvatar_device(input.device.index):
        return True
    if _is_metax_device(input.device.index):
        return input.dtype in (torch.float16, torch.bfloat16) or not input.is_contiguous()
    return False


def frac(input, *, out=None):
    input = _promote_unary_input(input)
    half = input.dtype == torch.float16
    bfloat16 = input.dtype == torch.bfloat16

    if _prefer_native(input):
        if out is None:
            return torch.frac(input)
        out = _prepare_out(out, input.shape, input.dtype, input.device, like=input)
        torch.frac(input, out=out)
        return out

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
