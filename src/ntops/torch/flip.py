import torch

import ntops
from ntops.torch import _vendor_triton
from ntops.torch.utils import _cached_make, _to_nt


def _normalize_dims(dims, ndim):
    if isinstance(dims, int):
        dims = (dims,)
    return tuple(sorted({d % ndim for d in dims}))


def _flip_kernel(input, dims):
    # Pure NineToothed path: the kernel reverses any subset of dims in one pass
    # (the arrangement applies negative-step slices to the output meta-tensor
    # and identity-copies), with the trailing dim tiled in blocks so large rows
    # work too.
    flip_mask = tuple(d in dims for d in range(input.ndim))
    output = torch.empty_like(input)
    kernel = _cached_make(
        ntops.kernels.flip.premake,
        input.ndim,
        flip_mask,
        dtype=_to_nt(input.dtype),
        block_size=ntops.kernels.flip.BLOCK_SIZE,
        num_warps=4,
        max_num_configs=1,
    )
    kernel(input, output)
    return output


def _can_kernel(input, dims):
    if _to_nt(input.dtype) is None:
        return False
    if input.ndim == 0 or input.numel() == 0:
        return False
    return True


def flip(input, dims):
    dims = _normalize_dims(dims, input.ndim)
    if not _can_kernel(input, dims):
        if _to_nt(input.dtype) is None:
            raise NotImplementedError(f"flip kernel does not support {input.dtype}")
        return torch.empty_like(input)

    output = torch.empty(input.shape, dtype=input.dtype, device=input.device)
    if _vendor_triton.flip_into(input, output, dims):
        return output

    if not input.is_contiguous():
        output = torch.empty(tuple(input.shape), dtype=input.dtype, device=input.device)
        if _vendor_triton.flip_2d_strided(input, output, dims):
            return output
        input = input.contiguous()
    return _flip_kernel(input, dims)
