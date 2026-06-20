import torch

import ntops
from ntops.torch import _vendor_triton
from ntops.torch.utils import _cached_make, _reshape_tensor, _to_nt


def pixel_unshuffle(input, downscale_factor):
    r = downscale_factor

    if input.ndim != 4:
        raise NotImplementedError("pixel_unshuffle kernel requires 4D input")
    if input.shape[2] % r != 0 or input.shape[3] % r != 0:
        raise RuntimeError("pixel_unshuffle expects spatial dimensions divisible by downscale_factor")
    if _to_nt(input.dtype) is None:
        raise NotImplementedError(f"pixel_unshuffle kernel does not support {input.dtype}")

    n, c, hr, wr = input.shape
    h, w = hr // r, wr // r
    output = torch.empty((n, c * r * r, h, w), dtype=input.dtype, device=input.device)

    if _vendor_triton.pixel_unshuffle_fast_path(input, output, r):
        return output

    if not input.is_contiguous():
        input = input.contiguous()

    # NineToothed fallback: view as (N, C, H, r, W, r), permute to
    # (N, C, r, r, H, W) at the meta level, and identity-copy.
    view6 = _reshape_tensor(input, (n, c, h, r, w, r))
    output6 = torch.empty((n, c, r, r, h, w), dtype=input.dtype, device=input.device)

    kernel = _cached_make(
        ntops.kernels.pixel_unshuffle.premake,
        perm=(0, 1, 3, 5, 2, 4),
        dtype=_to_nt(input.dtype),
        block_size=ntops.kernels.pixel_unshuffle.BLOCK_SIZE,
        num_warps=4,
        max_num_configs=1,
    )
    kernel(view6, output6)
    return _reshape_tensor(output6, (n, c * r * r, h, w))
