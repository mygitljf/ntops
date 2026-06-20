import torch

from ntops.torch import _vendor_triton
from ntops.torch.flip import _can_kernel, _flip_kernel


def fliplr(input):
    # fliplr reverses dim=1; the generalized flip kernel handles any dim (not
    # just the trailing one), so every eligible rank-2+ input goes through it.
    if input.ndim < 2:
        raise RuntimeError("fliplr requires a tensor with at least 2 dimensions")
    if not _can_kernel(input, (1,)):
        raise NotImplementedError(f"fliplr kernel does not support {input.dtype}")

    output = torch.empty(input.shape, dtype=input.dtype, device=input.device)
    if _vendor_triton.flip_into(input, output, (1,)):
        return output

    if not input.is_contiguous():
        input = input.contiguous()
    return _flip_kernel(input, (1,))
