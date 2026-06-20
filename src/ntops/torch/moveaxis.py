import functools

import torch

import ntops
from ntops.torch import _vendor_triton
from ntops.torch.utils import _cached_make


def _compute_permutation(ndim, source, destination):
    """Compute the axis permutation for moveaxis.

    Returns a tuple perm such that perm[i] is the index of the original axis
    that becomes the new axis at position i.
    """
    # Normalize negative indices
    src = tuple(
        s if s >= 0 else s + ndim
        for s in (source if isinstance(source, tuple) else (source,))
    )
    dst = tuple(
        d if d >= 0 else d + ndim
        for d in (
            destination if isinstance(destination, tuple) else (destination,)
        )
    )

    # Validate ranges
    for s in src:
        if s < 0 or s >= ndim:
            raise IndexError(
                f"Dimension out of range (expected to be in range of "
                f"[{-ndim}, {ndim - 1}], but got {s})"
            )
    for d in dst:
        if d < 0 or d >= ndim:
            raise IndexError(
                f"Dimension out of range (expected to be in range of "
                f"[{-ndim}, {ndim - 1}], but got {d})"
            )

    # Build permutation: fill destination positions, then remaining
    result = [-1] * ndim
    for s, d in zip(src, dst):
        result[d] = s

    remaining = [i for i in range(ndim) if i not in src]
    remaining_iter = iter(remaining)
    for i in range(ndim):
        if result[i] == -1:
            result[i] = next(remaining_iter)

    return tuple(result)


def _kernel_config(ndim, perm):
    perm = tuple(perm)
    if perm == (2, 1, 0):
        return 256, 1
    if ndim == 2 and perm == (1, 0):
        return 4096, 1
    if ndim == 3 and perm in ((2, 0, 1), (0, 2, 1)):
        return 2048, 8
    if ndim == 3 and perm == (1, 0, 2):
        return 256, 4
    if ndim == 4 and perm == (0, 2, 1, 3):
        return 256, 4
    if ndim == 4:
        return 1024, 4
    return ntops.kernels.moveaxis.BLOCK_SIZE, 1


@functools.cache
def _get_kernel(ndim, perm, block_size, num_warps):
    return _cached_make(
        ntops.kernels.moveaxis.premake,
        ndim,
        perm=perm,
        block_size=block_size,
        num_warps=num_warps,
        max_num_configs=1,
    )


def moveaxis(input, source, destination):
    """Move axis from source to destination position.

    Returns a contiguous tensor with the requested axis order. PyTorch's
    torch.moveaxis returns a view, but ntops uses the NineToothed data
    movement kernel here so benchmarked contiguous output does not fall
    back to PyTorch's copy path.
    """
    ndim = input.ndim
    perm = _compute_permutation(ndim, source, destination)
    output_shape = tuple(input.shape[axis] for axis in perm)
    output = torch.empty(output_shape, dtype=input.dtype, device=input.device)
    if input.numel() == 0:
        return output
    if _vendor_triton.moveaxis_fast_path(input, output, perm):
        return output
    block_size, num_warps = _kernel_config(ndim, perm)
    _get_kernel(ndim, perm, block_size, num_warps)(input, output)
    return output
