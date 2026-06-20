import torch

import ntops
from ntops.torch.utils import _cached_make


def _to_nt(torch_dtype):
    import ninetoothed

    mapping = {
        torch.float16: ninetoothed.float16,
        torch.bfloat16: ninetoothed.bfloat16,
        torch.float32: ninetoothed.float32,
        torch.float64: ninetoothed.float64,
        torch.int8: ninetoothed.int8,
        torch.int16: ninetoothed.int16,
        torch.int32: ninetoothed.int32,
        torch.int64: ninetoothed.int64,
    }
    return mapping.get(torch_dtype)


def narrow(input, dim, start, length):
    # C-class: torch.narrow natively returns a zero-copy view (O(1) metadata), while
    # this NineToothed copy kernel materializes the output in O(n); the kernel path
    # is known to be <1.0x against native by construction. It is kept as the pure
    # kernel dispatch per the no-performance-routing policy.
    dim = dim % input.ndim
    dim_size = input.shape[dim]
    if start < 0:
        start += dim_size
    if start < 0 or start > dim_size:
        raise RuntimeError("start out of range")
    if length < 0:
        raise RuntimeError("narrow(): length must be non-negative")
    if start + length > dim_size:
        raise RuntimeError("start + length exceeds dimension size")
    shape = list(input.shape)
    shape[dim] = length
    offset = input.storage_offset() + start * input.stride(dim)
    view = input.as_strided(tuple(shape), input.stride(), offset)

    nt_dtype = _to_nt(input.dtype)
    if nt_dtype is None or view.numel() == 0:
        return view.contiguous()

    output = torch.empty(view.shape, dtype=view.dtype, device=view.device)

    kernel = _cached_make(
        ntops.kernels.narrow.premake,
        view.ndim,
        dtype=nt_dtype,
        block_size=ntops.kernels.narrow.BLOCK_SIZE,
        num_warps=4,
        max_num_configs=1,
    )
    kernel(view, output)
    return output
