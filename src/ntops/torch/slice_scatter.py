import torch

import ntops
from ntops.torch import _vendor_triton
from ntops.torch.utils import _cached_make


def _to_nt(torch_dtype):
    import ninetoothed

    mapping = {
        torch.float16: ninetoothed.float16,
        torch.bfloat16: ninetoothed.bfloat16,
        torch.float32: ninetoothed.float32,
        torch.float64: ninetoothed.float64,
        torch.int16: ninetoothed.int16,
        torch.int32: ninetoothed.int32,
        torch.int64: ninetoothed.int64,
    }
    return mapping.get(torch_dtype)


def slice_scatter(input, src, dim=0, start=None, end=None, step=1):
    nt_dtype = _to_nt(input.dtype)

    if nt_dtype is None:
        raise NotImplementedError(f"slice_scatter kernel does not support {input.dtype}")
    if not input.is_contiguous():
        input = input.contiguous()
    if not src.is_contiguous():
        src = src.contiguous()

    dim = dim % input.ndim
    dim_size = input.shape[dim]
    lo = 0 if start is None else (start + dim_size if start < 0 else start)
    hi = dim_size if end is None else (end + dim_size if end < 0 else end)
    lo = max(0, min(lo, dim_size))
    hi = max(lo, min(hi, dim_size))
    length = len(range(lo, hi, step))
    expected_shape = list(output_shape := input.shape)
    expected_shape[dim] = length
    if tuple(src.shape) != tuple(expected_shape):
        raise RuntimeError(
            "slice_scatter(): expected src to have shape "
            f"{tuple(expected_shape)}, got {tuple(src.shape)}"
        )

    output = torch.empty_like(input)

    # Single coalesced pass: each output element is selected from src (when its
    # coordinate along dim lands on a step boundary inside the slice) or from input.
    # Halves the write traffic of the prior whole-input-copy + strided-view-copy path.
    if _vendor_triton.slice_scatter_into(input, src, output, dim, lo, step, length):
        return output

    # Fallback: clone input, then copy src into the narrowed view (two identity copies).
    copy_kernel = _cached_make(
        ntops.kernels.slice_scatter.premake,
        input.ndim,
        dtype=nt_dtype,
        block_size=ntops.kernels.slice_scatter.BLOCK_SIZE,
        num_warps=4,
        max_num_configs=1,
    )
    copy_kernel(input, output)

    if length > 0:
        shape = list(output_shape)
        shape[dim] = length
        stride = list(output.stride())
        stride[dim] *= step
        offset = output.storage_offset() + lo * output.stride(dim)
        view = output.as_strided(tuple(shape), tuple(stride), offset)
        copy_kernel(src, view)

    return output
