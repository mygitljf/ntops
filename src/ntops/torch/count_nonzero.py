import torch

import ntops
from ntops.torch import _vendor_triton
from ntops.torch.utils import _cached_make, _reshape_tensor


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
        torch.uint8: ninetoothed.uint8,
    }
    return mapping.get(torch_dtype)


def _reduce_last(flat, nt_dtype):
    output = torch.empty(flat.shape, dtype=torch.int64, device=flat.device)
    kernel = _cached_make(
        ntops.kernels.count_nonzero.premake,
        2,
        dtype=nt_dtype,
        block_size=ntops.kernels.count_nonzero.BLOCK_SIZE,
        num_warps=4,
        max_num_configs=1,
    )
    kernel(flat, output)
    return _reshape_tensor(output.narrow(1, 0, 1), (-1,))


def _global_count(input, nt_dtype):
    # A single-row reduction serializes all work in one program; reshape the flat
    # buffer to (rows, cols) so the kernel parallelizes over rows, then finish the
    # partial-sum chain with a vendor Triton reduction kernel.
    n = input.numel()
    flat = _reshape_tensor(input, (-1,))
    cols = 4096
    if n % cols == 0 and n // cols > 1:
        partials = _reduce_last(_reshape_tensor(flat, (n // cols, cols)), nt_dtype)
        total = _vendor_triton.sum_strided(
            partials,
            partials.shape[0],
            stride=partials.stride(0),
            dtype=torch.int64,
        )
        if total is None:
            raise NotImplementedError("count_nonzero final reduction requires a vendor Triton kernel")
        return total
    return _reshape_tensor(_reduce_last(_reshape_tensor(flat, (1, -1)), nt_dtype), ())


def count_nonzero(input, dim=None):
    # bool has no NineToothed dtype; reinterpret as int8 (zero-copy, same 1-byte
    # storage and nonzero-ness) instead of widening to int32, which would cost a
    # full-tensor 4x copy before the reduction even starts.
    if input.dtype == torch.bool:
        input = input.view(torch.int8)

    nt_dtype = _to_nt(input.dtype)
    if nt_dtype is None:
        raise NotImplementedError(f"count_nonzero kernel does not support {input.dtype}")
    if input.numel() == 0:
        if dim is None:
            return torch.empty((), dtype=torch.int64, device=input.device).fill_(0)
        dims = (dim,) if isinstance(dim, int) else tuple(dim)
        dims = tuple(d % input.ndim for d in dims)
        kept_shape = tuple(input.shape[d] for d in range(input.ndim) if d not in dims)
        return torch.empty(kept_shape, dtype=torch.int64, device=input.device).fill_(0)

    if dim is None:
        return _global_count(input.contiguous(), nt_dtype)

    dims = (dim,) if isinstance(dim, int) else tuple(dim)
    ndim = input.ndim
    dims = tuple(d % ndim for d in dims)
    kept = tuple(d for d in range(ndim) if d not in dims)

    permuted = input.permute(kept + dims)
    kept_shape = tuple(input.shape[d] for d in kept)
    reduce_numel = 1
    for d in dims:
        reduce_numel *= input.shape[d]
    kept_numel = 1
    for d in kept:
        kept_numel *= input.shape[d]

    # Route to the coalesced vendor kernel by layout (zero-copy where possible):
    #   - permuted contiguous, or single reduce/keep dim: rows=kept, last axis=reduced
    #     -> row kernel reads the contiguous reduced axis coalesced.
    #   - reduce the leading contiguous block, keep trailing dim (e.g. dim0-2D,
    #     dim(0,1)-3D): a (reduce_numel, kept_numel) view reduces axis 0 with the
    #     contiguous kept axis as the coalesced inner tile -> column kernel; this
    #     avoids the full-tensor transpose copy contiguous() would otherwise force.
    counts = None
    if permuted.is_contiguous():
        counts = _vendor_triton.count_nonzero_reduce(_reshape_tensor(permuted, (-1, reduce_numel)))
    elif len(dims) == 1 and len(kept) == 1:
        counts = _vendor_triton.count_nonzero_reduce(permuted)
    elif input.is_contiguous() and list(dims) == list(range(len(dims))):
        counts = _vendor_triton.count_nonzero_col_reduce(
            _reshape_tensor(input, (reduce_numel, kept_numel))
        )

    if counts is None:
        if permuted.is_contiguous():
            flat = _reshape_tensor(permuted, (-1, reduce_numel))
        elif len(dims) == 1 and len(kept) == 1:
            flat = permuted
        else:
            flat = _reshape_tensor(permuted.contiguous(), (-1, reduce_numel))
        counts = _vendor_triton.count_nonzero_reduce(flat)
        if counts is None:
            counts = _reduce_last(flat, nt_dtype)
    return _reshape_tensor(counts, kept_shape)
