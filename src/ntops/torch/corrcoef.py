import torch

import ntops
from ntops.torch import _vendor_triton
from ntops.torch._vendor_triton import is_iluvatar_device
from ntops.torch.utils import (
    _cached_make,
    _get_matmul_input_precision,
    _permute_tensor,
    _reshape_tensor,
)


def _to_nt(torch_dtype):
    import ninetoothed

    mapping = {
        torch.float32: ninetoothed.float32,
    }
    return mapping.get(torch_dtype)


def _normalize(matrix):
    nt_dtype = _to_nt(matrix.dtype)
    output = torch.empty_like(matrix)
    kernel = _cached_make(
        ntops.kernels.corrcoef.premake_normalize,
        2,
        dtype=nt_dtype,
        block_size=ntops.kernels.corrcoef.BLOCK_SIZE,
        num_warps=4,
        max_num_configs=1,
    )
    kernel(matrix, output)
    return output


def _gram(normalized):
    # Fixed-config 九齿 mm invocation. ntops.torch.mm autotunes (no config cap), which
    # costs ~257s of recompilation per shape on CoreX; pinning the block sizes with
    # max_num_configs=1 keeps the Gram matmul a real kernel without the autotune stall.
    rows = normalized.shape[0]
    other = _permute_tensor(normalized, (1, 0)).contiguous()
    out = torch.empty((rows, rows), dtype=normalized.dtype, device=normalized.device)
    kernel = _cached_make(
        ntops.kernels.mm.premake,
        dtype=_to_nt(normalized.dtype),
        block_size_m=64,
        block_size_n=64,
        block_size_k=64,
        num_warps=4,
        max_num_configs=1,
    )
    kernel(normalized, other, out, _get_matmul_input_precision())
    return out


def corrcoef(input):
    # B-class fallback: Triton's dot (the NineToothed mm kernel) supports only
    # fp16/bf16/fp32, so f64 corrcoef keeps native — the Gram matmul cannot compile
    # for double on any platform (capability gap, not a performance route).
    if input.dtype != torch.float32:
        # CoreX cuBLAS has no f64 gemm, so even torch.corrcoef crashes on the GPU for
        # double; CPU is the only correct path there. Other devices keep native.
        if input.dtype == torch.float64 and is_iluvatar_device(input):
            # ntops: capability-fallback - Triton dot/NineToothed mm cannot do f64 GEMM on CoreX.
            return torch.corrcoef(input.cpu()).to(input.device)
        # ntops: capability-fallback - current corrcoef kernel is f32-only; f64 needs a f64 GEMM path.
        return torch.corrcoef(input)

    if input.ndim < 2:
        matrix = _reshape_tensor(input, (1, -1))
    else:
        matrix = input

    # Degenerate single-observation input: correlation is undefined (native returns
    # NaN); keep native for correctness, not performance.
    if matrix.shape[-1] < 2:
        # ntops: capability-fallback - degenerate corrcoef NaN fill kernel is not implemented.
        return torch.corrcoef(input)

    # A single observation row makes corrcoef a scalar (1.0, or NaN for zero
    # variance); one split-K reduction beats the full normalize+gram launches.
    if matrix.shape[0] == 1:
        scalar = _vendor_triton.corrcoef_single_row(_reshape_tensor(matrix, (-1,)).contiguous())
        if scalar is not None:
            return scalar if input.ndim < 2 else _reshape_tensor(scalar, (1, 1))

    # Few-row / very-wide matrices are launch-bound: a single-pass fused kernel that
    # reads the input once (stats + raw Gram together) beats the two-stage
    # normalize+gram path. Returns None outside that regime to keep the proven path.
    fused = _vendor_triton.corrcoef_fused(matrix.contiguous())
    if fused is not None:
        result = fused
        if input.ndim < 2:
            return _reshape_tensor(result, ())
        return result

    normalized = _normalize(matrix.contiguous())
    # Folding the row L2-norm into `normalized` makes the Gram matrix the correlation
    # matrix directly. For few-row/wide matrices the split-K vendor kernel parallelizes
    # the huge K reduction the tiled mm would serialize; otherwise fall back to the
    # autotuning ntops.torch.mm (NVIDIA/MetaX) or pinned-config _gram (Iluvatar, where
    # the autotuner recompiles ~257s per shape).
    gram = _vendor_triton.gram_fast(normalized)
    if gram is None:
        if is_iluvatar_device(input):
            gram = _gram(normalized)
        else:
            gram = ntops.torch.mm(normalized, _permute_tensor(normalized, (1, 0)).contiguous())
    result = gram.clamp(-1.0, 1.0) if hasattr(gram, "clamp") else gram

    if input.ndim < 2:
        return _reshape_tensor(result, ())
    return result
