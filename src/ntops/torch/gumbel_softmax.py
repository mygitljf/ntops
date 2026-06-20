import random

import torch

import ntops
from ntops.torch import _vendor_triton
from ntops.torch.utils import _cached_make

_rt = __import__("torch")


def _to_nt(torch_dtype):
    import ninetoothed

    mapping = {
        _rt.float16: ninetoothed.float16,
        _rt.bfloat16: ninetoothed.bfloat16,
        _rt.float32: ninetoothed.float32,
        _rt.float64: ninetoothed.float64,
    }
    result = mapping.get(torch_dtype)
    if result is not None:
        return result
    # infinicore context: torch_dtype is an infinicore.dtype — match by name suffix.
    name = str(torch_dtype).rpartition(".")[2]
    name_map = {
        "float16": ninetoothed.float16,
        "bfloat16": ninetoothed.bfloat16,
        "float32": ninetoothed.float32,
        "float64": ninetoothed.float64,
    }
    return name_map.get(name)


def _soft_kernel(logits, tau):
    cols = logits.shape[-1]
    flat = logits.reshape(logits.numel() // cols, cols)
    output = torch.empty_like(flat)
    seed = random.randrange(0, 2**31)

    # Vendor single-pass per-row softmax beats the streaming NineToothed kernel
    # for the small/mid column counts that dominate gumbel use (one tile per row).
    if _vendor_triton.gumbel_softmax_soft(flat, output, seed, float(tau)):
        return output.reshape(logits.shape)

    block_size = 1
    while block_size < cols:
        block_size *= 2
    kernel = _cached_make(
        ntops.kernels.gumbel_softmax.premake,
        flat.shape[0],
        cols,
        dtype=_to_nt(logits.dtype),
        block_size=block_size,
        num_warps=4,
        max_num_configs=1,
    )
    kernel(flat, seed, float(tau), output)
    return output.reshape(logits.shape)


def _kernel_eligible(logits, dim):
    return (
        _to_nt(logits.dtype) is not None
        and logits.ndim > 0
        and 0 < logits.shape[dim] <= ntops.kernels.gumbel_softmax.MAX_BLOCK_SIZE
    )


def gumbel_softmax(logits, tau=1.0, hard=False, eps=1e-10, dim=-1):
    dim = dim % logits.ndim
    if not _kernel_eligible(logits, dim):
        if _to_nt(logits.dtype) is None:
            raise NotImplementedError(f"gumbel_softmax kernel does not support {logits.dtype}")
        raise NotImplementedError("gumbel_softmax row width exceeds kernel MAX_BLOCK_SIZE")

    moved = False
    if dim != logits.ndim - 1:
        logits = logits.movedim(dim, -1).contiguous()
        moved = True
    if not logits.is_contiguous():
        logits = logits.contiguous()

    y_soft = _soft_kernel(logits, tau)

    if not hard:
        return y_soft.movedim(-1, dim) if moved else y_soft

    y_hard = torch.empty_like(y_soft)
    if not _vendor_triton.gumbel_hard_one_hot(
        y_soft.reshape(y_soft.numel() // y_soft.shape[-1], y_soft.shape[-1]),
        y_hard.reshape(y_hard.numel() // y_hard.shape[-1], y_hard.shape[-1]),
    ):
        raise NotImplementedError("gumbel_softmax hard path requires a vendor Triton one-hot kernel")
    result = y_hard - y_soft.detach() + y_soft
    return result.movedim(-1, dim) if moved else result
