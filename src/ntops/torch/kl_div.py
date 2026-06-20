import torch

import ntops
from ntops.torch import _vendor_triton
from ntops.torch.utils import (
    _cached_make,
    _flatten_kernel_tensors,
    _prepare_out,
    _reshape_tensor,
)


def _kl_none(input, target, log_target):
    out = _prepare_out(None, input.shape, input.dtype, input.device, like=input)
    in_view, target_view, out_view = _flatten_kernel_tensors(input, target, out)

    kernel = _cached_make(
        ntops.kernels.kl_div.premake,
        in_view.ndim,
        dtype=_to_nt(input.dtype),
        log_target=log_target,
        block_size=ntops.kernels.kl_div.BLOCK_SIZE,
        num_warps=4,
        max_num_configs=1,
    )

    kernel(in_view, target_view, out_view)
    return out


def _kl_total(input, target, log_target, output_dtype, scale=None):
    # Fused term + two-stage row reduction: reshape the flat buffer to (rows, cols)
    # so the reduction kernel parallelizes across rows, then finish the partial-sum
    # chain with a vendor Triton reduction kernel.
    n = input.numel()
    flat_in = _reshape_tensor(input, (-1,))
    flat_tgt = _reshape_tensor(target, (-1,))
    cols = 4096
    if n % cols == 0 and n // cols > 1:
        rows = n // cols
        flat_in = _reshape_tensor(flat_in, (rows, cols))
        flat_tgt = _reshape_tensor(flat_tgt, (rows, cols))
    else:
        flat_in = _reshape_tensor(flat_in, (1, -1))
        flat_tgt = _reshape_tensor(flat_tgt, (1, -1))

    acc_dtype = torch.float64 if input.dtype == torch.float64 else torch.float32
    partials = torch.empty(flat_in.shape, dtype=acc_dtype, device=input.device)
    kernel = _cached_make(
        ntops.kernels.kl_div.premake_sum,
        2,
        dtype=_to_nt(input.dtype),
        log_target=log_target,
        block_size=ntops.kernels.kl_div.REDUCE_BLOCK_SIZE,
        num_warps=4,
        max_num_configs=1,
    )
    kernel(flat_in, flat_tgt, partials)
    total = _vendor_triton.sum_strided(
        partials,
        partials.shape[0],
        stride=partials.stride(0),
        dtype=output_dtype,
        scale=scale,
    )
    if total is None:
        raise NotImplementedError("kl_div final reduction requires a vendor Triton kernel")
    return total


def kl_div(input, target, reduction="mean", log_target=False):
    if _to_nt(input.dtype) is None:
        raise NotImplementedError(f"kl_div kernel does not support {input.dtype}")

    if input.dtype == torch.float64:
        return _kl_div_kernel(input, target, reduction, log_target)

    return _kl_div_kernel(input, target, reduction, log_target)


def _kl_div_kernel(input, target, reduction, log_target):
    if reduction == "none":
        return _kl_none(input, target, log_target)

    if reduction == "sum":
        scale = None
    elif reduction == "batchmean":
        scale = 1.0 / float(input.shape[0])
    else:
        scale = 1.0 / float(input.numel())

    return _kl_total(input.contiguous(), target.contiguous(), log_target, input.dtype, scale)


def _to_nt(torch_dtype):
    import ninetoothed

    mapping = {
        torch.float16: ninetoothed.float16,
        torch.bfloat16: ninetoothed.bfloat16,
        torch.float32: ninetoothed.float32,
        torch.float64: ninetoothed.float64,
    }
    return mapping.get(torch_dtype)
