import torch
import torch as _orig_torch

import ntops
from ntops.torch import _vendor_triton
from ntops.torch.utils import _cached_make, _cast_dtype, _flatten_kernel_tensors, _is_dtype, _prepare_out, _reshape_tensor, _to_nt


def _mse_none(input, target, result_dtype):
    out = _prepare_out(None, input.shape, result_dtype, input.device, like=input)
    in_view, target_view, out_view = _flatten_kernel_tensors(input, target, out)
    if _vendor_triton.mse_none(in_view, target_view, out_view):
        return out

    block_size = ntops.kernels.mse_loss.BLOCK_SIZE
    if _is_dtype(result_dtype, "float64"):
        block_size = 512

    kernel = _cached_make(
        ntops.kernels.mse_loss.premake,
        in_view.ndim,
        dtype=_to_nt(result_dtype),
        block_size=block_size,
        num_warps=4,
        max_num_configs=1,
    )

    kernel(in_view, target_view, out_view)
    return out


def _mse_total(input, target, result_dtype):
    # Single-pass fused squared-diff + reduction when eligible; otherwise the
    # 2-stage row-reduce + strided-sum path.
    fused = _vendor_triton.mse_sum_fused(input, target)
    if fused is not None:
        return fused

    n = input.numel()
    flat_in = _reshape_tensor(input, (-1,))
    flat_tgt = _reshape_tensor(target, (-1,))
    cols = 4096
    partials = None
    if n % cols == 0 and n // cols > 1:
        partials = _vendor_triton.mse_row_sums(flat_in, flat_tgt, cols)
    if partials is not None:
        total = _vendor_triton.sum_strided(
            partials,
            partials.shape[0],
            stride=1,
            dtype=_orig_torch.float32,
        )
        if total is None:
            raise NotImplementedError("mse_loss final reduction requires a vendor Triton kernel")
        return total

    if n % cols == 0 and n // cols > 1:
        rows = n // cols
        flat_in = _reshape_tensor(flat_in, (rows, cols))
        flat_tgt = _reshape_tensor(flat_tgt, (rows, cols))
    else:
        flat_in = flat_in[None, :]
        flat_tgt = flat_tgt[None, :]

    partials = torch.empty(flat_in.shape, dtype=torch.float32, device=input.device)
    kernel = _cached_make(
        ntops.kernels.mse_loss.premake_sum,
        2,
        dtype=_to_nt(result_dtype),
        block_size=ntops.kernels.mse_loss.REDUCE_BLOCK_SIZE,
        num_warps=4,
        max_num_configs=1,
    )
    kernel(flat_in, flat_tgt, partials)
    total = _vendor_triton.sum_strided(
        partials,
        partials.shape[0],
        stride=partials.stride(0),
        dtype=_orig_torch.float32,
    )
    if total is None:
        raise NotImplementedError("mse_loss final reduction requires a vendor Triton kernel")
    return total


def mse_loss(input, target, reduction="mean"):
    result_dtype = (
        torch.result_type(input, target) if hasattr(torch, "result_type") else input.dtype
    )

    if _to_nt(result_dtype) is None:
        raise NotImplementedError(f"mse_loss kernel does not support {result_dtype}")
    if input.numel() == 0 or target.numel() == 0:
        raise NotImplementedError("mse_loss empty-input semantics require a dedicated kernel")

    input = _cast_dtype(input, result_dtype)
    target = _cast_dtype(target, result_dtype)

    if tuple(input.shape) != tuple(target.shape):
        # Materialize the broadcast before the kernel: expand to the common
        # shape and make contiguous so the flat elementwise/reduction kernels
        # see plain dense buffers.
        input, target = torch.broadcast_tensors(input, target)
        input = input.contiguous()
        target = target.contiguous()

    if reduction == "none":
        return _mse_none(input, target, result_dtype)

    total = _mse_total(input.contiguous(), target.contiguous(), result_dtype)

    if reduction == "sum":
        result = total
    else:
        result = total / input.numel()

    return _cast_dtype(result, result_dtype)
