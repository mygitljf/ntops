import functools

import torch

import ntops
from ntops.torch import _vendor_triton
from ntops.torch.utils import _cached_make, _cast_dtype, _dtype_is_floating_point


@functools.cache
def _is_iluvatar_device(index):
    if not hasattr(torch, "cuda"):
        return False
    if not torch.cuda.is_available():
        return False
    try:
        return "Iluvatar" in torch.cuda.get_device_name(index)
    except Exception:
        return False


@functools.cache
def _is_metax_device(index):
    if not hasattr(torch, "cuda"):
        return False
    if not torch.cuda.is_available():
        return False
    try:
        name = torch.cuda.get_device_name(index)
    except Exception:
        return False
    return "MetaX" in name or "MXC" in name


def _legacy_reduction(size_average, reduce, reduction):
    if reduce is False:
        return "none"
    if reduce is True:
        return "mean" if size_average is not False else "sum"
    return reduction


def _normalize_inputs(input, target):
    if input.ndim not in (1, 2):
        raise RuntimeError("multilabel_margin_loss only supports 1D or 2D input")
    if target.shape != input.shape:
        raise RuntimeError("inconsistent target size")
    if target.dtype != torch.long:
        raise RuntimeError("multilabel_margin_loss target should be int64")
    if not _dtype_is_floating_point(input.dtype):
        input = _cast_dtype(input, torch.float32)
    squeezed = input.ndim == 1
    if squeezed:
        input = input.reshape((1, input.shape[0]))
        target = target.reshape((1, target.shape[0]))
    if not input.is_contiguous():
        input = input.contiguous()
    if not target.is_contiguous():
        target = target.contiguous()
    return input, target, squeezed


@functools.cache
def _get_loss_kernel(batch, num_classes, class_block):
    return _cached_make(
        ntops.kernels.multilabel_margin_loss.premake_loss,
        batch,
        num_classes,
        class_block,
        block_size=ntops.kernels.multilabel_margin_loss.BLOCK_SIZE,
        num_warps=4,
        max_num_configs=1,
    )


def multilabel_margin_loss(
    input,
    target,
    size_average=None,
    reduce=None,
    reduction="mean",
):
    reduction = _legacy_reduction(size_average, reduce, reduction)
    if reduction not in ("none", "sum", "mean"):
        raise ValueError(f"{reduction} is not a valid value for reduction")

    input, target, squeezed = _normalize_inputs(input, target)
    batch, num_classes = input.shape
    losses = torch.empty((batch,), dtype=input.dtype, device=input.device)
    if not _vendor_triton.multilabel_margin_loss_per_row(input, target, losses):
        class_block = 1 << (num_classes - 1).bit_length()
        _get_loss_kernel(batch, num_classes, class_block)(
            input,
            target,
            losses,
        )

    if reduction == "none":
        return losses.reshape(()) if squeezed else losses

    accumulated = _vendor_triton.sum_strided(
        losses,
        losses.shape[0],
        stride=losses.stride(0),
        dtype=torch.float32,
        cast_fp32=True,
    )
    if accumulated is None:
        raise NotImplementedError("multilabel_margin_loss final reduction requires a vendor Triton kernel")
    if reduction == "mean":
        accumulated = accumulated / batch
    return _cast_dtype(accumulated, losses.dtype)
