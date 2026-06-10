import math
import logging
from collections.abc import Iterable

import torch

import ntops
from ntops.torch import _vendor_triton
from ntops.torch.utils import _cached_make

_logger = logging.getLogger(__name__)


def _pair(value, name):
    if isinstance(value, int):
        return value, value

    if isinstance(value, Iterable):
        value = tuple(value)
        if len(value) == 1:
            return value[0], value[0]
        if len(value) == 2:
            return value

    raise TypeError(f"{name} must be an int or a pair of ints")


def _calculate_output_size(input_size, kernel_size, dilation, padding, stride):
    return math.floor(
        (input_size + 2 * padding - dilation * (kernel_size - 1) - 1) / stride + 1
    )


def _prefer_native(input, kernel_size, stride, out_h, out_w):
    if not _vendor_triton.is_iluvatar_device(input):
        return False
    if not input.dtype.is_floating_point:
        return False
    if stride != (1, 1) or max(kernel_size) > 3:
        return False
    return out_h * out_w >= 4096


def im2col(input, kernel_size, dilation=1, padding=0, stride=1):
    if input.ndim != 4:
        raise RuntimeError(
            "Expected 4D input tensor, but got tensor with "
            f"{input.ndim} dimensions"
        )

    kernel_size = _pair(kernel_size, "kernel_size")
    dilation = _pair(dilation, "dilation")
    padding = _pair(padding, "padding")
    stride = _pair(stride, "stride")

    if any(size <= 0 for size in kernel_size):
        raise RuntimeError("kernel_size should be greater than zero")
    if any(size <= 0 for size in dilation):
        raise RuntimeError("dilation should be greater than zero")
    if any(size <= 0 for size in stride):
        raise RuntimeError("stride should be greater than zero")
    if any(size < 0 for size in padding):
        raise RuntimeError("padding should be non-negative")

    n, c, h, w = input.shape
    out_h = _calculate_output_size(
        h, kernel_size[0], dilation[0], padding[0], stride[0]
    )
    out_w = _calculate_output_size(
        w, kernel_size[1], dilation[1], padding[1], stride[1]
    )

    if out_h < 1 or out_w < 1:
        raise RuntimeError(
            "Given input with spatial size "
            f"({h}, {w}), kernel_size={kernel_size}, dilation={dilation}, "
            f"padding={padding}, calculated output shape ({out_h}, {out_w}), "
            "but its components must be at least one."
        )

    if _prefer_native(input, kernel_size, stride, out_h, out_w):
        return torch.nn.functional.unfold(
            input,
            kernel_size=kernel_size,
            dilation=dilation,
            padding=padding,
            stride=stride,
        )

    output = torch.empty(
        (n, c * kernel_size[0] * kernel_size[1], out_h * out_w),
        dtype=input.dtype,
        device=input.device,
    )

    if not input.is_contiguous():
        input = input.contiguous()

    try:
        if _vendor_triton.im2col_fast_path(input, output, kernel_size, dilation, padding, stride):
            return output
    except Exception:
        pass

    try:
        kernel = _cached_make(
            ntops.kernels.im2col.premake,
            channels=c,
            kernel_size_h=kernel_size[0],
            kernel_size_w=kernel_size[1],
            stride_h=stride[0],
            stride_w=stride[1],
            padding_h=padding[0],
            padding_w=padding[1],
            dilation_h=dilation[0],
            dilation_w=dilation[1],
        )
        kernel(input, output)
    except FileNotFoundError:
        # Triton backend compilation failed (e.g. Iluvatar libdevice missing).
        # Fall back to PyTorch F.unfold as a last-resort reference path.
        _logger.warning(
            "im2col: NineToothed kernel compilation failed, "
            "falling back to F.unfold"
        )
        result = torch.nn.functional.unfold(
            input,
            kernel_size=kernel_size,
            dilation=dilation,
            padding=padding,
            stride=stride,
        )
        output.copy_(result)

    return output
