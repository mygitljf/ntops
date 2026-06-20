import math
import logging
from collections.abc import Iterable

import torch

import ntops
from ntops.torch import _vendor_triton
from ntops.torch.utils import _cached_make

_logger = logging.getLogger(__name__)


def _compile_errors():
    errors = [FileNotFoundError]
    try:
        from triton.compiler.errors import CompilationError

        errors.append(CompilationError)
    except ImportError:
        pass
    return tuple(errors)


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
    except _compile_errors():
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
        # ntops: capability-fallback - backend compiler failed before a kernel can run.
        result = torch.nn.functional.unfold(
            input,
            kernel_size=kernel_size,
            dilation=dilation,
            padding=padding,
            stride=stride,
        )
        _copy_kernel(result.ndim, result.dtype)(result, output)

    return output


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


def _copy_kernel(ndim, torch_dtype):
    nt_dtype = _to_nt(torch_dtype)
    if nt_dtype is None:
        raise NotImplementedError(f"im2col fallback copy kernel does not support {torch_dtype}")
    return _cached_make(
        ntops.kernels.identity_copy.premake,
        ndim,
        dtype=nt_dtype,
        block_size=ntops.kernels.identity_copy.BLOCK_SIZE,
        num_warps=4,
        max_num_configs=1,
    )
