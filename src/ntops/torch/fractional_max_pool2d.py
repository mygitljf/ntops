import functools
from collections.abc import Iterable

import torch
import torch.nn.functional as F

import ntops
from ntops.torch import _vendor_triton
from ntops.torch.utils import _cached_make


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


def _pair(value, name):
    if isinstance(value, int):
        return value, value
    if isinstance(value, Iterable):
        value = tuple(value)
        if len(value) == 1:
            return value[0], value[0]
        if len(value) == 2:
            return value
    raise TypeError(f"{name} must be an int or pair of ints")


def _output_size(input, output_size, output_ratio):
    if output_size is None and output_ratio is None:
        raise ValueError(
            "fractional_max_pool2d requires specifying either an output_size or an output_ratio"
        )
    if output_size is not None:
        return _pair(output_size, "output_size")
    ratio = _pair(output_ratio, "output_ratio")
    return int(input.shape[-2] * ratio[0]), int(input.shape[-1] * ratio[1])


def _check(input, kernel_size, output_size, random_samples):
    if input.ndim not in (3, 4):
        raise RuntimeError(
            f"fractional_max_pool2d: Expected 3D or 4D tensor, but got: {input.ndim}"
        )
    if any(size <= 0 for size in kernel_size):
        raise RuntimeError("fractional_max_pool2d: kernel_size must be greater than zero")
    if any(size <= 0 for size in output_size):
        raise RuntimeError("fractional_max_pool2d: output_size must be greater than zero")
    if kernel_size[0] > input.shape[-2] or kernel_size[1] > input.shape[-1]:
        raise RuntimeError("fractional_max_pool2d: kernel_size is too large")
    if output_size[0] + kernel_size[0] - 1 > input.shape[-2]:
        raise RuntimeError("fractional_max_pool2d: output height is too large")
    if output_size[1] + kernel_size[1] - 1 > input.shape[-1]:
        raise RuntimeError("fractional_max_pool2d: output width is too large")
    expected = (1 if input.ndim == 3 else input.shape[0], input.shape[-3], 2)
    if tuple(random_samples.shape) != expected:
        raise RuntimeError(
            f"fractional_max_pool2d: expected _random_samples shape {expected}, "
            f"but got {tuple(random_samples.shape)}"
        )


@functools.cache
def _get_kernel(
    batch,
    channels,
    input_h,
    input_w,
    output_h,
    output_w,
    kernel_h,
    kernel_w,
):
    return _cached_make(
        ntops.kernels.fractional_max_pool2d.premake,
        batch,
        channels,
        input_h,
        input_w,
        output_h,
        output_w,
        kernel_h,
        kernel_w,
        block_size=ntops.kernels.fractional_max_pool2d.BLOCK_SIZE,
        num_warps=4,
        max_num_configs=1,
    )


def fractional_max_pool2d(
    input,
    kernel_size,
    output_size=None,
    output_ratio=None,
    return_indices=False,
    _random_samples=None,
):
    kernel_size = _pair(kernel_size, "kernel_size")
    output_size = _output_size(input, output_size, output_ratio)

    if not input.dtype.is_floating_point:
        input = input.to(torch.float32)

    if _random_samples is None:
        n_batch = 1 if input.ndim == 3 else input.shape[0]
        _random_samples = torch.rand(
            (n_batch, input.shape[-3], 2),
            dtype=input.dtype,
            device=input.device,
        )
    elif _random_samples.dtype != input.dtype:
        _random_samples = _random_samples.to(dtype=input.dtype, device=input.device)

    _check(input, kernel_size, output_size, _random_samples)

    squeeze_batch = input.ndim == 3
    if not _random_samples.is_contiguous():
        _random_samples = _random_samples.contiguous()

    if _is_metax_device(input.device.index) and input.dtype != torch.float16:
        output, indices = F.fractional_max_pool2d(
            input,
            kernel_size,
            output_size=output_size,
            return_indices=True,
            _random_samples=_random_samples,
        )
        return (output, indices) if return_indices else output

    batched = input.unsqueeze(0) if squeeze_batch else input

    output = torch.empty(
        (batched.shape[0], batched.shape[1], output_size[0], output_size[1]),
        dtype=input.dtype,
        device=input.device,
    )
    indices = torch.empty(output.shape, dtype=torch.long, device=input.device)

    if _vendor_triton.fractional_max_pool2d_fast(
        output, indices, batched, _random_samples, kernel_size, output_size,
    ):
        if squeeze_batch:
            output = output.reshape(output.shape[1:])
            indices = indices.reshape(indices.shape[1:])
        return (output, indices) if return_indices else output

    input = batched.contiguous()

    kernel = _get_kernel(
        input.shape[0],
        input.shape[1],
        input.shape[2],
        input.shape[3],
        output_size[0],
        output_size[1],
        kernel_size[0],
        kernel_size[1],
    )
    kernel(
        output,
        indices,
        input,
        _random_samples,
        input_h=input.shape[2],
        input_w=input.shape[3],
        output_h=output_size[0],
        output_w=output_size[1],
        kernel_h=kernel_size[0],
        kernel_w=kernel_size[1],
    )

    if squeeze_batch:
        output = output.reshape(output.shape[1:])
        indices = indices.reshape(indices.shape[1:])
    return (output, indices) if return_indices else output
