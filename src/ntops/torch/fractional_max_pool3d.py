import functools
from collections.abc import Iterable

import torch
import torch as _orig_torch

import ntops
from ntops.torch import _vendor_triton
from ntops.torch.utils import _cached_make, _cast_dtype, _dtype_is_floating_point


def _triple(value, name):
    if isinstance(value, int):
        return value, value, value
    if isinstance(value, Iterable):
        value = tuple(value)
        if len(value) == 1:
            return value[0], value[0], value[0]
        if len(value) == 3:
            return value
    raise TypeError(f"{name} must be an int or triple of ints")


def _output_size(input, output_size, output_ratio):
    if output_size is None and output_ratio is None:
        raise ValueError(
            "fractional_max_pool3d requires specifying either an output_size or an output_ratio"
        )
    if output_size is not None:
        return _triple(output_size, "output_size")
    ratio = _triple(output_ratio, "output_ratio")
    return (
        int(input.shape[-3] * ratio[0]),
        int(input.shape[-2] * ratio[1]),
        int(input.shape[-1] * ratio[2]),
    )


def _check(input, kernel_size, output_size, random_samples):
    if input.ndim not in (4, 5):
        raise RuntimeError(
            f"fractional_max_pool3d: Expected 4D or 5D tensor, but got: {input.ndim}"
        )
    if any(size <= 0 for size in kernel_size):
        raise RuntimeError("fractional_max_pool3d: kernel_size must be greater than zero")
    if any(size <= 0 for size in output_size):
        raise RuntimeError("fractional_max_pool3d: output_size must be greater than zero")
    if any(kernel_size[i] > input.shape[-3 + i] for i in range(3)):
        raise RuntimeError("fractional_max_pool3d: kernel_size is too large")
    if any(output_size[i] + kernel_size[i] - 1 > input.shape[-3 + i] for i in range(3)):
        raise RuntimeError("fractional_max_pool3d: output_size is too large")
    expected = (1 if input.ndim == 4 else input.shape[0], input.shape[-4], 3)
    if tuple(random_samples.shape) != expected:
        raise RuntimeError(
            f"fractional_max_pool3d: expected _random_samples shape {expected}, "
            f"but got {tuple(random_samples.shape)}"
        )


@functools.cache
def _get_kernel(
    batch,
    channels,
    input_t,
    input_h,
    input_w,
    output_t,
    output_h,
    output_w,
    kernel_t,
    kernel_h,
    kernel_w,
):
    return _cached_make(
        ntops.kernels.fractional_max_pool3d.premake,
        batch,
        channels,
        input_t,
        input_h,
        input_w,
        output_t,
        output_h,
        output_w,
        kernel_t,
        kernel_h,
        kernel_w,
        block_size=ntops.kernels.fractional_max_pool3d.BLOCK_SIZE,
        num_warps=4,
        max_num_configs=1,
    )


def fractional_max_pool3d(
    input,
    kernel_size,
    output_size=None,
    output_ratio=None,
    return_indices=False,
    _random_samples=None,
):
    kernel_size = _triple(kernel_size, "kernel_size")
    output_size = _output_size(input, output_size, output_ratio)

    if not _dtype_is_floating_point(input.dtype):
        input = _cast_dtype(input, torch.float32)

    if _random_samples is None:
        n_batch = 1 if input.ndim == 4 else input.shape[0]
        dev_index = input.device.index if input.device.index is not None else 0
        dev = _orig_torch.device("cuda", dev_index)
        dt_str = str(input.dtype).rsplit(".", 1)[-1]
        dt = getattr(_orig_torch, dt_str, _orig_torch.float32)
        _random_samples = _orig_torch.rand(
            (n_batch, input.shape[-4], 3),
            dtype=dt,
            device=dev,
        )
    elif str(_random_samples.dtype).rsplit(".", 1)[-1] != str(input.dtype).rsplit(".", 1)[-1]:
        dev_index = input.device.index if input.device.index is not None else 0
        dev = _orig_torch.device("cuda", dev_index)
        dt_str = str(input.dtype).rsplit(".", 1)[-1]
        dt = getattr(_orig_torch, dt_str, _orig_torch.float32)
        _random_samples = _random_samples.to(device=dev, dtype=dt)

    _check(input, kernel_size, output_size, _random_samples)

    squeeze_batch = input.ndim == 4
    if not _random_samples.is_contiguous():
        _random_samples = _random_samples.contiguous()

    batched = input.unsqueeze(0) if squeeze_batch else input

    output = torch.empty(
        (
            batched.shape[0],
            batched.shape[1],
            output_size[0],
            output_size[1],
            output_size[2],
        ),
        dtype=input.dtype,
        device=input.device,
    )
    indices = torch.empty(output.shape, dtype=torch.long, device=input.device)

    if _vendor_triton.fractional_max_pool3d_fast(
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
        input.shape[4],
        output_size[0],
        output_size[1],
        output_size[2],
        kernel_size[0],
        kernel_size[1],
        kernel_size[2],
    )
    kernel(
        output,
        indices,
        input,
        _random_samples,
        input_t=input.shape[2],
        input_h=input.shape[3],
        input_w=input.shape[4],
        output_t=output_size[0],
        output_h=output_size[1],
        output_w=output_size[2],
        kernel_t=kernel_size[0],
        kernel_h=kernel_size[1],
        kernel_w=kernel_size[2],
    )

    if squeeze_batch:
        output = output.reshape(output.shape[1:])
        indices = indices.reshape(indices.shape[1:])
    return (output, indices) if return_indices else output
