import functools

import ninetoothed
import ninetoothed.language as ntl
from ninetoothed import Tensor


BLOCK_SIZE = 256


def arrangement(
    output,
    indices,
    input,
    random_samples,
    input_h,
    input_w,
    output_h,
    output_w,
    kernel_h,
    kernel_w,
    block_size=None,
):
    if block_size is None:
        block_size = ninetoothed.block_size()

    return (
        output.flatten().tile((block_size,)),
        indices.flatten().tile((block_size,)),
        input,
        random_samples,
        input_h,
        input_w,
        output_h,
        output_w,
        kernel_h,
        kernel_w,
    )


def _pool_start(output_index, input_size, output_size, kernel_size, sample):
    if output_size == 1:
        return 0
    alpha = (input_size - kernel_size) / (output_size - 1)
    sample = ntl.cast(sample, ntl.float32)
    start = ntl.floor((ntl.cast(output_index, ntl.float32) + sample) * alpha)
    start -= ntl.floor(sample * alpha)
    start = ntl.where(output_index == output_size - 1, input_size - kernel_size, start)
    return ntl.cast(start, ntl.int64)


def application(
    output,
    indices,
    input,
    random_samples,
    input_h,
    input_w,
    output_h,
    output_w,
    kernel_h,
    kernel_w,
):
    n = output.offsets(0)
    c = output.offsets(1)
    oh = output.offsets(2)
    ow = output.offsets(3)
    valid_output = (
        (n < output.source.shape[0])
        & (c < output.source.shape[1])
        & (oh < output.source.shape[2])
        & (ow < output.source.shape[3])
    )

    random_base = n * random_samples.stride(0) + c * random_samples.stride(1)
    sample_h = ntl.load(random_samples.data_ptr() + random_base, mask=valid_output, other=0.0)
    sample_w = ntl.load(
        random_samples.data_ptr() + random_base + random_samples.stride(2),
        mask=valid_output,
        other=0.0,
    )

    start_h = _pool_start(
        oh,
        input_h,
        output_h,
        kernel_h,
        sample_h,
    )
    start_w = _pool_start(
        ow,
        input_w,
        output_w,
        kernel_w,
        sample_w,
    )

    max_value = ntl.cast(n * 0, input.dtype) - float("inf")
    max_index = ntl.cast(n * 0, ntl.int64)
    input_base = n * input.stride(0) + c * input.stride(1)

    for kh in range(kernel_h):
        for kw in range(kernel_w):
            ih = start_h + kh
            iw = start_w + kw
            input_offset = input_base + ih * input.stride(2) + iw * input.stride(3)
            value = ntl.load(input.data_ptr() + input_offset, mask=valid_output, other=-float("inf"))
            index = ih * input_w + iw
            take = (value > max_value) | (value != value)
            max_value = ntl.where(take, value, max_value)
            max_index = ntl.where(take, index, max_index)

    output = max_value  # noqa: F841
    indices = max_index  # noqa: F841


def premake(
    batch,
    channels,
    input_h,
    input_w,
    output_h,
    output_w,
    kernel_h,
    kernel_w,
    dtype=None,
    block_size=BLOCK_SIZE,
):
    arrangement_ = functools.partial(arrangement, block_size=block_size)
    tensors = (
        Tensor(4, shape=(batch, channels, output_h, output_w), dtype=dtype),
        Tensor(4, shape=(batch, channels, output_h, output_w), dtype=ninetoothed.int64),
        Tensor(4, shape=(batch, channels, input_h, input_w), dtype=dtype, other=-float("inf")),
        Tensor(3, shape=(batch, channels, 2), dtype=dtype),
        Tensor(0, dtype=None, constexpr=True, name="input_h"),
        Tensor(0, dtype=None, constexpr=True, name="input_w"),
        Tensor(0, dtype=None, constexpr=True, name="output_h"),
        Tensor(0, dtype=None, constexpr=True, name="output_w"),
        Tensor(0, dtype=None, constexpr=True, name="kernel_h"),
        Tensor(0, dtype=None, constexpr=True, name="kernel_w"),
    )
    return arrangement_, application, tensors
