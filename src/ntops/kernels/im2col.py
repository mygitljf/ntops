import functools

import ninetoothed
from ninetoothed import Symbol, Tensor


BLOCK_SIZE = 2048


def arrangement(
    input,
    output,
    kernel_size_h=None,
    kernel_size_w=None,
    stride_h=None,
    stride_w=None,
    padding_h=None,
    padding_w=None,
    dilation_h=None,
    dilation_w=None,
    block_size=None,
):
    if kernel_size_h is None:
        kernel_size_h = Symbol("kernel_size_h", constexpr=True, upper_bound=16)

    if kernel_size_w is None:
        kernel_size_w = Symbol("kernel_size_w", constexpr=True, upper_bound=16)

    if stride_h is None:
        stride_h = Symbol("stride_h", constexpr=True)

    if stride_w is None:
        stride_w = Symbol("stride_w", constexpr=True)

    if padding_h is None:
        padding_h = Symbol("padding_h", constexpr=True)

    if padding_w is None:
        padding_w = Symbol("padding_w", constexpr=True)

    if dilation_h is None:
        dilation_h = Symbol("dilation_h", constexpr=True)

    if dilation_w is None:
        dilation_w = Symbol("dilation_w", constexpr=True)

    if block_size is None:
        block_size = ninetoothed.block_size()

    input_arranged = input.pad(
        ((0, 0), (0, 0), (padding_h, padding_h), (padding_w, padding_w))
    )
    input_arranged = input_arranged.tile(
        (1, -1, kernel_size_h, kernel_size_w),
        strides=(-1, -1, stride_h, stride_w),
        dilation=(1, 1, dilation_h, dilation_w),
        floor_mode=True,
    )
    input_arranged = input_arranged.squeeze(1)
    input_arranged.dtype = input_arranged.dtype.squeeze(0)
    input_arranged = input_arranged.ravel()
    input_arranged = input_arranged.permute((0, 3, 4, 5, 1, 2))
    input_arranged = input_arranged.flatten(end_dim=4).flatten(start_dim=1)
    input_arranged = input_arranged.tile((1, block_size))

    output_arranged = output.flatten(end_dim=2)
    output_arranged = output_arranged.tile((1, block_size))

    return input_arranged, output_arranged


def application(input, output):
    output = input  # noqa: F841


def premake(
    channels=None,
    kernel_size_h=None,
    kernel_size_w=None,
    stride_h=None,
    stride_w=None,
    padding_h=None,
    padding_w=None,
    dilation_h=None,
    dilation_w=None,
    dtype=None,
    block_size=BLOCK_SIZE,
):
    arrangement_ = functools.partial(
        arrangement,
        kernel_size_h=kernel_size_h,
        kernel_size_w=kernel_size_w,
        stride_h=stride_h,
        stride_w=stride_w,
        padding_h=padding_h,
        padding_w=padding_w,
        dilation_h=dilation_h,
        dilation_w=dilation_w,
        block_size=block_size,
    )

    if (
        channels is not None
        and kernel_size_h is not None
        and kernel_size_w is not None
    ):
        output_channels = channels * kernel_size_h * kernel_size_w
    else:
        output_channels = None

    tensors = (
        Tensor(4, shape=(None, channels, None, None), dtype=dtype, other=0),
        Tensor(3, shape=(None, output_channels, None), dtype=dtype),
    )

    return arrangement_, application, tensors
