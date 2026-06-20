import functools

import ninetoothed.language as ntl
from ninetoothed import Tensor

from ntops.kernels.element_wise import arrangement


BLOCK_SIZE = 2048


def application(input, output):
    truncated = ntl.where(input < 0, ntl.ceil(input), ntl.floor(input))
    output = input - truncated  # noqa: F841


def half_application(input, output):
    value = ntl.cast(input, ntl.float32)
    truncated = ntl.where(value < 0, ntl.ceil(value), ntl.floor(value))
    output = ntl.cast(value - truncated, ntl.float16)  # noqa: F841


def bfloat16_application(input, output):
    value = ntl.cast(input, ntl.float32)
    truncated = ntl.where(value < 0, ntl.ceil(value), ntl.floor(value))
    output = ntl.cast(value - truncated, ntl.bfloat16)  # noqa: F841


def premake(ndim, half=False, bfloat16=False, dtype=None, block_size=BLOCK_SIZE):
    arrangement_ = functools.partial(arrangement, block_size=block_size)
    tensors = (Tensor(ndim, dtype=dtype), Tensor(ndim, dtype=dtype))
    if half:
        application_ = half_application
    elif bfloat16:
        application_ = bfloat16_application
    else:
        application_ = application
    return arrangement_, application_, tensors
