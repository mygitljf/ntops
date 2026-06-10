import functools

import ninetoothed
import ninetoothed.language as ntl
from ninetoothed import Tensor

from ntops.kernels.element_wise import arrangement


BLOCK_SIZE = 2048


def application(input, output):
    output = input * 57.29577951308232  # noqa: F841


def bfloat16_application(input, output):
    output = ntl.cast(ntl.cast(input, ntl.float32) * 57.29577951308232, ntl.bfloat16)  # noqa: F841


def premake(ndim, dtype=None, block_size=BLOCK_SIZE):
    arrangement_ = functools.partial(arrangement, block_size=block_size)

    tensors = (Tensor(ndim, dtype=dtype), Tensor(ndim, dtype=dtype))

    application_ = bfloat16_application if dtype == ninetoothed.bfloat16 else application

    return arrangement_, application_, tensors
