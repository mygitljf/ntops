import functools

import ninetoothed
import ninetoothed.language as ntl
from ninetoothed import Tensor

from ntops.kernels.element_wise import arrangement
from ntops.kernels.reduction import arrangement as reduce_arrangement


BLOCK_SIZE = 2048
REDUCE_BLOCK_SIZE = 1024


def application(input, target, output):
    diff = input - target
    output = diff * diff  # noqa: F841


def sum_application(input, target, output):
    # Fused squared-diff + row reduction into fp32 partial sums (broadcast back;
    # the reduction arrangement requires same-shape output, wrapper reads
    # column 0).
    acc = ntl.zeros(input.dtype.shape, dtype=ntl.float32)
    for i in range(input.shape[0]):
        inp = ntl.cast(input[i], ntl.float32)
        tgt = ntl.cast(target[i], ntl.float32)
        diff = inp - tgt
        valid = input[i].offsets(-1) < input.source.shape[-1]
        acc += ntl.where(valid, diff * diff, 0.0)
    total = ntl.sum(acc, 0)
    for i in range(input.shape[0]):
        output[i] = total


def premake(ndim, dtype=None, block_size=BLOCK_SIZE):
    arrangement_ = functools.partial(arrangement, block_size=block_size)

    tensors = (
        Tensor(ndim, dtype=dtype),
        Tensor(ndim, dtype=dtype),
        Tensor(ndim, dtype=dtype),
    )

    return arrangement_, application, tensors


def premake_sum(ndim, dtype=None, block_size=REDUCE_BLOCK_SIZE):
    arrangement_ = functools.partial(
        reduce_arrangement, dim=(-1,), block_size=block_size
    )

    tensors = (
        Tensor(ndim, other=0, dtype=dtype),
        Tensor(ndim, other=0, dtype=dtype),
        Tensor(ndim, dtype=ninetoothed.float32),
    )

    return arrangement_, sum_application, tensors
