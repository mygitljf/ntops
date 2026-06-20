import functools

import ninetoothed
from ninetoothed import Tensor


BLOCK_SIZE = 2048


def arrangement(input, output, perm=None, block_size=None):
    if block_size is None:
        block_size = ninetoothed.block_size()

    input_arranged = input.permute(perm)
    input_arranged = input_arranged.flatten()
    input_arranged = input_arranged.tile((block_size,))

    output_arranged = output.flatten()
    output_arranged = output_arranged.tile((block_size,))

    return input_arranged, output_arranged


def application(input, output):
    output = input  # noqa: F841


def premake(ndim, perm=None, dtype=None, block_size=None):
    arrangement_ = functools.partial(arrangement, perm=perm, block_size=block_size)

    tensors = (Tensor(ndim, dtype=dtype), Tensor(ndim, dtype=dtype))

    return arrangement_, application, tensors
