import functools

import ninetoothed
from ninetoothed import Tensor


BLOCK_SIZE = 2048


def arrangement(input, output, perm=None, block_size=None):
    if block_size is None:
        block_size = ninetoothed.block_size()

    input_arranged = input.permute(perm).flatten().tile((block_size,))
    output_arranged = output.flatten().tile((block_size,))

    return input_arranged, output_arranged


def application(input, output):
    output = input  # noqa: F841


def premake(perm=None, dtype=None, block_size=BLOCK_SIZE):
    arrangement_ = functools.partial(arrangement, perm=perm, block_size=block_size)

    tensors = (Tensor(6, dtype=dtype), Tensor(6, dtype=dtype))

    return arrangement_, application, tensors
