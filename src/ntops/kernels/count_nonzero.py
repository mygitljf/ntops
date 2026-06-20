import functools

import ninetoothed
import ninetoothed.language as ntl
from ninetoothed import Tensor

from ntops.kernels.reduction import arrangement


BLOCK_SIZE = 1024


def application(input, output):
    # count_nonzero: per reduced row, sum of (element != 0). Accumulate int32 across
    # blocks then broadcast the int64 total back to every row position; the standard
    # reduction arrangement requires same-shape output, so the wrapper slices column 0.
    count = ntl.zeros(input.dtype.shape, dtype=ntl.int32)

    for i in range(input.shape[0]):
        nz = ntl.where(input[i] != 0, 1, 0)
        nz = ntl.where(input[i].offsets(-1) < input.source.shape[-1], nz, 0)
        count += nz

    total = ntl.sum(count, 0)

    for i in range(input.shape[0]):
        output[i] = ntl.cast(total, output.dtype.dtype)


def premake(ndim, dtype=None, block_size=BLOCK_SIZE):
    arrangement_ = functools.partial(arrangement, dim=(-1,), block_size=block_size)

    tensors = (
        Tensor(ndim, other=0, dtype=dtype),
        Tensor(ndim, dtype=ninetoothed.int64),
    )

    return arrangement_, application, tensors
