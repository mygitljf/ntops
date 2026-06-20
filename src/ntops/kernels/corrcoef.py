import functools

import ninetoothed.language as ntl
from ninetoothed import Tensor

from ntops.kernels.reduction import arrangement


BLOCK_SIZE = 1024


def normalize_application(input, output):
    # Emit Xn[i] = (X[i] - mean_i) / sqrt(sum_j (X[i,j]-mean_i)^2) so the Gram matrix
    # Xn @ Xn^T equals corrcoef directly (the 1/(D-1) variance factors cancel between
    # numerator and denominator).
    n = input.source.shape[-1]
    _sum = ntl.zeros(input.dtype.shape, dtype=ntl.float32)
    for i in range(input.shape[0]):
        valid = input[i].offsets(-1) < n
        _sum += ntl.where(valid, ntl.cast(input[i], ntl.float32), 0.0)
    mean = ntl.sum(_sum, 0) / n

    _ss = ntl.zeros(input.dtype.shape, dtype=ntl.float32)
    for i in range(input.shape[0]):
        valid = input[i].offsets(-1) < n
        diff = ntl.where(valid, ntl.cast(input[i], ntl.float32) - mean, 0.0)
        _ss += diff * diff
    inv = 1.0 / ntl.sqrt(ntl.sum(_ss, 0))

    for i in range(input.shape[0]):
        diff = ntl.cast(input[i], ntl.float32) - mean
        output[i] = ntl.cast(diff * inv, output.dtype.dtype)


def premake_normalize(ndim, dtype=None, block_size=BLOCK_SIZE):
    arrangement_ = functools.partial(arrangement, dim=(-1,), block_size=block_size)

    tensors = (
        Tensor(ndim, other=0, dtype=dtype),
        Tensor(ndim, dtype=dtype),
    )

    return arrangement_, normalize_application, tensors
