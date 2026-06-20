import functools

import ninetoothed
import ninetoothed.language as ntl
from ninetoothed import Tensor

from ntops.kernels.element_wise import arrangement


BLOCK_SIZE = 2048


def application(input, values, output):
    dtype = output.dtype
    zero = ntl.cast(0, dtype)
    one = ntl.cast(1, dtype)
    # torch.heaviside: x>0 -> 1, x==0 -> values, everything else (x<0, NaN) -> 0.
    # Works for both float and integer dtypes (integer compares lower to icmp).
    output = ntl.where(  # noqa: F841
        input > zero,
        one,
        ntl.where(input == zero, values, zero),
    )


def application_bf16(input, values, output):
    dtype = output.dtype
    zero = ntl.cast(0, dtype)
    one = ntl.cast(1, dtype)
    # bf16 comparisons lower to fcmp on i16 and fail to compile; compare in f32
    # (exact: f32 represents every bf16 value, NaN is preserved by the cast).
    x = ntl.cast(input, ntl.float32)
    fzero = ntl.cast(0, ntl.float32)
    output = ntl.where(  # noqa: F841
        x > fzero,
        one,
        ntl.where(x == fzero, values, zero),
    )


def premake(ndim, dtype=None, block_size=BLOCK_SIZE):
    arrangement_ = functools.partial(arrangement, block_size=block_size)

    application_ = application_bf16 if dtype == ninetoothed.bfloat16 else application

    tensors = (
        Tensor(ndim, dtype=dtype),
        Tensor(ndim, dtype=dtype),
        Tensor(ndim, dtype=dtype),
    )

    return arrangement_, application_, tensors
