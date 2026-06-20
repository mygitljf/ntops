import functools

import ninetoothed
import ninetoothed.language as ntl
from ninetoothed import Tensor

from ntops.kernels.element_wise import arrangement


BLOCK_SIZE = 2048


def application(input, p, seed, spatial, a, b, dropped, output):
    # feature_alpha_dropout masks whole channels: one bernoulli per (N,C) channel,
    # then an affine map a*x+b on kept channels and a constant `dropped` saturation
    # on masked ones. channel = flat_offset // spatial keys the per-channel RNG.
    channel = input.offsets() // spatial
    keep = ntl.rand(seed, channel) > p
    affine = a * ntl.cast(input, ntl.float32) + b
    output = ntl.cast(ntl.where(keep, affine, dropped), output.dtype)  # noqa: F841


def premake(ndim, dtype=None, block_size=BLOCK_SIZE):
    arrangement_ = functools.partial(arrangement, block_size=block_size)

    tensors = (
        Tensor(ndim, dtype=dtype),
        Tensor(0, dtype=ninetoothed.float64),
        Tensor(0, dtype=ninetoothed.int64),
        Tensor(0, dtype=ninetoothed.int64),
        Tensor(0, dtype=ninetoothed.float64),
        Tensor(0, dtype=ninetoothed.float64),
        Tensor(0, dtype=ninetoothed.float64),
        Tensor(ndim, dtype=dtype),
    )

    return arrangement_, application, tensors
