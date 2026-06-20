import functools

import ninetoothed
import ninetoothed.language as ntl
from ninetoothed import Tensor

from ntops.kernels.reduction import arrangement


BLOCK_SIZE = 1024
MAX_BLOCK_SIZE = 8192


def application(logits, seed, tau, output):
    # gumbel_softmax soft sample over the trailing dim: draw g = -log(-log(U)) per
    # element, then a numerically-stable softmax of (logits + g) / tau. Online max +
    # denominator accumulation across blocks (same streaming softmax as kernels.softmax).
    n = logits.source.shape[-1]
    prev_max = ntl.cast(float("-inf"), ntl.float32)
    denom = ntl.cast(0.0, ntl.float32)

    for i in range(logits.shape[0]):
        off = logits[i].offsets(-1)
        u = ntl.rand(seed, off)
        g = -ntl.log(-ntl.log(u + 1e-20) + 1e-20)
        val = (ntl.cast(logits[i], ntl.float32) + g) / tau
        val = ntl.where(off < n, val, float("-inf"))
        cur_max = ntl.maximum(prev_max, ntl.max(val))
        denom = denom * ntl.exp(prev_max - cur_max) + ntl.sum(ntl.exp(val - cur_max))
        prev_max = cur_max

    for i in range(logits.shape[0]):
        off = logits[i].offsets(-1)
        u = ntl.rand(seed, off)
        g = -ntl.log(-ntl.log(u + 1e-20) + 1e-20)
        val = (ntl.cast(logits[i], ntl.float32) + g) / tau
        output[i] = ntl.cast(ntl.exp(val - prev_max) / denom, output.dtype.dtype)


def premake(rows, cols, dtype=None, block_size=BLOCK_SIZE):
    arrangement_ = functools.partial(arrangement, dim=(-1,), block_size=block_size)

    tensors = (
        Tensor(2, shape=(rows, cols), dtype=dtype, other=float("-inf")),
        Tensor(0, dtype=ninetoothed.int64),
        Tensor(0, dtype=ninetoothed.float64),
        Tensor(2, shape=(rows, cols), dtype=dtype),
    )

    return arrangement_, application, tensors
