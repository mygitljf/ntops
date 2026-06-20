import functools

import ninetoothed
import ninetoothed.language as ntl
from ninetoothed import Tensor


BLOCK_SIZE = 1024


def arrangement(output, values, n, block_size=None):
    if block_size is None:
        block_size = ninetoothed.block_size()

    # output rows are tiled so each program covers one block of consecutive pair
    # indices; values stays unarranged and is read with raw-pointer gathers, so the
    # kernel works across any number of blocks (no block-local gather limitation).
    return (
        output[:, 0].flatten().tile((block_size,)),
        values,
        n,
    )


def application(output, values, n):
    # combinations(r=2): output row p maps to pair (i, j) with i<j via triangular
    # unranking. With n items there are n*(n-1)/2 pairs; row p sits in triangle row
    # i = max{k : F(k) <= p} where F(k) = k*(2n-1-k)/2 is the number of pairs before
    # row k, and j = p - F(i) + i + 1. i is recovered with an exact int64 binary
    # search on F (32 fixed steps cover any realistic n), avoiding float sqrt
    # precision limits entirely, so any pair count / block count is supported.
    p = ntl.cast(output.offsets(-1), ntl.int64)
    num = ntl.cast(n, ntl.int64)
    pairs = num * (num - 1) // 2
    valid = p < pairs

    lo = ntl.zeros_like(p)
    hi = lo + (num - 2)
    for _ in range(32):
        mid = (lo + hi + 1) // 2
        f_mid = mid * (2 * num - 1 - mid) // 2
        take = f_mid <= p
        lo = ntl.where(take, mid, lo)
        hi = ntl.where(take, hi, mid - 1)

    i_idx = lo
    f_i = i_idx * (2 * num - 1 - i_idx) // 2
    j_idx = p - f_i + i_idx + 1

    val0 = ntl.load(values.data_ptr() + i_idx, mask=valid, other=0)
    val1 = ntl.load(values.data_ptr() + j_idx, mask=valid, other=0)
    ntl.store(output.data_ptr() + p * 2, val0, mask=valid)
    ntl.store(output.data_ptr() + p * 2 + 1, val1, mask=valid)


def premake(n, dtype=None, block_size=BLOCK_SIZE):
    arrangement_ = functools.partial(arrangement, block_size=block_size)

    tensors = (
        Tensor(2, dtype=dtype),
        # Concrete shape: `values` stays unarranged (raw-pointer gathers), so its
        # size must not remain a free launch symbol (autotune needs bounds).
        Tensor(1, shape=(n,), dtype=dtype),
        Tensor(0, dtype=None, constexpr=True, name="n"),
    )

    return arrangement_, application, tensors
