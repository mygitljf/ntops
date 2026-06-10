import functools

import ninetoothed
import ninetoothed.language as ntl
from ninetoothed import Tensor
from ninetoothed.language import libdevice

from ntops.kernels.element_wise import arrangement


BLOCK_SIZE = 512


def application_8(input, other, output):
    dtype = output.dtype
    compute_dtype = ntl.int32
    abs_a = ntl.abs(ntl.cast(input, compute_dtype))
    abs_b = ntl.abs(ntl.cast(other, compute_dtype))
    or_ab = abs_a | abs_b
    safe_or = ntl.where(or_ab != 0, or_ab, 1)
    k = ntl.cast(libdevice.ffs(safe_or) - 1, compute_dtype)
    a0 = abs_a >> k
    b0 = abs_b >> k
    nonzero_a0 = a0 != 0
    safe_a0 = ntl.where(nonzero_a0, a0, 1)
    ctz_a0 = ntl.cast(libdevice.ffs(safe_a0) - 1, compute_dtype)
    a = ntl.where(nonzero_a0, a0 >> ctz_a0, b0)
    b = ntl.where(nonzero_a0, b0, ntl.cast(0, compute_dtype))
    a = ntl.where(a == 0, ntl.cast(1, compute_dtype), a)
    for _ in range(8):
        b_for_calc = ntl.where(b != 0, b, a)
        ctz_b = ntl.cast(libdevice.ffs(b_for_calc) - 1, compute_dtype)
        b_odd = b_for_calc >> ctz_b
        diff = b_odd - a
        a = ntl.minimum(a, b_odd)
        b = ntl.abs(diff)
    gcd = a << k
    safe_gcd = ntl.where(gcd == 0, 1, gcd)
    output = ntl.cast(  # noqa: F841
        ntl.where(or_ab == 0, 0, ntl.abs((abs_a // safe_gcd) * abs_b)), dtype
    )


def application_16(input, other, output):
    dtype = output.dtype
    compute_dtype = ntl.int32
    abs_a = ntl.abs(ntl.cast(input, compute_dtype))
    abs_b = ntl.abs(ntl.cast(other, compute_dtype))
    or_ab = abs_a | abs_b
    safe_or = ntl.where(or_ab != 0, or_ab, 1)
    k = ntl.cast(libdevice.ffs(safe_or) - 1, compute_dtype)
    a0 = abs_a >> k
    b0 = abs_b >> k
    nonzero_a0 = a0 != 0
    safe_a0 = ntl.where(nonzero_a0, a0, 1)
    ctz_a0 = ntl.cast(libdevice.ffs(safe_a0) - 1, compute_dtype)
    a = ntl.where(nonzero_a0, a0 >> ctz_a0, b0)
    b = ntl.where(nonzero_a0, b0, ntl.cast(0, compute_dtype))
    a = ntl.where(a == 0, ntl.cast(1, compute_dtype), a)
    for _ in range(16):
        b_for_calc = ntl.where(b != 0, b, a)
        ctz_b = ntl.cast(libdevice.ffs(b_for_calc) - 1, compute_dtype)
        b_odd = b_for_calc >> ctz_b
        diff = b_odd - a
        a = ntl.minimum(a, b_odd)
        b = ntl.abs(diff)
    gcd = a << k
    safe_gcd = ntl.where(gcd == 0, 1, gcd)
    output = ntl.cast(  # noqa: F841
        ntl.where(or_ab == 0, 0, ntl.abs((abs_a // safe_gcd) * abs_b)), dtype
    )


def application_36(input, other, output):
    dtype = output.dtype
    compute_dtype = ntl.int32
    abs_a = ntl.abs(ntl.cast(input, compute_dtype))
    abs_b = ntl.abs(ntl.cast(other, compute_dtype))
    or_ab = abs_a | abs_b
    safe_or = ntl.where(or_ab != 0, or_ab, 1)
    k = ntl.cast(libdevice.ffs(safe_or) - 1, compute_dtype)
    a0 = abs_a >> k
    b0 = abs_b >> k
    nonzero_a0 = a0 != 0
    safe_a0 = ntl.where(nonzero_a0, a0, 1)
    ctz_a0 = ntl.cast(libdevice.ffs(safe_a0) - 1, compute_dtype)
    a = ntl.where(nonzero_a0, a0 >> ctz_a0, b0)
    b = ntl.where(nonzero_a0, b0, ntl.cast(0, compute_dtype))
    a = ntl.where(a == 0, ntl.cast(1, compute_dtype), a)
    for _ in range(36):
        b_for_calc = ntl.where(b != 0, b, a)
        ctz_b = ntl.cast(libdevice.ffs(b_for_calc) - 1, compute_dtype)
        b_odd = b_for_calc >> ctz_b
        diff = b_odd - a
        a = ntl.minimum(a, b_odd)
        b = ntl.abs(diff)
    gcd = a << k
    safe_gcd = ntl.where(gcd == 0, 1, gcd)
    output = ntl.cast(  # noqa: F841
        ntl.where(or_ab == 0, 0, ntl.abs((abs_a // safe_gcd) * abs_b)), dtype
    )


def application_euclidean_dyn(input, other, output):
    # Dynamic Euclidean for int64.
    # Block-level early stop every 8 inner iters; outer cap 12 -> N=96.
    # Convergence for random uniform full-range int64 averages ~36 iters;
    # for v2-style range (<=2^20) averages ~14 iters.
    dtype = output.dtype
    abs_a = ntl.abs(input)
    abs_b = ntl.abs(other)
    or_ab = abs_a | abs_b
    a = ntl.where(abs_a >= abs_b, abs_a, abs_b)
    b = ntl.where(abs_a >= abs_b, abs_b, abs_a)
    outer = 0
    while ntl.max(b) != 0 and outer < 12:
        for _ in range(8):
            b_safe = ntl.where(b != 0, b, ntl.cast(1, dtype))
            r = a % b_safe
            a = ntl.where(b != 0, b, a)
            b = r
        outer += 1
    gcd = a
    safe_gcd = ntl.where(gcd == 0, 1, gcd)
    output = ntl.cast(  # noqa: F841
        ntl.where(or_ab == 0, 0, ntl.abs((abs_a // safe_gcd) * abs_b)), dtype
    )


def premake(ndim, dtype=None, block_size=BLOCK_SIZE):
    arrangement_ = functools.partial(arrangement, block_size=block_size)
    if dtype == ninetoothed.int64:
        application = application_euclidean_dyn
    elif dtype == ninetoothed.int32:
        application = application_36
    elif dtype == ninetoothed.int16:
        application = application_16
    else:
        application = application_8
    tensors = (
        Tensor(ndim, dtype=dtype),
        Tensor(ndim, dtype=dtype),
        Tensor(ndim, dtype=dtype),
    )
    return arrangement_, application, tensors
