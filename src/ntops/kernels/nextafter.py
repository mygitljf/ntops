import functools

import ninetoothed
import ninetoothed.language as ntl
from ninetoothed import Tensor

from ntops.kernels.element_wise import arrangement


BLOCK_SIZE = 1024


def application_int16(input, other, output):
    # PyTorch nextafter spec, implemented via IEEE bit manipulation:
    #   if either is NaN: result is NaN
    #   if a == b: result is b (preserves sign of zero)
    #   if a == 0: result is smallest subnormal with sign of b
    #   otherwise: walk one ULP toward b in IEEE bit space
    dtype = output.dtype
    int_dtype = ntl.int16

    a = input
    b = other
    a_cmp = ntl.cast(a, ntl.float32)
    b_cmp = ntl.cast(b, ntl.float32)
    a_i = ntl.cast(a, int_dtype, bitcast=True)
    b_i = ntl.cast(b, int_dtype, bitcast=True)

    one = ntl.cast(1, int_dtype)
    zero = ntl.cast(0, int_dtype)
    sign_bit = one << 15

    is_nan = (a_cmp != a_cmp) | (b_cmp != b_cmp)
    eq = a_cmp == b_cmp
    is_zero = a_cmp == ntl.cast(0, ntl.float32)

    b_sign = b_i & sign_bit
    zero_result = b_sign | one

    a_neg = a_i < zero
    a_lt_b = a_cmp < b_cmp
    step_up = a_neg ^ a_lt_b
    step = ntl.where(step_up, one, -one)
    general = a_i + step

    nan_bits = ntl.cast(ntl.cast(float("nan"), dtype), int_dtype, bitcast=True)
    result_i = ntl.where(
        is_nan,
        nan_bits,
        ntl.where(eq, b_i, ntl.where(is_zero, zero_result, general)),
    )
    output = ntl.cast(result_i, dtype, bitcast=True)  # noqa: F841


def application_int32(input, other, output):
    dtype = output.dtype
    int_dtype = ntl.int32

    a = input
    b = other
    a_i = ntl.cast(a, int_dtype, bitcast=True)
    b_i = ntl.cast(b, int_dtype, bitcast=True)

    one = ntl.cast(1, int_dtype)
    zero = ntl.cast(0, int_dtype)
    sign_bit = one << 31

    is_nan = (a != a) | (b != b)
    eq = a == b
    is_zero = a == ntl.cast(0, dtype)

    b_sign = b_i & sign_bit
    zero_result = b_sign | one

    a_neg = a_i < zero
    a_lt_b = a < b
    step_up = a_neg ^ a_lt_b
    step = ntl.where(step_up, one, -one)
    general = a_i + step

    nan_bits = ntl.cast(ntl.cast(float("nan"), dtype), int_dtype, bitcast=True)
    result_i = ntl.where(
        is_nan,
        nan_bits,
        ntl.where(eq, b_i, ntl.where(is_zero, zero_result, general)),
    )
    output = ntl.cast(result_i, dtype, bitcast=True)  # noqa: F841


def application_int64(input, other, output):
    dtype = output.dtype
    int_dtype = ntl.int64

    a = input
    b = other
    a_i = ntl.cast(a, int_dtype, bitcast=True)
    b_i = ntl.cast(b, int_dtype, bitcast=True)

    one = ntl.cast(1, int_dtype)
    zero = ntl.cast(0, int_dtype)
    sign_bit = one << 63

    is_nan = (a != a) | (b != b)
    eq = a == b
    is_zero = a == ntl.cast(0, dtype)

    b_sign = b_i & sign_bit
    zero_result = b_sign | one

    a_neg = a_i < zero
    a_lt_b = a < b
    step_up = a_neg ^ a_lt_b
    step = ntl.where(step_up, one, -one)
    general = a_i + step

    nan_bits = ntl.cast(ntl.cast(float("nan"), dtype), int_dtype, bitcast=True)
    result_i = ntl.where(
        is_nan,
        nan_bits,
        ntl.where(eq, b_i, ntl.where(is_zero, zero_result, general)),
    )
    output = ntl.cast(result_i, dtype, bitcast=True)  # noqa: F841


def iluvatar_fp16_application(input, other, output):
    # Iluvatar Triton doesn't support bitcast between fp16 and int16.
    # Walk 1 fp16 ULP using fp32 int representation.
    # fp16 has 10-bit mantissa (5-bit exponent, bias 15).
    dtype = output.dtype
    a = ntl.cast(input, ntl.float32)
    b = ntl.cast(other, ntl.float32)
    int_dtype = ntl.int32

    a_i = ntl.cast(a, int_dtype, bitcast=True)
    b_i = ntl.cast(b, int_dtype, bitcast=True)

    one = ntl.cast(1, int_dtype)
    zero = ntl.cast(0, int_dtype)
    sign_bit = one << 31

    is_nan = (a != a) | (b != b)
    eq = a == b
    is_zero = a == ntl.cast(0, ntl.float32)

    b_sign = b_i & sign_bit
    a_neg = a_i < zero
    a_lt_b = a < b
    step_up = a_neg ^ a_lt_b

    nan_bits = ntl.cast(ntl.cast(float("nan"), ntl.float32), int_dtype, bitcast=True)

    # fp16 subnormal threshold: 2^-14 = 0x38800000 in fp32 int.
    subnormal_threshold = ntl.cast(0x38800000, int_dtype)
    abs_a_i = a_i & (sign_bit - one)
    is_subnormal = abs_a_i < subnormal_threshold
    # Normal fp16: 1 mantissa ULP = 2^(23-10) = 8192 fp32 int steps (constant
    # within and across binades, including the subnormal/normal boundary).
    normal_general = a_i + ntl.where(step_up, ntl.cast(8192, int_dtype), -ntl.cast(8192, int_dtype))
    # Subnormal fp16: neighbors are spaced a fixed absolute 2^-24, NOT a constant
    # fp32 int step, so walk in absolute fp32 (exact: all fp16 subnormals and
    # 2^-24 are representable in fp32).
    eps_sub = ntl.cast(5.9604644775390625e-08, ntl.float32)
    subnormal_f = a + ntl.where(step_up, eps_sub, -eps_sub)
    subnormal_general = ntl.cast(subnormal_f, int_dtype, bitcast=True)
    general = ntl.where(is_subnormal, subnormal_general, normal_general)
    # Smallest fp16 subnormal is 2^-24 = 0x33800000 in fp32 int.
    zero_result_int = b_sign | ntl.cast(0x33800000, int_dtype)

    result_i = ntl.where(
        is_nan,
        nan_bits,
        ntl.where(eq, b_i, ntl.where(is_zero, zero_result_int, general)),
    )
    output = ntl.cast(  # noqa: F841
        ntl.cast(result_i, ntl.float32, bitcast=True), dtype
    )


def iluvatar_bf16_application(input, other, output):
    # Iluvatar Triton doesn't support bitcast between bf16 and int16.
    # Walk 1 bf16 ULP using fp32 int representation.
    # bf16 has 7-bit mantissa (same exponent range as fp32).
    # 1 bf16 ULP = 2^16 = 65536 fp32 ULPs for both normal and subnormal.
    dtype = output.dtype
    a = ntl.cast(input, ntl.float32)
    b = ntl.cast(other, ntl.float32)
    int_dtype = ntl.int32

    a_i = ntl.cast(a, int_dtype, bitcast=True)
    b_i = ntl.cast(b, int_dtype, bitcast=True)

    one = ntl.cast(1, int_dtype)
    zero = ntl.cast(0, int_dtype)
    sign_bit = one << 31

    is_nan = (a != a) | (b != b)
    eq = a == b
    is_zero = a == ntl.cast(0, ntl.float32)

    b_sign = b_i & sign_bit
    a_neg = a_i < zero
    a_lt_b = a < b
    step_up = a_neg ^ a_lt_b

    nan_bits = ntl.cast(ntl.cast(float("nan"), ntl.float32), int_dtype, bitcast=True)

    step_size = ntl.cast(65536, int_dtype)
    # Smallest bf16 positive subnormal 2^-133 is 0x00010000 in fp32 int space
    # (fp32 int 1 == 2^-149, which underflows to 0 when cast back to bf16).
    zero_result_int = b_sign | ntl.cast(0x00010000, int_dtype)

    step = ntl.where(step_up, step_size, -step_size)
    general = a_i + step

    result_i = ntl.where(
        is_nan,
        nan_bits,
        ntl.where(eq, b_i, ntl.where(is_zero, zero_result_int, general)),
    )
    output = ntl.cast(  # noqa: F841
        ntl.cast(result_i, ntl.float32, bitcast=True), dtype
    )


def premake(ndim, dtype=None, iluvatar_half=False, block_size=BLOCK_SIZE):
    arrangement_ = functools.partial(arrangement, block_size=block_size)

    if iluvatar_half:
        if dtype == ninetoothed.bfloat16:
            application = iluvatar_bf16_application
        else:
            application = iluvatar_fp16_application
    elif dtype in (ninetoothed.float16, ninetoothed.bfloat16):
        application = application_int16
    elif dtype == ninetoothed.float32:
        application = application_int32
    else:
        application = application_int64

    tensors = (
        Tensor(ndim, dtype=dtype),
        Tensor(ndim, dtype=dtype),
        Tensor(ndim, dtype=dtype),
    )

    return arrangement_, application, tensors
