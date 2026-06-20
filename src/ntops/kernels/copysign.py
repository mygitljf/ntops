import functools

import ninetoothed
import ninetoothed.language as ntl
from ninetoothed import Tensor

from ntops.kernels.element_wise import arrangement


BLOCK_SIZE = 1024


def application_int16(input, other, output):
    # Pure bit manipulation: take magnitude bits of input, sign bit of other.
    # Avoids the fp16/bf16 -> fp32 -> fp16/bf16 round-trip required by
    # libdevice.copysign, which doesn't support narrow floats.
    dtype = output.dtype
    int_dtype = ntl.int16

    input_bits = ntl.cast(input, int_dtype, bitcast=True)
    other_bits = ntl.cast(other, int_dtype, bitcast=True)
    sign_bit = ntl.cast(1, int_dtype) << 15
    magn_mask = sign_bit - ntl.cast(1, int_dtype)
    output = ntl.cast(  # noqa: F841
        (input_bits & magn_mask) | (other_bits & sign_bit), dtype, bitcast=True
    )


def application_int32(input, other, output):
    dtype = output.dtype
    int_dtype = ntl.int32

    input_bits = ntl.cast(input, int_dtype, bitcast=True)
    other_bits = ntl.cast(other, int_dtype, bitcast=True)
    sign_bit = ntl.cast(1, int_dtype) << 31
    magn_mask = sign_bit - ntl.cast(1, int_dtype)
    output = ntl.cast(  # noqa: F841
        (input_bits & magn_mask) | (other_bits & sign_bit), dtype, bitcast=True
    )


def application_int64(input, other, output):
    dtype = output.dtype
    int_dtype = ntl.int64

    input_bits = ntl.cast(input, int_dtype, bitcast=True)
    other_bits = ntl.cast(other, int_dtype, bitcast=True)
    sign_bit = ntl.cast(1, int_dtype) << 63
    magn_mask = sign_bit - ntl.cast(1, int_dtype)
    output = ntl.cast(  # noqa: F841
        (input_bits & magn_mask) | (other_bits & sign_bit), dtype, bitcast=True
    )


def iluvatar_half_application(input, other, output):
    # Iluvatar Triton doesn't support bitcast between fp16/bf16 and int16.
    # Work around by casting to fp32, doing fp32 bit manipulation (int32),
    # then casting back.
    dtype = output.dtype
    input_f32 = ntl.cast(input, ntl.float32)
    other_f32 = ntl.cast(other, ntl.float32)
    int_dtype = ntl.int32

    input_bits = ntl.cast(input_f32, int_dtype, bitcast=True)
    other_bits = ntl.cast(other_f32, int_dtype, bitcast=True)
    sign_bit = ntl.cast(1, int_dtype) << 31
    magn_mask = sign_bit - ntl.cast(1, int_dtype)
    output = ntl.cast(  # noqa: F841
        ntl.cast(
            (input_bits & magn_mask) | (other_bits & sign_bit),
            ntl.float32,
            bitcast=True,
        ),
        dtype,
    )


def premake(ndim, dtype=None, iluvatar_half=False, block_size=BLOCK_SIZE):
    arrangement_ = functools.partial(arrangement, block_size=block_size)

    if iluvatar_half:
        application = iluvatar_half_application
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
