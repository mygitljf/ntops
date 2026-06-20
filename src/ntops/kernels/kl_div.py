import functools

import ninetoothed
import ninetoothed.language as ntl
from ninetoothed import Tensor

from ntops.kernels.element_wise import arrangement
from ntops.kernels.reduction import arrangement as reduce_arrangement


BLOCK_SIZE = 2048
REDUCE_BLOCK_SIZE = 1024


def application(input, target, output):
    # F.kl_div pointwise (log_target=False): target * (log(target) - input), with
    # the convention that target <= 0 contributes 0 (torch zeroes the term there).
    # log() needs fp32/fp64, so narrow floats compute in fp32 then cast back.
    dtype = output.dtype
    inp = ntl.cast(input, ntl.float32)
    tgt = ntl.cast(target, ntl.float32)
    zero = ntl.cast(0, ntl.float32)
    term = tgt * (ntl.log(tgt) - inp)
    output = ntl.cast(ntl.where(tgt > zero, term, zero), dtype)  # noqa: F841


def log_target_application(input, target, output):
    # log_target=True: exp(target) * (target - input). exp() needs fp32/fp64.
    dtype = output.dtype
    inp = ntl.cast(input, ntl.float32)
    tgt = ntl.cast(target, ntl.float32)
    output = ntl.cast(ntl.exp(tgt) * (tgt - inp), dtype)  # noqa: F841


def application_f64(input, target, output):
    # f64 variant: compute directly in fp64 (no narrowing cast through fp32).
    zero = ntl.cast(0, ntl.float64)
    term = target * (ntl.log(target) - input)
    output = ntl.where(target > zero, term, zero)  # noqa: F841


def log_target_application_f64(input, target, output):
    output = ntl.exp(target) * (target - input)  # noqa: F841


def sum_application(input, target, output):
    # Fused pointwise kl term + row reduction into fp32 partial sums (broadcast back;
    # the reduction arrangement requires same-shape output, wrapper reads column 0).
    acc = ntl.zeros(input.dtype.shape, dtype=ntl.float32)
    for i in range(input.shape[0]):
        inp = ntl.cast(input[i], ntl.float32)
        tgt = ntl.cast(target[i], ntl.float32)
        term = tgt * (ntl.log(tgt) - inp)
        term = ntl.where(tgt > 0.0, term, 0.0)
        valid = input[i].offsets(-1) < input.source.shape[-1]
        acc += ntl.where(valid, term, 0.0)
    total = ntl.sum(acc, 0)
    for i in range(input.shape[0]):
        output[i] = total


def sum_log_application(input, target, output):
    acc = ntl.zeros(input.dtype.shape, dtype=ntl.float32)
    for i in range(input.shape[0]):
        inp = ntl.cast(input[i], ntl.float32)
        tgt = ntl.cast(target[i], ntl.float32)
        term = ntl.exp(tgt) * (tgt - inp)
        valid = input[i].offsets(-1) < input.source.shape[-1]
        acc += ntl.where(valid, term, 0.0)
    total = ntl.sum(acc, 0)
    for i in range(input.shape[0]):
        output[i] = total


def sum_application_f64(input, target, output):
    acc = ntl.zeros(input.dtype.shape, dtype=ntl.float64)
    for i in range(input.shape[0]):
        inp = ntl.cast(input[i], ntl.float64)
        tgt = ntl.cast(target[i], ntl.float64)
        term = tgt * (ntl.log(tgt) - inp)
        term = ntl.where(tgt > 0.0, term, 0.0)
        valid = input[i].offsets(-1) < input.source.shape[-1]
        acc += ntl.where(valid, term, 0.0)
    total = ntl.sum(acc, 0)
    for i in range(input.shape[0]):
        output[i] = total


def sum_log_application_f64(input, target, output):
    acc = ntl.zeros(input.dtype.shape, dtype=ntl.float64)
    for i in range(input.shape[0]):
        inp = ntl.cast(input[i], ntl.float64)
        tgt = ntl.cast(target[i], ntl.float64)
        term = ntl.exp(tgt) * (tgt - inp)
        valid = input[i].offsets(-1) < input.source.shape[-1]
        acc += ntl.where(valid, term, 0.0)
    total = ntl.sum(acc, 0)
    for i in range(input.shape[0]):
        output[i] = total


def premake(ndim, dtype=None, log_target=False, block_size=BLOCK_SIZE):
    arrangement_ = functools.partial(arrangement, block_size=block_size)

    if dtype == ninetoothed.float64:
        application_ = log_target_application_f64 if log_target else application_f64
    else:
        application_ = log_target_application if log_target else application

    tensors = (
        Tensor(ndim, dtype=dtype),
        Tensor(ndim, dtype=dtype),
        Tensor(ndim, dtype=dtype),
    )

    return arrangement_, application_, tensors


def premake_sum(ndim, dtype=None, log_target=False, block_size=REDUCE_BLOCK_SIZE):
    arrangement_ = functools.partial(reduce_arrangement, dim=(-1,), block_size=block_size)

    if dtype == ninetoothed.float64:
        application_ = sum_log_application_f64 if log_target else sum_application_f64
        acc_dtype = ninetoothed.float64
    else:
        application_ = sum_log_application if log_target else sum_application
        acc_dtype = ninetoothed.float32

    tensors = (
        Tensor(ndim, other=0, dtype=dtype),
        Tensor(ndim, other=0, dtype=dtype),
        Tensor(ndim, dtype=acc_dtype),
    )

    return arrangement_, application_, tensors
