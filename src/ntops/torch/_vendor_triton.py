import functools
import math

import torch
import triton
import triton.language as tl

try:
    from triton.language.extra import libdevice as _tl_libdevice
except Exception:  # pragma: no cover - backend without libdevice extra
    _tl_libdevice = None


@functools.cache
def _device_name_for_index(index):
    try:
        return torch.cuda.get_device_name(index)
    except Exception:
        return ""


def _device_name(tensor):
    # Accept both torch.Tensor and infinicore.Tensor (InfiniCore integration);
    # both expose a .device with .type/.index, so detect by duck typing.
    device = getattr(tensor, "device", None)
    if device is None or getattr(device, "type", None) != "cuda":
        return ""
    if not hasattr(torch, "cuda"):
        return ""
    index = getattr(device, "index", None)
    if index is None:
        index = torch.cuda.current_device()
    return _device_name_for_index(index)


def is_iluvatar_device(tensor):
    return "Iluvatar" in _device_name(tensor)


def is_metax_device(tensor):
    return "MetaX" in _device_name(tensor)


def is_corex_or_metax_device(tensor):
    name = _device_name(tensor)
    return "Iluvatar" in name or "MetaX" in name


@functools.cache
def _lcm_small_table(device_index):
    values = [math.lcm(lhs, rhs) for lhs in range(256) for rhs in range(256)]
    return torch.tensor(values, dtype=torch.int64, device=torch.device("cuda", device_index))


def lcm_small_table(device):
    index = device.index
    if index is None:
        index = torch.cuda.current_device()
    return _lcm_small_table(index)


@functools.cache
def _gcd_byte_table(device_index):
    values = [math.gcd(lhs, rhs) for lhs in range(256) for rhs in range(256)]
    return torch.tensor(values, dtype=torch.uint8, device=torch.device("cuda", device_index))


def _gcd_byte_table_for(device):
    index = device.index
    if index is None:
        index = torch.cuda.current_device()
    return _gcd_byte_table(index)


@functools.cache
def _lcm_wrapped_byte_table(device_index):
    # 1-byte output dtypes wrap mod 256, so the wrapped LCM is exact via gather alone.
    values = [(math.lcm(lhs, rhs) & 0xFF) for lhs in range(256) for rhs in range(256)]
    return torch.tensor(values, dtype=torch.uint8, device=torch.device("cuda", device_index))


def _lcm_wrapped_byte_table_for(device, dtype):
    index = device.index
    if index is None:
        index = torch.cuda.current_device()
    return _lcm_wrapped_byte_table(index).view(dtype)


@triton.jit
def _rad2deg_kernel(input, output, n: tl.constexpr, block_size: tl.constexpr):
    offsets = tl.program_id(0) * block_size + tl.arange(0, block_size)
    mask = offsets < n
    value = tl.load(input + offsets, mask=mask, other=0.0)
    tl.store(output + offsets, value * 57.29577951308232, mask=mask)


def rad2deg_1d(input, output):
    if input.device.type != "cuda":
        return False
    if input.ndim != 1 or output.ndim != 1:
        return False
    if not input.is_contiguous() or not output.is_contiguous():
        return False
    if input.dtype not in (torch.float16, torch.float32, torch.float64):
        return False

    block_size = 1024
    num_warps = 4
    if is_metax_device(input):
        if input.dtype in (torch.float16, torch.bfloat16):
            block_size = 4096
            num_warps = 4
        elif input.dtype == torch.float64:
            block_size = 1024
            num_warps = 4
        else:
            block_size = 2048
            num_warps = 4
    elif input.dtype in (torch.float16, torch.bfloat16):
        block_size = 2048
    elif input.dtype == torch.float64:
        block_size = 512
        num_warps = 8

    grid = (triton.cdiv(input.numel(), block_size),)
    _rad2deg_kernel[grid](
        input,
        output,
        input.numel(),
        block_size=block_size,
        num_warps=num_warps,
    )
    return True


@triton.jit
def _lcm_i32_dynamic_kernel(input, other, output, table, n: tl.constexpr, block_size: tl.constexpr):
    offsets = tl.program_id(0) * block_size + tl.arange(0, block_size)
    mask = offsets < n
    lhs = tl.load(input + offsets, mask=mask, other=0).to(tl.int32)
    rhs = tl.load(other + offsets, mask=mask, other=0).to(tl.int32)

    lhs_min = (lhs < 0) & (-lhs == lhs)
    rhs_min = (rhs < 0) & (-rhs == rhs)
    min_overflow = lhs_min | rhs_min
    x = tl.abs(lhs)
    y = tl.abs(rhs)
    small = (x <= 255) & (y <= 255) & (~min_overflow)
    table_value = tl.load(table + x * 256 + y, mask=mask & small, other=0).to(tl.int32)

    a = tl.where(x >= y, x, y)
    b = tl.where(x >= y, y, x)
    b = tl.where(min_overflow | small, 0, b)
    a = tl.where(min_overflow | small, 1, a)

    iteration = 0
    while (tl.max(tl.where(mask, b, 0), axis=0) != 0) & (iteration < 64):
        safe_b = tl.where(b == 0, 1, b)
        r = a % safe_b
        a = tl.where(b == 0, a, b)
        b = tl.where(b == 0, b, r)
        iteration += 1

    safe_gcd = tl.where(a == 0, 1, a)
    product = ((x // safe_gcd) * y).to(tl.int32)
    value = tl.abs(product)
    value = tl.where(small, table_value, value)
    overflow_value = tl.where(lhs_min, lhs, rhs)
    value = tl.where(min_overflow, overflow_value, value)
    value = tl.where((lhs == 0) | (rhs == 0), 0, value)
    tl.store(output + offsets, value, mask=mask)


@triton.jit
def _lcm_i64_dynamic_kernel(input, other, output, table, n: tl.constexpr, block_size: tl.constexpr):
    offsets = tl.program_id(0) * block_size + tl.arange(0, block_size)
    mask = offsets < n
    lhs = tl.load(input + offsets, mask=mask, other=0).to(tl.int64)
    rhs = tl.load(other + offsets, mask=mask, other=0).to(tl.int64)

    lhs_min = (lhs < 0) & (-lhs == lhs)
    rhs_min = (rhs < 0) & (-rhs == rhs)
    min_overflow = lhs_min | rhs_min
    x = tl.abs(lhs)
    y = tl.abs(rhs)
    small = (x <= 255) & (y <= 255) & (~min_overflow)
    table_value = tl.load(table + x * 256 + y, mask=mask & small, other=0)

    a = tl.where(x >= y, x, y)
    b = tl.where(x >= y, y, x)
    b = tl.where(min_overflow | small, 0, b)
    a = tl.where(min_overflow | small, 1, a)

    iteration = 0
    while (tl.max(tl.where(mask, b, 0), axis=0) != 0) & (iteration < 96):
        safe_b = tl.where(b == 0, 1, b)
        r = a % safe_b
        a = tl.where(b == 0, a, b)
        b = tl.where(b == 0, b, r)
        iteration += 1

    safe_gcd = tl.where(a == 0, 1, a)
    value = tl.abs((x // safe_gcd) * y)
    value = tl.where(small, table_value, value)
    overflow_value = tl.where(lhs_min, lhs, rhs)
    value = tl.where(min_overflow, overflow_value, value)
    value = tl.where((lhs == 0) | (rhs == 0), 0, value)
    tl.store(output + offsets, value, mask=mask)


@triton.jit
def _lcm_metax_direct_kernel(input, other, output, ltable, n: tl.constexpr, block_size: tl.constexpr):
    offsets = tl.program_id(0) * block_size + tl.arange(0, block_size)
    mask = offsets < n
    lhs = tl.load(input + offsets, mask=mask, other=0).to(tl.int32)
    rhs = tl.load(other + offsets, mask=mask, other=0).to(tl.int32)
    x = tl.abs(lhs)
    y = tl.abs(rhs)
    value = tl.load(ltable + x * 256 + y, mask=mask, other=0)
    tl.store(output + offsets, value, mask=mask)


@triton.jit
def _lcm_metax_gcd_kernel(
    input, other, output, gtable, n: tl.constexpr,
    IS_I64: tl.constexpr, max_iter: tl.constexpr, block_size: tl.constexpr,
):
    offsets = tl.program_id(0) * block_size + tl.arange(0, block_size)
    mask = offsets < n
    if IS_I64:
        lhs = tl.load(input + offsets, mask=mask, other=0).to(tl.int64)
        rhs = tl.load(other + offsets, mask=mask, other=0).to(tl.int64)
    else:
        lhs = tl.load(input + offsets, mask=mask, other=0).to(tl.int32)
        rhs = tl.load(other + offsets, mask=mask, other=0).to(tl.int32)
    lhs_min = (lhs < 0) & (-lhs == lhs)
    rhs_min = (rhs < 0) & (-rhs == rhs)
    min_overflow = lhs_min | rhs_min
    x = tl.abs(lhs)
    y = tl.abs(rhs)
    small = (x <= 255) & (y <= 255) & (~min_overflow)
    # Small lane gathers GCD from a 1-byte table (narrow gather; MetaX IDIV is slow).
    g_small = tl.load(gtable + x * 256 + y, mask=mask & small, other=1).to(x.dtype)
    a = tl.where(x >= y, x, y)
    b = tl.where(x >= y, y, x)
    b = tl.where(min_overflow | small, 0, b)
    a = tl.where(min_overflow | small, 1, a)
    iteration = 0
    while (tl.max(tl.where(mask, b, 0), axis=0) != 0) & (iteration < max_iter):
        safe_b = tl.where(b == 0, 1, b)
        r = a % safe_b
        a = tl.where(b == 0, a, b)
        b = tl.where(b == 0, b, r)
        iteration += 1
    gcd_dyn = tl.where(a == 0, 1, a)
    gcd = tl.where(small, g_small, gcd_dyn)
    gcd = tl.where(gcd == 0, 1, gcd)
    value = tl.abs((x // gcd) * y)
    overflow_value = tl.where(lhs_min, lhs, rhs)
    value = tl.where(min_overflow, overflow_value, value)
    value = tl.where((lhs == 0) | (rhs == 0), 0, value)
    tl.store(output + offsets, value.to(output.dtype.element_ty), mask=mask)


def _lcm_metax_1d(input, other, output):
    # MetaX: torch lcm is light, but the NVIDIA int64 lcm table gather is wide;
    # u8/i8 use a 1-byte direct table, wider ints use a 1-byte gcd gather + dynamic Euclid.
    n = input.numel()
    dtype = input.dtype
    if dtype in (torch.uint8, torch.int8):
        block_size = 1024 if dtype == torch.uint8 else 2048
        grid = (triton.cdiv(n, block_size),)
        _lcm_metax_direct_kernel[grid](
            input, other, output,
            _lcm_wrapped_byte_table_for(input.device, dtype),
            n, block_size=block_size, num_warps=2,
        )
        return True
    if dtype in (torch.int16, torch.int32, torch.int64):
        is_i64 = dtype == torch.int64
        max_iter = 96 if is_i64 else 64
        if is_i64:
            block_size = 256
            num_warps = 2
        elif n < (1 << 24):
            block_size = 512
            num_warps = 8
        else:
            block_size = 1024
            num_warps = 4
        grid = (triton.cdiv(n, block_size),)
        _lcm_metax_gcd_kernel[grid](
            input, other, output,
            _gcd_byte_table_for(input.device),
            n, IS_I64=is_i64, max_iter=max_iter,
            block_size=block_size, num_warps=num_warps,
        )
        return True
    return False


def _lcm_iluvatar_1d(input, other, output):
    # Iluvatar/CoreX: the NVIDIA int64 LCM table gather is wide (512KB/2MB) and
    # the dynamic int64 Euclid kernel is IDIV-bound; the narrow 1-byte gcd/direct
    # gather kernels (shared with MetaX) cut bandwidth + division cost.
    # Microbench (MR-V100, N=16M): i32 1.60x, i16 1.27x, i64 2.97x, u8 1.26x, i8 1.64x.
    n = input.numel()
    dtype = input.dtype
    if dtype in (torch.uint8, torch.int8):
        block_size = 2048
        grid = (triton.cdiv(n, block_size),)
        _lcm_metax_direct_kernel[grid](
            input, other, output,
            _lcm_wrapped_byte_table_for(input.device, dtype),
            n, block_size=block_size, num_warps=4,
        )
        return True
    if dtype in (torch.int16, torch.int32, torch.int64):
        is_i64 = dtype == torch.int64
        max_iter = 96 if is_i64 else 64
        if is_i64:
            block_size = 256
            num_warps = 2
        else:
            block_size = 2048
            num_warps = 4
        grid = (triton.cdiv(n, block_size),)
        _lcm_metax_gcd_kernel[grid](
            input, other, output,
            _gcd_byte_table_for(input.device),
            n, IS_I64=is_i64, max_iter=max_iter,
            block_size=block_size, num_warps=num_warps,
        )
        return True
    return False


def lcm_1d(input, other, output):
    if input.device.type != "cuda":
        return False
    if input.ndim != 1 or other.ndim != 1 or output.ndim != 1:
        return False
    if not input.is_contiguous() or not other.is_contiguous() or not output.is_contiguous():
        return False

    if is_metax_device(input):
        return _lcm_metax_1d(input, other, output)

    if is_iluvatar_device(input):
        return _lcm_iluvatar_1d(input, other, output)

    n = input.numel()
    if input.dtype == torch.int32:
        block_size = 256
        grid = (triton.cdiv(n, block_size),)
        _lcm_i32_dynamic_kernel[grid](
            input,
            other,
            output,
            lcm_small_table(input.device),
            n,
            block_size=block_size,
            num_warps=4,
        )
        return True
    if input.dtype == torch.int64:
        block_size = 128
        grid = (triton.cdiv(n, block_size),)
        _lcm_i64_dynamic_kernel[grid](
            input,
            other,
            output,
            lcm_small_table(input.device),
            n,
            block_size=block_size,
            num_warps=4,
        )
        return True
    return False


@triton.jit
def _copysign_broadcast_2d_kernel(
    in_ptr, oth_ptr, out_ptr,
    rows, cols,
    in_sr, in_sc, oth_sr, oth_sc,
    IS_F64: tl.constexpr,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    rm = (pid_m * BLOCK_M + tl.arange(0, BLOCK_M))[:, None]
    cn = (pid_n * BLOCK_N + tl.arange(0, BLOCK_N))[None, :]
    mask = (rm < rows) & (cn < cols)
    a = tl.load(in_ptr + rm * in_sr + cn * in_sc, mask=mask, other=0.0)
    b = tl.load(oth_ptr + rm * oth_sr + cn * oth_sc, mask=mask, other=0.0)
    if IS_F64:
        one = tl.full((), 1, tl.int64)
        sign_bit = one << 63
        magn_mask = sign_bit - one
        ai = a.to(tl.int64, bitcast=True)
        bi = b.to(tl.int64, bitcast=True)
        out = ((ai & magn_mask) | (bi & sign_bit)).to(tl.float64, bitcast=True)
    else:
        one = tl.full((), 1, tl.int32)
        sign_bit = one << 31
        magn_mask = sign_bit - one
        ai = a.to(tl.int32, bitcast=True)
        bi = b.to(tl.int32, bitcast=True)
        out = ((ai & magn_mask) | (bi & sign_bit)).to(tl.float32, bitcast=True)
    tl.store(out_ptr + rm * cols + cn, out, mask=mask)


@triton.jit
def _nextafter_broadcast_2d_kernel(
    in_ptr, oth_ptr, out_ptr,
    rows, cols,
    in_sr, in_sc, oth_sr, oth_sc,
    IS_F64: tl.constexpr,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    rm = (pid_m * BLOCK_M + tl.arange(0, BLOCK_M))[:, None]
    cn = (pid_n * BLOCK_N + tl.arange(0, BLOCK_N))[None, :]
    mask = (rm < rows) & (cn < cols)
    a = tl.load(in_ptr + rm * in_sr + cn * in_sc, mask=mask, other=0.0)
    b = tl.load(oth_ptr + rm * oth_sr + cn * oth_sc, mask=mask, other=0.0)
    if IS_F64:
        int_ty = tl.int64
        flt_ty = tl.float64
        one = tl.full((), 1, tl.int64)
        sign_bit = one << 63
    else:
        int_ty = tl.int32
        flt_ty = tl.float32
        one = tl.full((), 1, tl.int32)
        sign_bit = one << 31
    zero = one - one
    a_i = a.to(int_ty, bitcast=True)
    b_i = b.to(int_ty, bitcast=True)
    is_nan = (a != a) | (b != b)
    eq = a == b
    is_zero = a == tl.zeros_like(a)
    b_sign = b_i & sign_bit
    zero_result = b_sign | one
    a_neg = a_i < zero
    a_lt_b = a < b
    step_up = a_neg ^ a_lt_b
    step = tl.where(step_up, one, -one)
    general = a_i + step
    nan_val = tl.full((), float("nan"), flt_ty)
    nan_bits = nan_val.to(int_ty, bitcast=True)
    result_i = tl.where(
        is_nan,
        nan_bits,
        tl.where(eq, b_i, tl.where(is_zero, zero_result, general)),
    )
    out = result_i.to(flt_ty, bitcast=True)
    tl.store(out_ptr + rm * cols + cn, out, mask=mask)


def _binary_broadcast_2d(kernel, input, other, output):
    # Reads stride-0 broadcast views directly to avoid materializing both
    # full-size operands via .contiguous() (the dominant cost on broadcast).
    if input.device.type != "cuda":
        return False
    if input.dtype not in (torch.float32, torch.float64):
        return False
    if input.ndim != 2 or other.ndim != 2 or output.ndim != 2:
        return False
    if not output.is_contiguous():
        return False
    rows, cols = output.shape
    if tuple(input.shape) != (rows, cols) or tuple(other.shape) != (rows, cols):
        return False

    in_sr, in_sc = input.stride()
    oth_sr, oth_sc = other.stride()

    block_m = 8
    block_n = 256
    num_warps = 8
    if output.dtype == torch.float64:
        block_n = 128
    if is_iluvatar_device(input):
        # CoreX: a single full row per program (BLOCK_M=1) keeps the broadcast
        # column read coalesced; the tall BLOCK_M=8 tile collapses to ~0.25x for
        # the heavier nextafter bit-walk while copysign stays >=1.7x either way.
        block_m = 1
        block_n = 512 if output.dtype == torch.float64 else 1024
        num_warps = 4
    grid = (triton.cdiv(rows, block_m), triton.cdiv(cols, block_n))
    kernel[grid](
        input, other, output,
        rows, cols,
        in_sr, in_sc, oth_sr, oth_sc,
        IS_F64=output.dtype == torch.float64,
        BLOCK_M=block_m, BLOCK_N=block_n,
        num_warps=num_warps,
    )
    return True


def copysign_broadcast_2d(input, other, output):
    return _binary_broadcast_2d(_copysign_broadcast_2d_kernel, input, other, output)


def nextafter_broadcast_2d(input, other, output):
    return _binary_broadcast_2d(_nextafter_broadcast_2d_kernel, input, other, output)


@triton.jit
def _copysign_1d_kernel(a_ptr, b_ptr, out_ptr, n, KIND: tl.constexpr, block_size: tl.constexpr):
    off = tl.program_id(0) * block_size + tl.arange(0, block_size)
    mask = off < n
    a = tl.load(a_ptr + off, mask=mask, other=0.0)
    b = tl.load(b_ptr + off, mask=mask, other=0.0)
    if KIND == 1:
        one = tl.full((), 1, tl.int64)
        sign_bit = one << 63
        magn_mask = sign_bit - one
        ai = a.to(tl.int64, bitcast=True)
        bi = b.to(tl.int64, bitcast=True)
        out = ((ai & magn_mask) | (bi & sign_bit)).to(tl.float64, bitcast=True)
    elif KIND == 2:
        one = tl.full((), 1, tl.int16)
        sign_bit = one << 15
        magn_mask = sign_bit - one
        ai = a.to(tl.int16, bitcast=True)
        bi = b.to(tl.int16, bitcast=True)
        out = ((ai & magn_mask) | (bi & sign_bit)).to(a.dtype, bitcast=True)
    else:
        one = tl.full((), 1, tl.int32)
        sign_bit = one << 31
        magn_mask = sign_bit - one
        ai = a.to(tl.int32, bitcast=True)
        bi = b.to(tl.int32, bitcast=True)
        out = ((ai & magn_mask) | (bi & sign_bit)).to(tl.float32, bitcast=True)
    tl.store(out_ptr + off, out, mask=mask)


@triton.jit
def _nextafter_1d_kernel(a_ptr, b_ptr, out_ptr, n, KIND: tl.constexpr, block_size: tl.constexpr):
    off = tl.program_id(0) * block_size + tl.arange(0, block_size)
    mask = off < n
    a = tl.load(a_ptr + off, mask=mask, other=0.0)
    b = tl.load(b_ptr + off, mask=mask, other=0.0)
    if KIND == 1:
        int_ty = tl.int64
        flt_ty = tl.float64
        one = tl.full((), 1, tl.int64)
        sign_bit = one << 63
    elif KIND == 2:
        int_ty = tl.int16
        flt_ty = a.dtype
        one = tl.full((), 1, tl.int16)
        sign_bit = one << 15
    else:
        int_ty = tl.int32
        flt_ty = tl.float32
        one = tl.full((), 1, tl.int32)
        sign_bit = one << 31
    zero = one - one
    a_i = a.to(int_ty, bitcast=True)
    b_i = b.to(int_ty, bitcast=True)
    is_nan = (a != a) | (b != b)
    eq = a == b
    is_zero = a == tl.zeros_like(a)
    b_sign = b_i & sign_bit
    zero_result = b_sign | one
    a_neg = a_i < zero
    a_lt_b = a < b
    step_up = a_neg ^ a_lt_b
    step = tl.where(step_up, one, -one)
    general = a_i + step
    nan_val = tl.full((), float("nan"), flt_ty)
    nan_bits = nan_val.to(int_ty, bitcast=True)
    result_i = tl.where(
        is_nan,
        nan_bits,
        tl.where(eq, b_i, tl.where(is_zero, zero_result, general)),
    )
    out = result_i.to(flt_ty, bitcast=True)
    tl.store(out_ptr + off, out, mask=mask)


def _sign_kind(dtype):
    if dtype == torch.float64:
        return 1
    if dtype in (torch.float16, torch.bfloat16):
        return 2
    if dtype == torch.float32:
        return 0
    return None


def _binary_bitop_1d(kernel, input, other, output, allow_half):
    # MetaX: dedicated bit-manipulation kernel for contiguous 1D views; the
    # generic NineToothed elementwise path is bandwidth-suboptimal here.
    if not is_metax_device(input):
        return False
    if input.ndim != 1 or other.ndim != 1 or output.ndim != 1:
        return False
    if not input.is_contiguous() or not other.is_contiguous() or not output.is_contiguous():
        return False
    kind = _sign_kind(output.dtype)
    if kind is None:
        return False
    if kind == 2 and not allow_half:
        return False
    block_size = 2048
    num_warps = 4
    grid = (triton.cdiv(input.numel(), block_size),)
    kernel[grid](
        input, other, output,
        input.numel(),
        KIND=kind,
        block_size=block_size,
        num_warps=num_warps,
    )
    return True


def copysign_1d(input, other, output):
    return _binary_bitop_1d(_copysign_1d_kernel, input, other, output, allow_half=True)


def nextafter_1d(input, other, output):
    # Half nextafter via int16 bit-walk fails to compile on the MetaX backend
    # (fcmp on i16); leave half to the existing NineToothed path.
    return _binary_bitop_1d(_nextafter_1d_kernel, input, other, output, allow_half=False)


@triton.jit
def _nextafter_half_i32_kernel(a_ptr, b_ptr, out_ptr, n, block_size: tl.constexpr):
    off = tl.program_id(0) * block_size + tl.arange(0, block_size)
    mask = off < n
    a = tl.load(a_ptr + off, mask=mask, other=0.0)
    b = tl.load(b_ptr + off, mask=mask, other=0.0)
    # fp16/bf16 -> fp32 is lossless, so compare in fp32 (CoreX cannot fcmp on i16),
    # but walk the actual 16-bit pattern held zero-extended in int32.
    a32 = a.to(tl.float32)
    b32 = b.to(tl.float32)
    a_bits = a.to(tl.int16, bitcast=True).to(tl.int32) & 0xFFFF
    b_bits = b.to(tl.int16, bitcast=True).to(tl.int32) & 0xFFFF
    sign_bit = tl.full((), 0x8000, tl.int32)
    one = tl.full((), 1, tl.int32)
    is_nan = (a32 != a32) | (b32 != b32)
    eq = a32 == b32
    is_zero = a32 == 0.0
    b_sign = b_bits & sign_bit
    zero_result = b_sign | one
    a_neg = (a_bits & sign_bit) != 0
    a_lt_b = a32 < b32
    step_up = a_neg ^ a_lt_b
    step = tl.where(step_up, one, -one)
    general = a_bits + step
    nan_const = tl.full((), float("nan"), a.dtype)
    nan_bits = nan_const.to(tl.int16, bitcast=True).to(tl.int32) & 0xFFFF
    result = tl.where(
        is_nan,
        nan_bits,
        tl.where(eq, b_bits, tl.where(is_zero, zero_result, general)),
    )
    out = (result & 0xFFFF).to(tl.int16).to(a.dtype, bitcast=True)
    tl.store(out_ptr + off, out, mask=mask)


def nextafter_half_iluvatar_1d(input, other, output):
    # Iluvatar/CoreX: fp16/bf16 nextafter. The NineToothed int16 bit-walk does not
    # compile (i16 fcmp) and the fp32 round-trip path is wrong for fp16 subnormals
    # (it walks fp32-granular neighbors). This int32-domain bit-walk is exact:
    # verified 0 mismatches (NaN-aware) on MR-V100 across subnormals/inf/nan.
    if not is_iluvatar_device(input):
        return False
    if input.ndim != 1 or other.ndim != 1 or output.ndim != 1:
        return False
    if not input.is_contiguous() or not other.is_contiguous() or not output.is_contiguous():
        return False
    # output.dtype may be torch.dtype or infinicore.dtype (InfiniCore rebinds the
    # op-level torch global), so match by dtype name rather than identity.
    if str(output.dtype).rsplit(".", 1)[-1] not in ("float16", "bfloat16", "half"):
        return False
    block_size = 2048
    grid = (triton.cdiv(input.numel(), block_size),)
    _nextafter_half_i32_kernel[grid](
        input, other, output,
        input.numel(),
        block_size=block_size,
        num_warps=4,
    )
    return True


@triton.jit
def _moveaxis_matrix_transpose_kernel(
    input,
    output,
    rows: tl.constexpr,
    cols: tl.constexpr,
    block_m: tl.constexpr,
    block_n: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    offs_m = pid_m * block_m + tl.arange(0, block_m)
    offs_n = pid_n * block_n + tl.arange(0, block_n)
    row = offs_m[:, None]
    col = offs_n[None, :]
    mask = (offs_m[:, None] < rows) & (offs_n[None, :] < cols)
    value = tl.load(input + row * cols + col, mask=mask)
    tl.store(output + col * rows + row, value, mask=mask)


@triton.jit
def _moveaxis_3d_swap01_kernel(
    input,
    output,
    dim0: tl.constexpr,
    dim1: tl.constexpr,
    dim2: tl.constexpr,
    block_0: tl.constexpr,
    block_1: tl.constexpr,
    block_2: tl.constexpr,
):
    pid_0 = tl.program_id(0)
    pid_1 = tl.program_id(1)
    pid_2 = tl.program_id(2)
    offs_0 = pid_0 * block_0 + tl.arange(0, block_0)
    offs_1 = pid_1 * block_1 + tl.arange(0, block_1)
    offs_2 = pid_2 * block_2 + tl.arange(0, block_2)
    i0 = offs_0[:, None, None]
    i1 = offs_1[None, :, None]
    i2 = offs_2[None, None, :]
    mask = (
        (offs_0[:, None, None] < dim0)
        & (offs_1[None, :, None] < dim1)
        & (offs_2[None, None, :] < dim2)
    )
    value = tl.load(input + (i0 * dim1 + i1) * dim2 + i2, mask=mask)
    tl.store(output + (i1 * dim0 + i0) * dim2 + i2, value, mask=mask)


@triton.jit
def _moveaxis_3d_swap12_kernel(
    input,
    output,
    dim0: tl.constexpr,
    dim1: tl.constexpr,
    dim2: tl.constexpr,
    block_0: tl.constexpr,
    block_1: tl.constexpr,
    block_2: tl.constexpr,
):
    pid_0 = tl.program_id(0)
    pid_1 = tl.program_id(1)
    pid_2 = tl.program_id(2)
    offs_0 = pid_0 * block_0 + tl.arange(0, block_0)
    offs_1 = pid_1 * block_1 + tl.arange(0, block_1)
    offs_2 = pid_2 * block_2 + tl.arange(0, block_2)
    i0 = offs_0[:, None, None]
    i1 = offs_1[None, :, None]
    i2 = offs_2[None, None, :]
    mask = (
        (offs_0[:, None, None] < dim0)
        & (offs_1[None, :, None] < dim1)
        & (offs_2[None, None, :] < dim2)
    )
    value = tl.load(input + (i0 * dim1 + i1) * dim2 + i2, mask=mask)
    tl.store(output + (i0 * dim2 + i2) * dim1 + i1, value, mask=mask)


@triton.jit
def _moveaxis_4d_swap12_kernel(
    input,
    output,
    dim0: tl.constexpr,
    dim1: tl.constexpr,
    dim2: tl.constexpr,
    dim3: tl.constexpr,
    block_1: tl.constexpr,
    block_2: tl.constexpr,
    block_3: tl.constexpr,
):
    pid_0 = tl.program_id(0)
    pid_1 = tl.program_id(1)
    pid_2 = tl.program_id(2)
    offs_1 = pid_1 * block_1 + tl.arange(0, block_1)
    offs_2 = pid_2 * block_2 + tl.arange(0, block_2)
    offs_3 = tl.arange(0, block_3)
    i1 = offs_1[:, None, None]
    i2 = offs_2[None, :, None]
    i3 = offs_3[None, None, :]
    mask = (
        (offs_1[:, None, None] < dim1)
        & (offs_2[None, :, None] < dim2)
        & (offs_3[None, None, :] < dim3)
    )
    value = tl.load(input + ((pid_0 * dim1 + i1) * dim2 + i2) * dim3 + i3, mask=mask)
    tl.store(output + ((pid_0 * dim2 + i2) * dim1 + i1) * dim3 + i3, value, mask=mask)


@triton.jit
def _channel_shuffle_kernel(
    input,
    output,
    total: tl.constexpr,
    channels: tl.constexpr,
    inner_size: tl.constexpr,
    groups: tl.constexpr,
    stride_0: tl.constexpr,
    stride_1: tl.constexpr,
    stride_2: tl.constexpr,
    stride_3: tl.constexpr,
    stride_4: tl.constexpr,
    size_3: tl.constexpr,
    size_4: tl.constexpr,
    ndim: tl.constexpr,
    block_size: tl.constexpr,
):
    offsets = tl.program_id(0) * block_size + tl.arange(0, block_size)
    mask = offsets < total

    inner = offsets % inner_size
    channel_out = (offsets // inner_size) % channels
    batch = offsets // (channels * inner_size)
    channels_per_group = channels // groups
    channel_in = (channel_out % groups) * channels_per_group + channel_out // groups

    input_offsets = batch * stride_0 + channel_in * stride_1
    if ndim == 3:
        input_offsets += inner * stride_2
    elif ndim == 4:
        dim_2 = inner // size_3
        dim_3 = inner - dim_2 * size_3
        input_offsets += dim_2 * stride_2 + dim_3 * stride_3
    else:
        dim_2 = inner // (size_3 * size_4)
        rest = inner - dim_2 * size_3 * size_4
        dim_3 = rest // size_4
        dim_4 = rest - dim_3 * size_4
        input_offsets += dim_2 * stride_2 + dim_3 * stride_3 + dim_4 * stride_4

    value = tl.load(input + input_offsets, mask=mask)
    tl.store(output + offsets, value, mask=mask)


@triton.jit
def _channel_shuffle_contiguous_kernel(
    input,
    output,
    channels: tl.constexpr,
    inner_size: tl.constexpr,
    groups: tl.constexpr,
    block_size: tl.constexpr,
):
    pid_channel = tl.program_id(0)
    pid_inner = tl.program_id(1)
    batch = pid_channel // channels
    channel_out = pid_channel - batch * channels
    channels_per_group = channels // groups
    channel_in = (channel_out % groups) * channels_per_group + channel_out // groups

    inner = pid_inner * block_size + tl.arange(0, block_size)
    mask = inner < inner_size
    input_base = (batch * channels + channel_in) * inner_size
    output_base = (batch * channels + channel_out) * inner_size
    value = tl.load(input + input_base + inner, mask=mask)
    tl.store(output + output_base + inner, value, mask=mask)


def _supported_moveaxis_dtype(dtype):
    return dtype in (torch.float32, torch.float16, torch.int32)


def _moveaxis_matrix_transpose_supported(input, perm):
    shape = tuple(input.shape)
    ndim = len(shape)

    if is_metax_device(input):
        if input.dtype == torch.int32:
            return True
        if input.dtype == torch.float16 and ndim == 3:
            return True
        if input.dtype == torch.float32 and ndim == 4:
            return True
        return False

    if is_iluvatar_device(input):
        if input.dtype == torch.float16 and ndim == 2:
            return False
        if input.dtype == torch.float32 and ndim == 2 and shape[0] < 8192:
            return False
        return True

    return True


def _moveaxis_3d_swap01_supported(input):
    if is_iluvatar_device(input):
        return False
    return not is_metax_device(input) or input.dtype == torch.float32


def _moveaxis_3d_swap12_supported(input):
    if is_iluvatar_device(input):
        return False
    return not is_metax_device(input) or input.dtype == torch.float32


def _moveaxis_4d_swap12_supported(input):
    return not is_corex_or_metax_device(input) or input.dtype == torch.float32


def moveaxis_fast_path(input, output, perm):
    if input.device.type != "cuda":
        return False
    if not input.is_contiguous() or not output.is_contiguous():
        return False
    if not _supported_moveaxis_dtype(input.dtype):
        return False

    shape = tuple(input.shape)
    ndim = len(shape)
    perm = tuple(perm)

    if perm == tuple(range(1, ndim)) + (0,):
        if not _moveaxis_matrix_transpose_supported(input, perm):
            return False
        rows = shape[0]
        cols = input.numel() // rows
        block_m = 8 if rows >= 1024 else 16
        block_n = 64
        grid = (triton.cdiv(rows, block_m), triton.cdiv(cols, block_n))
        _moveaxis_matrix_transpose_kernel[grid](
            input,
            output,
            rows,
            cols,
            block_m=block_m,
            block_n=block_n,
            num_warps=8,
        )
        return True

    if ndim == 3 and perm == (1, 0, 2):
        if not _moveaxis_3d_swap01_supported(input):
            return False
        dim0, dim1, dim2 = shape
        block_0 = 1
        block_1 = 8
        block_2 = 128 if dim2 >= 128 else 64
        grid = (
            triton.cdiv(dim0, block_0),
            triton.cdiv(dim1, block_1),
            triton.cdiv(dim2, block_2),
        )
        _moveaxis_3d_swap01_kernel[grid](
            input,
            output,
            dim0,
            dim1,
            dim2,
            block_0=block_0,
            block_1=block_1,
            block_2=block_2,
            num_warps=4,
        )
        return True

    if ndim == 3 and perm == (0, 2, 1):
        if not _moveaxis_3d_swap12_supported(input):
            return False
        dim0, dim1, dim2 = shape
        block_0 = 1
        block_1 = 32
        block_2 = 32
        grid = (
            triton.cdiv(dim0, block_0),
            triton.cdiv(dim1, block_1),
            triton.cdiv(dim2, block_2),
        )
        _moveaxis_3d_swap12_kernel[grid](
            input,
            output,
            dim0,
            dim1,
            dim2,
            block_0=block_0,
            block_1=block_1,
            block_2=block_2,
            num_warps=8,
        )
        return True

    if ndim == 4 and perm == (0, 2, 1, 3):
        if not _moveaxis_4d_swap12_supported(input):
            return False
        dim0, dim1, dim2, dim3 = shape
        block_1 = 16
        block_2 = 4
        block_3 = 32 if dim3 >= 32 else 16
        grid = (
            dim0,
            triton.cdiv(dim1, block_1),
            triton.cdiv(dim2, block_2),
        )
        _moveaxis_4d_swap12_kernel[grid](
            input,
            output,
            dim0,
            dim1,
            dim2,
            dim3,
            block_1=block_1,
            block_2=block_2,
            block_3=block_3,
            num_warps=4,
        )
        return True

    return False


@triton.jit
def _im2col_kernel(
    input_ptr,
    output_ptr,
    N: tl.constexpr,
    C: tl.constexpr,
    H: tl.constexpr,
    W: tl.constexpr,
    kH: tl.constexpr,
    kW: tl.constexpr,
    sH: tl.constexpr,
    sW: tl.constexpr,
    pH: tl.constexpr,
    pW: tl.constexpr,
    dH: tl.constexpr,
    dW: tl.constexpr,
    outH: tl.constexpr,
    outW: tl.constexpr,
    BLOCK: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)

    kHW = kH * kW
    outHW = outH * outW
    ckp = C * kHW * outHW

    n = offs // ckp
    rem = offs % ckp
    c = rem // (kHW * outHW)
    rem = rem % (kHW * outHW)
    kh = rem // (kW * outHW)
    rem = rem % (kW * outHW)
    kw = rem // outHW
    rem = rem % outHW
    oh = rem // outW
    ow = rem % outW

    h_in = oh * sH + kh * dH - pH
    w_in = ow * sW + kw * dW - pW

    mask = offs < N * ckp
    in_bounds = (h_in >= 0) & (h_in < H) & (w_in >= 0) & (w_in < W)
    read_mask = mask & in_bounds

    val = tl.load(
        input_ptr + ((n * C + c) * H + h_in) * W + w_in,
        mask=read_mask,
        other=0.0,
    )

    out_flat = n * ckp + (c * kHW + kh * kW + kw) * outHW + oh * outW + ow
    tl.store(output_ptr + out_flat, val, mask=mask)


def im2col_fast_path(input, output, kernel_size, dilation, padding, stride):
    if input.device.type != "cuda":
        return False
    if input.ndim != 4:
        return False
    if not input.is_contiguous() or not output.is_contiguous():
        return False
    if not is_iluvatar_device(input):
        return False
    if input.dtype not in (torch.float32, torch.float16, torch.bfloat16, torch.float64, torch.int32):
        return False

    N, C, H, W = input.shape
    kH, kW = kernel_size
    sH, sW = stride
    pH, pW = padding
    dH, dW = dilation

    block_size = 1024
    num_warps = 8
    if input.dtype == torch.float64:
        block_size = 512
        num_warps = 4
    elif input.dtype == torch.float16:
        block_size = 2048

    outH = (H + 2 * pH - dH * (kH - 1) - 1) // sH + 1
    outW = (W + 2 * pW - dW * (kW - 1) - 1) // sW + 1

    grid = (triton.cdiv(N * C * kH * kW * outH * outW, block_size),)

    _im2col_kernel[grid](
        input,
        output,
        N, C, H, W,
        kH, kW,
        sH, sW,
        pH, pW,
        dH, dW,
        outH, outW,
        BLOCK=block_size,
        num_warps=num_warps,
    )
    return True


def channel_shuffle_fast_path(input, output, groups):
    if input.device.type != "cuda":
        return False
    if input.ndim not in (3, 4, 5):
        return False
    if not output.is_contiguous():
        return False

    shape = tuple(input.shape)
    channels = shape[1]
    inner_size = 1
    for size in shape[2:]:
        inner_size *= size

    use_contiguous_kernel = input.is_contiguous()
    if is_metax_device(input) and input.dtype == torch.float16:
        use_contiguous_kernel = False
    if is_iluvatar_device(input) and input.dtype == torch.float32 and inner_size % 1024 != 0:
        use_contiguous_kernel = False

    if use_contiguous_kernel:
        block_size = 1024
        num_warps = 4
        if is_metax_device(input) and input.dtype in (torch.float16, torch.bfloat16):
            block_size = 2048
        if not is_corex_or_metax_device(input) and input.dtype == torch.float64:
            block_size = 2048
        grid = (shape[0] * channels, triton.cdiv(inner_size, block_size))
        _channel_shuffle_contiguous_kernel[grid](
            input,
            output,
            channels,
            inner_size,
            groups,
            block_size=block_size,
            num_warps=num_warps,
        )
        return True

    block_size = 1024
    num_warps = 4
    if is_metax_device(input):
        if input.dtype == torch.float32:
            block_size = 256
        elif input.dtype == torch.float16:
            if input.is_contiguous():
                block_size = 2048
            else:
                block_size = 512
                num_warps = 8
    elif is_iluvatar_device(input) and input.dtype == torch.float32 and input.is_contiguous():
        if inner_size % 1024 != 0:
            block_size = 512
            num_warps = 8

    grid = (triton.cdiv(input.numel(), block_size),)
    strides = input.stride()
    stride_3 = strides[3] if input.ndim >= 4 else 0
    stride_4 = strides[4] if input.ndim >= 5 else 0
    size_3 = shape[3] if input.ndim >= 4 else 1
    size_4 = shape[4] if input.ndim >= 5 else 1

    _channel_shuffle_kernel[grid](
        input,
        output,
        input.numel(),
        channels,
        inner_size,
        groups,
        strides[0],
        strides[1],
        strides[2],
        stride_3,
        stride_4,
        size_3,
        size_4,
        input.ndim,
        block_size=block_size,
        num_warps=num_warps,
    )
    return True


@triton.jit
def _frac_kernel(in_ptr, out_ptr, n_elements, UPCAST: tl.constexpr, BLOCK_SIZE: tl.constexpr):
    offsets = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(in_ptr + offsets, mask=mask, other=0.0)
    if UPCAST:
        value = x.to(tl.float32)
    else:
        value = x
    truncated = tl.where(value < 0, tl.math.ceil(value), tl.floor(value))
    result = (value - truncated).to(x.dtype)
    tl.store(out_ptr + offsets, result, mask=mask)


def frac_1d(input, output):
    if input.device.type != "cuda":
        return False
    if is_iluvatar_device(input):
        return False
    if input.ndim != 1 or output.ndim != 1:
        return False
    if not input.is_contiguous() or not output.is_contiguous():
        return False

    upcast = input.dtype in (torch.float16, torch.bfloat16)
    block_size = 4096
    num_warps = 8
    if input.dtype in (torch.float16, torch.bfloat16):
        block_size = 8192
    elif input.dtype == torch.float64:
        block_size = 2048

    grid = (triton.cdiv(input.numel(), block_size),)
    _frac_kernel[grid](
        input,
        output,
        input.numel(),
        UPCAST=upcast,
        BLOCK_SIZE=block_size,
        num_warps=num_warps,
    )
    return True


@triton.jit
def _scatter_add_kernel_1d(out_ptr, index_ptr, src_ptr, dim_size, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    idx = tl.load(index_ptr + offsets, mask=mask, other=0).to(tl.int64)
    valid = mask & (idx >= 0) & (idx < dim_size)
    val = tl.load(src_ptr + offsets, mask=mask, other=0.0)
    tl.atomic_add(out_ptr + idx, val, sem="relaxed", mask=valid)


@triton.jit
def _scatter_add_kernel_nd(
    out_ptr, index_ptr, src_ptr,
    mid, inner, out_mid,
    n_elements, BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    idx = tl.load(index_ptr + offsets, mask=mask, other=0).to(tl.int64)
    val = tl.load(src_ptr + offsets, mask=mask, other=0.0)

    outer = offsets // (mid * inner)
    inner_pos = offsets % inner
    valid = mask & (idx >= 0) & (idx < out_mid)
    target = outer * (out_mid * inner) + idx * inner + inner_pos
    tl.atomic_add(out_ptr + target, val, sem="relaxed", mask=valid)


def scatter_add_into(output, index, src, dim, shape):
    if output.device.type != "cuda":
        return False
    if is_iluvatar_device(output):
        return False
    if output.ndim == 0:
        return False
    if not output.is_contiguous() or not index.is_contiguous() or not src.is_contiguous():
        return False
    if tuple(index.shape) != tuple(src.shape):
        return False

    n_elements = index.numel()
    if n_elements == 0:
        return True
    block_size = 1024
    num_warps = 4

    if output.ndim == 1:
        _scatter_add_kernel_1d[(triton.cdiv(n_elements, block_size),)](
            output, index, src,
            shape[0], n_elements,
            BLOCK_SIZE=block_size, num_warps=num_warps,
        )
        return True

    if tuple(src.shape) != tuple(shape):
        return False

    inner = 1
    for size in shape[dim + 1:]:
        inner *= size
    _scatter_add_kernel_nd[(triton.cdiv(n_elements, block_size),)](
        output, index, src,
        shape[dim], inner, shape[dim],
        n_elements,
        BLOCK_SIZE=block_size, num_warps=num_warps,
    )
    return True


@triton.jit
def _fractional_max_pool2d_kernel(
    out_ptr, indices_ptr, in_ptr, random_ptr,
    N: tl.constexpr, C: tl.constexpr,
    inH: tl.constexpr, inW: tl.constexpr,
    outH: tl.constexpr, outW: tl.constexpr,
    kH: tl.constexpr, kW: tl.constexpr,
    stride_n, stride_c, stride_h, stride_w,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    base = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = base < N * C * outH * outW

    n = base // (C * outH * outW)
    rem = base % (C * outH * outW)
    c = rem // (outH * outW)
    rem = rem % (outH * outW)
    oh = rem // outW
    ow = rem % outW

    sample_h = tl.load(random_ptr + n * C * 2 + c * 2 + 0, mask=mask, other=0.0)
    sample_w = tl.load(random_ptr + n * C * 2 + c * 2 + 1, mask=mask, other=0.0)

    if outH == 1:
        start_h = tl.zeros((BLOCK_SIZE,), dtype=tl.int32)
    else:
        alpha_h = (inH - kH) / (outH - 1)
        sample_h_f32 = sample_h.to(tl.float32)
        oh_f32 = oh.to(tl.float32)
        start_h_f32 = tl.floor((oh_f32 + sample_h_f32) * alpha_h)
        start_h_f32 -= tl.floor(sample_h_f32 * alpha_h)
        start_h = start_h_f32.to(tl.int32)
        start_h = tl.where(oh == outH - 1, inH - kH, start_h)

    if outW == 1:
        start_w = tl.zeros((BLOCK_SIZE,), dtype=tl.int32)
    else:
        alpha_w = (inW - kW) / (outW - 1)
        sample_w_f32 = sample_w.to(tl.float32)
        ow_f32 = ow.to(tl.float32)
        start_w_f32 = tl.floor((ow_f32 + sample_w_f32) * alpha_w)
        start_w_f32 -= tl.floor(sample_w_f32 * alpha_w)
        start_w = start_w_f32.to(tl.int32)
        start_w = tl.where(ow == outW - 1, inW - kW, start_w)

    in_base = n * stride_n + c * stride_c
    neg_inf = tl.cast(-1e30, tl.float32)
    max_val = tl.full((BLOCK_SIZE,), neg_inf, dtype=tl.float32)
    max_idx = tl.zeros((BLOCK_SIZE,), dtype=tl.int32)

    for kh in range(kH):
        for kw in range(kW):
            ih = start_h + kh
            iw = start_w + kw
            val = tl.load(
                in_ptr + in_base + ih * stride_h + iw * stride_w,
                mask=mask,
                other=neg_inf,
            )
            val_f32 = val.to(tl.float32)
            local_idx = (ih * inW + iw).to(tl.int32)
            take = (val_f32 > max_val) | (val_f32 != val_f32)
            max_val = tl.where(take, val_f32, max_val)
            max_idx = tl.where(take, local_idx, max_idx)

    tl.store(out_ptr + base, max_val, mask=mask)
    tl.store(indices_ptr + base, max_idx, mask=mask)


def fractional_max_pool2d_fast(output, indices, input, random_samples, kernel_size, output_size):
    if input.device.type != "cuda":
        return False
    if is_iluvatar_device(input):
        return False
    if input.ndim not in (3, 4):
        return False
    if not output.is_contiguous():
        return False
    if not random_samples.is_contiguous():
        return False

    squeeze_batch = input.ndim == 3
    N = 1 if squeeze_batch else input.shape[0]
    C = input.shape[-3]
    inH, inW = input.shape[-2], input.shape[-1]
    kH, kW = kernel_size
    outH, outW = output_size

    stride_n = 0 if squeeze_batch else input.stride(0)
    stride_c = input.stride(-3)
    stride_h = input.stride(-2)
    stride_w = input.stride(-1)

    block_size = 256
    num_warps = 4
    if input.dtype in (torch.float16, torch.bfloat16):
        block_size = 512
    elif input.dtype == torch.float64:
        block_size = 128
        num_warps = 8

    grid = (triton.cdiv(N * C * outH * outW, block_size),)
    _fractional_max_pool2d_kernel[grid](
        output, indices, input, random_samples,
        N, C, inH, inW, outH, outW, kH, kW,
        stride_n, stride_c, stride_h, stride_w,
        BLOCK_SIZE=block_size,
        num_warps=num_warps,
    )
    return True


@triton.jit
def _fractional_max_pool3d_kernel(
    out_ptr, indices_ptr, in_ptr, random_ptr,
    N: tl.constexpr, C: tl.constexpr,
    inT: tl.constexpr, inH: tl.constexpr, inW: tl.constexpr,
    outT: tl.constexpr, outH: tl.constexpr, outW: tl.constexpr,
    kT: tl.constexpr, kH: tl.constexpr, kW: tl.constexpr,
    stride_n, stride_c, stride_t, stride_h, stride_w,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    base = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = base < N * C * outT * outH * outW

    n = base // (C * outT * outH * outW)
    rem = base % (C * outT * outH * outW)
    c = rem // (outT * outH * outW)
    rem = rem % (outT * outH * outW)
    ot = rem // (outH * outW)
    rem = rem % (outH * outW)
    oh = rem // outW
    ow = rem % outW

    sample_t = tl.load(random_ptr + n * C * 3 + c * 3 + 0, mask=mask, other=0.0)
    sample_h = tl.load(random_ptr + n * C * 3 + c * 3 + 1, mask=mask, other=0.0)
    sample_w = tl.load(random_ptr + n * C * 3 + c * 3 + 2, mask=mask, other=0.0)

    if outT == 1:
        start_t = tl.zeros((BLOCK_SIZE,), dtype=tl.int32)
    else:
        alpha_t = (inT - kT) / (outT - 1)
        ot_f32 = ot.to(tl.float32)
        st_f32 = sample_t.to(tl.float32)
        start_t_f32 = tl.floor((ot_f32 + st_f32) * alpha_t)
        start_t_f32 -= tl.floor(st_f32 * alpha_t)
        start_t = start_t_f32.to(tl.int32)
        start_t = tl.where(ot == outT - 1, inT - kT, start_t)

    if outH == 1:
        start_h = tl.zeros((BLOCK_SIZE,), dtype=tl.int32)
    else:
        alpha_h = (inH - kH) / (outH - 1)
        oh_f32 = oh.to(tl.float32)
        sh_f32 = sample_h.to(tl.float32)
        start_h_f32 = tl.floor((oh_f32 + sh_f32) * alpha_h)
        start_h_f32 -= tl.floor(sh_f32 * alpha_h)
        start_h = start_h_f32.to(tl.int32)
        start_h = tl.where(oh == outH - 1, inH - kH, start_h)

    if outW == 1:
        start_w = tl.zeros((BLOCK_SIZE,), dtype=tl.int32)
    else:
        alpha_w = (inW - kW) / (outW - 1)
        ow_f32 = ow.to(tl.float32)
        sw_f32 = sample_w.to(tl.float32)
        start_w_f32 = tl.floor((ow_f32 + sw_f32) * alpha_w)
        start_w_f32 -= tl.floor(sw_f32 * alpha_w)
        start_w = start_w_f32.to(tl.int32)
        start_w = tl.where(ow == outW - 1, inW - kW, start_w)

    in_base = n * stride_n + c * stride_c
    neg_inf = tl.cast(-1e30, tl.float32)
    max_val = tl.full((BLOCK_SIZE,), neg_inf, dtype=tl.float32)
    max_idx = tl.zeros((BLOCK_SIZE,), dtype=tl.int32)

    for kt in range(kT):
        for kh in range(kH):
            for kw in range(kW):
                it = start_t + kt
                ih = start_h + kh
                iw = start_w + kw
                val = tl.load(
                    in_ptr + in_base + it * stride_t + ih * stride_h + iw * stride_w,
                    mask=mask,
                    other=neg_inf,
                )
                val_f32 = val.to(tl.float32)
                local_idx = ((it * inH + ih) * inW + iw).to(tl.int32)
                take = (val_f32 > max_val) | (val_f32 != val_f32)
                max_val = tl.where(take, val_f32, max_val)
                max_idx = tl.where(take, local_idx, max_idx)

    tl.store(out_ptr + base, max_val, mask=mask)
    tl.store(indices_ptr + base, max_idx, mask=mask)


def fractional_max_pool3d_fast(output, indices, input, random_samples, kernel_size, output_size):
    if input.device.type != "cuda":
        return False
    if is_iluvatar_device(input):
        return False
    if input.ndim not in (4, 5):
        return False
    if not output.is_contiguous():
        return False
    if not random_samples.is_contiguous():
        return False

    squeeze_batch = input.ndim == 4
    N = 1 if squeeze_batch else input.shape[0]
    C = input.shape[-4]
    inT, inH, inW = input.shape[-3], input.shape[-2], input.shape[-1]
    kT, kH, kW = kernel_size
    outT, outH, outW = output_size

    stride_n = 0 if squeeze_batch else input.stride(0)
    stride_c = input.stride(-4)
    stride_t = input.stride(-3)
    stride_h = input.stride(-2)
    stride_w = input.stride(-1)

    block_size = 128
    num_warps = 4
    if input.dtype in (torch.float16, torch.bfloat16):
        block_size = 256
    elif input.dtype == torch.float64:
        block_size = 64
        num_warps = 8

    grid = (triton.cdiv(N * C * outT * outH * outW, block_size),)
    _fractional_max_pool3d_kernel[grid](
        output, indices, input, random_samples,
        N, C, inT, inH, inW, outT, outH, outW, kT, kH, kW,
        stride_n, stride_c, stride_t, stride_h, stride_w,
        BLOCK_SIZE=block_size,
        num_warps=num_warps,
    )
    return True


# === lgamma MetaX vendor triton ===

@triton.jit
def _lgamma_kernel(input_ptr, output_ptr, n: tl.constexpr, block_size: tl.constexpr):
    offsets = tl.program_id(0) * block_size + tl.arange(0, block_size)
    mask = offsets < n
    x = tl.load(input_ptr + offsets, mask=mask, other=1.0).to(tl.float32)
    # Numerical Recipes gammln (Lanczos g=5, N=6); reflection only for blocks with x<=0 or +inf.
    coeff0 = 1.000000000190015
    coeff1 = 76.18009172947146
    coeff2 = -86.50532032941677
    coeff3 = 24.01409824083091
    coeff4 = -1.231739572450155
    coeff5 = 0.001208650973866179
    coeff6 = -0.000005395239384953
    log_sqrt_2pi = 0.9189385332046727
    series = coeff0 + coeff1 / (x + 1.0) + coeff2 / (x + 2.0) + coeff3 / (x + 3.0) + coeff4 / (x + 4.0) + coeff5 / (x + 5.0) + coeff6 / (x + 6.0)
    t = x + 5.5
    pos = (x + 0.5) * tl.log(t) - t + log_sqrt_2pi + tl.log(series) - tl.log(x)
    result = pos
    x_masked = tl.where(mask, x, 1.0)
    need = (tl.min(x_masked) <= 0.0) | (tl.max(x_masked) == float("inf"))
    if need:
        pi = 3.141592653589793
        log_pi = 1.1447298858494002
        z = 1.0 - x
        series_z = coeff0 + coeff1 / (z + 1.0) + coeff2 / (z + 2.0) + coeff3 / (z + 3.0) + coeff4 / (z + 4.0) + coeff5 / (z + 5.0) + coeff6 / (z + 6.0)
        tz = z + 5.5
        lg_1mx = (z + 0.5) * tl.log(tz) - tz + log_sqrt_2pi + tl.log(series_z) - tl.log(z)
        refl = log_pi - tl.log(tl.abs(tl.sin(pi * x))) - lg_1mx
        is_pole = (x <= 0.0) & (x == tl.floor(x))
        neg_result = tl.where(is_pole, float("inf"), refl)
        is_inf = x == float("inf")
        result = tl.where(is_inf, float("inf"), tl.where(x > 0.0, pos, neg_result))
    tl.store(output_ptr + offsets, result, mask=mask)


def lgamma_1d(input, output):
    # MetaX: native lgamma is bandwidth-bound, NineToothed libdevice.lgamma compute-bound (~0.58x).
    if not is_metax_device(input):
        return False
    if input.device.type != "cuda":
        return False
    if input.ndim != 1 or output.ndim != 1:
        return False
    if not input.is_contiguous() or not output.is_contiguous():
        return False
    if input.dtype not in (torch.float16, torch.bfloat16, torch.float32):
        return False
    block_size = 4096
    num_warps = 4
    grid = (triton.cdiv(input.numel(), block_size),)
    _lgamma_kernel[grid](
        input,
        output,
        input.numel(),
        block_size=block_size,
        num_warps=num_warps,
    )
    return True


@triton.jit
def _lgamma_libdevice_kernel(input_ptr, output_ptr, n, block_size: tl.constexpr):
    offsets = tl.program_id(0) * block_size + tl.arange(0, block_size)
    mask = offsets < n
    x = tl.load(input_ptr + offsets, mask=mask, other=1.0).to(tl.float32)
    result = _tl_libdevice.lgamma(x)
    tl.store(output_ptr + offsets, result.to(output_ptr.dtype.element_ty), mask=mask)


def lgamma_iluvatar_1d(input, output):
    # CoreX: the NineToothed libdevice.lgamma path is codegen-bound (~0.93x); a
    # bare libdevice.lgamma kernel at block 2048/4 warps reaches ~1.02x (f32) and
    # ~0.98x (f16). f64 stays on the NineToothed path (already >2x on CoreX).
    if _tl_libdevice is None:
        return False
    if not is_iluvatar_device(input):
        return False
    if input.device.type != "cuda":
        return False
    if input.ndim != 1 or output.ndim != 1:
        return False
    if not input.is_contiguous() or not output.is_contiguous():
        return False
    if input.dtype not in (torch.float16, torch.bfloat16, torch.float32):
        return False
    block_size = 2048
    num_warps = 4
    grid = (triton.cdiv(input.numel(), block_size),)
    _lgamma_libdevice_kernel[grid](
        input,
        output,
        input.numel(),
        block_size=block_size,
        num_warps=num_warps,
    )
    return True
