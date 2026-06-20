import functools
import math

import torch
import triton
import triton.language as tl

from ntops.torch.utils import _is_dtype, _is_dtype_any


@functools.cache
def _device_name_for_index(index):
    try:
        return torch.cuda.get_device_name(index)
    except Exception:
        return ""


def _device_name(tensor):
    # Duck-type: accept torch, infinicore, and any object with .device
    dev = getattr(tensor, "device", None)
    if dev is None or dev.type != "cuda":
        return ""
    if not hasattr(torch, "cuda"):
        return ""
    index = dev.index
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
    if input.dtype in (torch.float16, torch.bfloat16):
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


def lcm_1d(input, other, output):
    if input.device.type != "cuda":
        return False
    if input.ndim != 1 or other.ndim != 1 or output.ndim != 1:
        return False
    if not input.is_contiguous() or not other.is_contiguous() or not output.is_contiguous():
        return False

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
def _frac_kernel(in_ptr, out_ptr, n_elements, USE_INT_TRUNC: tl.constexpr, BLOCK_SIZE: tl.constexpr):
    offsets = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(in_ptr + offsets, mask=mask, other=0.0)
    if USE_INT_TRUNC:
        value = x.to(tl.float32)
        # int32 truncation avoids the slow floor/ceil libdevice path; guard
        # |v|>=2^23 where f32 has no fractional part (and int32 would overflow).
        small = tl.abs(value) < 8388608.0
        in_range = tl.where(small, value, 0.0)
        truncated = tl.where(small, in_range.to(tl.int32).to(tl.float32), value)
        result = (value - truncated).to(x.dtype)
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

    if input.dtype == torch.float64:
        use_int_trunc = False
        block_size = 1024
    else:
        use_int_trunc = True
        block_size = 2048
    num_warps = 4

    grid = (triton.cdiv(input.numel(), block_size),)
    _frac_kernel[grid](
        input,
        output,
        input.numel(),
        USE_INT_TRUNC=use_int_trunc,
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
    if is_iluvatar_device(output) and output.dtype == torch.float64:
        # Iluvatar CoreX 10.2 cannot compile f64 global atomic_add; routed to
        # native torch by the wrapper. Non-f64 dtypes compile and run correctly.
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
    x = tl.load(input_ptr + offsets, mask=mask, other=6.0).to(tl.float32)
    # Stirling log-gamma: recurrence-reduce to z>=6 (product accumulated so one
    # log() is paid), Euler reflection for x<=0. Avoids slow libdevice.lgamma.
    log_sqrt_2pi = 0.9189385332046727
    log_pi = 1.1447298858494002
    pi = 3.141592653589793
    neg = x <= 0.0
    w = tl.where(neg, 1.0 - x, x)
    prod = 1.0
    z = w
    for _i in tl.static_range(6):
        do = z < 6.0
        prod = tl.where(do, prod * z, prod)
        z = tl.where(do, z + 1.0, z)
    inv = 1.0 / z
    inv2 = inv * inv
    poly = inv * (1.0 / 12.0 + inv2 * (-1.0 / 360.0 + inv2 * (1.0 / 1260.0)))
    lg_z = (z - 0.5) * tl.log(z) - z + log_sqrt_2pi + poly
    lg_w = lg_z - tl.log(prod)
    refl = log_pi - tl.log(tl.abs(tl.sin(pi * x))) - lg_w
    result = tl.where(neg, refl, lg_w)
    is_nonpos_int = neg & (x == tl.floor(x))
    result = tl.where(is_nonpos_int, float("inf"), result)
    result = tl.where(x == float("inf"), float("inf"), result)
    tl.store(output_ptr + offsets, result, mask=mask)


def lgamma_1d(input, output):
    if input.device.type != "cuda":
        return False
    if not is_metax_device(input):
        return False
    if input.ndim != 1 or output.ndim != 1:
        return False
    if not input.is_contiguous() or not output.is_contiguous():
        return False
    if input.dtype not in (torch.float16, torch.bfloat16, torch.float32):
        return False
    block_size = 1024
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


@triton.jit(do_not_specialize=["seed"])
def _feature_alpha_dropout_channel_kernel(
    input,
    output,
    channels: tl.constexpr,
    spatial: tl.constexpr,
    p,
    seed,
    a,
    b,
    dropped,
    block_size: tl.constexpr,
):
    channel = tl.program_id(0)
    block = tl.program_id(1)
    offsets = block * block_size + tl.arange(0, block_size)
    mask = (channel < channels) & (offsets < spatial)
    base = channel * spatial + offsets
    value = tl.load(input + base, mask=mask, other=0.0).to(tl.float32)
    keep = tl.rand(seed, channel) > p
    result = tl.where(keep, a * value + b, dropped)
    tl.store(output + base, result, mask=mask)


def feature_alpha_dropout_fast_path(input, output, p, seed, spatial, a, b, dropped):
    if input.device.type != "cuda":
        return False
    if not input.is_contiguous() or not output.is_contiguous():
        return False
    if tuple(input.shape) != tuple(output.shape) or input.dtype != output.dtype:
        return False
    if not _is_dtype_any(input.dtype, ["float16", "float32", "float64"]):
        return False
    if spatial <= 0 or input.numel() % spatial != 0:
        return False

    channels = input.numel() // spatial
    block_size = 2048
    num_warps = 4
    if _is_dtype(input.dtype, "float16"):
        block_size = 1024
    if _is_dtype(input.dtype, "float64"):
        if is_iluvatar_device(input):
            # Iluvatar f64 regresses badly at num_warps=8 (0.75x); bs=512/nw=4 peaks ~4.4x.
            block_size = 512
            num_warps = 4
        else:
            block_size = 1024
            num_warps = 8
    grid = (channels, triton.cdiv(spatial, block_size))
    _feature_alpha_dropout_channel_kernel[grid](
        input,
        output,
        channels,
        spatial,
        float(p),
        int(seed),
        float(a),
        float(b),
        float(dropped),
        block_size=block_size,
        num_warps=num_warps,
    )
    return True


@triton.jit
def _pixel_unshuffle_input_kernel(
    input,
    output,
    total: tl.constexpr,
    channels: tl.constexpr,
    in_h: tl.constexpr,
    in_w: tl.constexpr,
    out_h: tl.constexpr,
    out_w: tl.constexpr,
    factor: tl.constexpr,
    block_size: tl.constexpr,
):
    offsets = tl.program_id(0) * block_size + tl.arange(0, block_size)
    mask = offsets < total
    in_x = offsets % in_w
    rem = offsets // in_w
    in_y = rem % in_h
    rem = rem // in_h
    channel = rem % channels
    batch = rem // channels

    out_y = in_y // factor
    sub_y = in_y - out_y * factor
    out_x = in_x // factor
    sub_x = in_x - out_x * factor
    out_channel = (channel * factor + sub_y) * factor + sub_x
    out_channels = channels * factor * factor
    out_offsets = ((batch * out_channels + out_channel) * out_h + out_y) * out_w + out_x

    value = tl.load(input + offsets, mask=mask)
    tl.store(output + out_offsets, value, mask=mask)


@triton.jit
def _pixel_unshuffle_output_strided_kernel(
    input,
    output,
    total: tl.constexpr,
    channels: tl.constexpr,
    out_h: tl.constexpr,
    out_w: tl.constexpr,
    factor: tl.constexpr,
    stride_0: tl.constexpr,
    stride_1: tl.constexpr,
    stride_2: tl.constexpr,
    stride_3: tl.constexpr,
    block_size: tl.constexpr,
):
    offsets = tl.program_id(0) * block_size + tl.arange(0, block_size)
    mask = offsets < total
    out_x = offsets % out_w
    rem = offsets // out_w
    out_y = rem % out_h
    rem = rem // out_h
    out_channel = rem % (channels * factor * factor)
    batch = rem // (channels * factor * factor)

    channel = out_channel // (factor * factor)
    sub = out_channel - channel * factor * factor
    sub_y = sub // factor
    sub_x = sub - sub_y * factor
    in_y = out_y * factor + sub_y
    in_x = out_x * factor + sub_x
    input_offsets = batch * stride_0 + channel * stride_1 + in_y * stride_2 + in_x * stride_3

    value = tl.load(input + input_offsets, mask=mask)
    tl.store(output + offsets, value, mask=mask)


def pixel_unshuffle_fast_path(input, output, factor):
    if input.device.type != "cuda":
        return False
    if not is_corex_or_metax_device(input):
        return False
    if input.ndim != 4 or not output.is_contiguous():
        return False
    if not _is_dtype_any(input.dtype, ["float16", "bfloat16", "float32", "float64", "int16", "int32", "int64"]):
        return False
    batch, channels, in_h, in_w = input.shape
    if in_h % factor != 0 or in_w % factor != 0:
        return False
    out_h = in_h // factor
    out_w = in_w // factor
    expected = (batch, channels * factor * factor, out_h, out_w)
    if tuple(output.shape) != expected or input.dtype != output.dtype:
        return False

    contiguous = input.is_contiguous()

    if is_iluvatar_device(input):
        # Iluvatar MR-V100 microbench optima (differ from MetaX block sizes).
        if contiguous:
            if _is_dtype_any(input.dtype, ["float16", "bfloat16"]):
                block_size, num_warps = 1024, 4
            elif _is_dtype(input.dtype, "float64"):
                block_size, num_warps = 256, 4
            else:
                block_size, num_warps = 512, 8
        else:
            block_size, num_warps = 512, 8
    else:
        block_size = 1024
        num_warps = 4
        if _is_dtype_any(input.dtype, ["float16", "bfloat16"]):
            block_size = 2048
        elif _is_dtype(input.dtype, "float64"):
            block_size = 512
            num_warps = 8

    if contiguous:
        _pixel_unshuffle_input_kernel[(triton.cdiv(input.numel(), block_size),)](
            input,
            output,
            input.numel(),
            channels,
            in_h,
            in_w,
            out_h,
            out_w,
            factor,
            block_size=block_size,
            num_warps=num_warps,
        )
        return True

    strides = input.stride()
    _pixel_unshuffle_output_strided_kernel[(triton.cdiv(output.numel(), block_size),)](
        input,
        output,
        output.numel(),
        channels,
        out_h,
        out_w,
        factor,
        strides[0],
        strides[1],
        strides[2],
        strides[3],
        block_size=block_size,
        num_warps=num_warps,
    )
    return True


@triton.jit
def _mse_none_kernel(
    input,
    target,
    output,
    total: tl.constexpr,
    cast_fp32: tl.constexpr,
    block_size: tl.constexpr,
):
    offsets = tl.program_id(0) * block_size + tl.arange(0, block_size)
    mask = offsets < total
    lhs = tl.load(input + offsets, mask=mask, other=0.0)
    rhs = tl.load(target + offsets, mask=mask, other=0.0)
    if cast_fp32:
        lhs = lhs.to(tl.float32)
        rhs = rhs.to(tl.float32)
    diff = lhs - rhs
    tl.store(output + offsets, diff * diff, mask=mask)


def mse_none(input, target, output):
    if input.device.type != "cuda":
        return False
    if not input.is_contiguous() or not target.is_contiguous() or not output.is_contiguous():
        return False
    if tuple(input.shape) != tuple(target.shape) or tuple(input.shape) != tuple(output.shape):
        return False
    if input.dtype != target.dtype or input.dtype != output.dtype:
        return False
    if input.dtype not in (torch.float16, torch.float32):
        return False

    block_size = 2048
    num_warps = 4
    if _is_dtype(input.dtype, "float64"):
        block_size = 1024
        num_warps = 8

    _mse_none_kernel[(triton.cdiv(input.numel(), block_size),)](
        input,
        target,
        output,
        input.numel(),
        cast_fp32=not _is_dtype(input.dtype, "float16"),
        block_size=block_size,
        num_warps=num_warps,
    )
    return True


@triton.jit
def _mse_row_sum_kernel(
    input,
    target,
    partials,
    rows: tl.constexpr,
    cols: tl.constexpr,
    block_size: tl.constexpr,
):
    row = tl.program_id(0)
    offsets = tl.arange(0, block_size)
    mask = offsets < cols
    base = row * cols + offsets
    lhs = tl.load(input + base, mask=mask, other=0.0).to(tl.float32)
    rhs = tl.load(target + base, mask=mask, other=0.0).to(tl.float32)
    diff = lhs - rhs
    total = tl.sum(tl.where(mask, diff * diff, 0.0), axis=0)
    tl.store(partials + row, total, mask=row < rows)


def mse_row_sums(input, target, cols):
    if input.device.type != "cuda":
        return None
    if not input.is_contiguous() or not target.is_contiguous():
        return None
    if input.ndim != 1 or target.ndim != 1 or input.numel() != target.numel():
        return None
    if input.dtype != target.dtype:
        return None
    if not _is_dtype_any(input.dtype, ["float16", "bfloat16", "float32", "float64"]):
        return None
    if cols <= 0 or input.numel() % cols != 0:
        return None

    rows = input.numel() // cols
    partials = torch.empty((rows,), dtype=torch.float32, device="cuda")
    block_size = 1
    while block_size < cols:
        block_size *= 2
    if block_size > 8192:
        return None
    num_warps = 4
    if block_size >= 2048:
        num_warps = 8
    _mse_row_sum_kernel[(rows,)](
        input,
        target,
        partials,
        rows,
        cols,
        block_size=block_size,
        num_warps=num_warps,
    )
    return partials


@triton.jit
def _flip_contiguous_kernel(
    input,
    output,
    total: tl.constexpr,
    size_0: tl.constexpr,
    size_1: tl.constexpr,
    size_2: tl.constexpr,
    size_3: tl.constexpr,
    size_4: tl.constexpr,
    flip_0: tl.constexpr,
    flip_1: tl.constexpr,
    flip_2: tl.constexpr,
    flip_3: tl.constexpr,
    flip_4: tl.constexpr,
    block_size: tl.constexpr,
):
    offsets = tl.program_id(0) * block_size + tl.arange(0, block_size)
    mask = offsets < total
    rem = offsets
    idx_4 = rem % size_4
    rem = rem // size_4
    idx_3 = rem % size_3
    rem = rem // size_3
    idx_2 = rem % size_2
    rem = rem // size_2
    idx_1 = rem % size_1
    idx_0 = rem // size_1

    src_0 = idx_0
    src_1 = idx_1
    src_2 = idx_2
    src_3 = idx_3
    src_4 = idx_4
    if flip_0:
        src_0 = size_0 - 1 - idx_0
    if flip_1:
        src_1 = size_1 - 1 - idx_1
    if flip_2:
        src_2 = size_2 - 1 - idx_2
    if flip_3:
        src_3 = size_3 - 1 - idx_3
    if flip_4:
        src_4 = size_4 - 1 - idx_4
    src_offsets = (
        (((src_0 * size_1 + src_1) * size_2 + src_2) * size_3 + src_3) * size_4
        + src_4
    )
    value = tl.load(input + src_offsets, mask=mask)
    tl.store(output + offsets, value, mask=mask)


def flip_contiguous(input, output, dims):
    if input.device.type != "cuda":
        return False
    if tuple(input.shape) != tuple(output.shape) or input.dtype != output.dtype:
        return False
    if not input.is_contiguous() or not output.is_contiguous():
        return False
    if input.ndim == 0 or input.ndim > 5:
        return False
    if not _is_dtype_any(input.dtype, ["float16", "bfloat16", "float32", "float64", "int16", "int32", "int64"]):
        return False

    shape = tuple(input.shape) + (1,) * (5 - input.ndim)
    flip_dims = tuple(dim in dims for dim in range(input.ndim)) + (False,) * (
        5 - input.ndim
    )
    block_size = 1024
    num_warps = 4
    if _is_dtype_any(input.dtype, ["float16", "bfloat16"]):
        block_size = 2048
    elif _is_dtype(input.dtype, "float64"):
        if is_iluvatar_device(input):
            # Iluvatar f64 flip: smaller block / fewer warps stays >1.0x; the
            # default 512/8 drifts to the 0.90 boundary under sweep jitter.
            block_size = 256
            num_warps = 4
        else:
            block_size = 512
            num_warps = 8

    grid = (triton.cdiv(input.numel(), block_size),)
    _flip_contiguous_kernel[grid](
        input,
        output,
        input.numel(),
        shape[0],
        shape[1],
        shape[2],
        shape[3],
        shape[4],
        flip_dims[0],
        flip_dims[1],
        flip_dims[2],
        flip_dims[3],
        flip_dims[4],
        block_size=block_size,
        num_warps=num_warps,
    )
    return True


@triton.jit
def _flip_2d_strided_kernel(
    input,
    output,
    rows: tl.constexpr,
    cols: tl.constexpr,
    stride_0: tl.constexpr,
    stride_1: tl.constexpr,
    flip_0: tl.constexpr,
    flip_1: tl.constexpr,
    block_m: tl.constexpr,
    block_n: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    offs_m = pid_m * block_m + tl.arange(0, block_m)
    offs_n = pid_n * block_n + tl.arange(0, block_n)
    src_m = offs_m
    src_n = offs_n
    if flip_0:
        src_m = rows - 1 - offs_m
    if flip_1:
        src_n = cols - 1 - offs_n
    mask = (offs_m[:, None] < rows) & (offs_n[None, :] < cols)
    value = tl.load(
        input + src_m[:, None] * stride_0 + src_n[None, :] * stride_1,
        mask=mask,
    )
    tl.store(output + offs_m[:, None] * cols + offs_n[None, :], value, mask=mask)


def flip_2d_strided(input, output, dims):
    if input.device.type != "cuda":
        return False
    if input.ndim != 2 or output.ndim != 2:
        return False
    if input.is_contiguous() or not output.is_contiguous():
        return False
    if tuple(input.shape) != tuple(output.shape) or input.dtype != output.dtype:
        return False
    if not _is_dtype_any(input.dtype, ["float16", "bfloat16", "float32", "float64", "int16", "int32", "int64"]):
        return False
    rows, cols = input.shape
    strides = input.stride()
    if strides[0] != 1:
        return False

    block_m = 64
    block_n = 32
    num_warps = 8
    if _is_dtype(input.dtype, "float16"):
        block_m = 64
        block_n = 32
        num_warps = 8
    elif _is_dtype(input.dtype, "float64"):
        block_m = 16
        block_n = 32
        num_warps = 4

    _flip_2d_strided_kernel[(triton.cdiv(rows, block_m), triton.cdiv(cols, block_n))](
        input,
        output,
        rows,
        cols,
        strides[0],
        strides[1],
        0 in dims,
        1 in dims,
        block_m=block_m,
        block_n=block_n,
        num_warps=num_warps,
    )
    return True


@triton.jit
def _sum_strided_kernel(input, output, count, stride, BLOCK: tl.constexpr):
    accumulator = tl.zeros((BLOCK,), dtype=input.dtype.element_ty)
    offsets = tl.arange(0, BLOCK)
    for start in tl.range(0, count, BLOCK):
        index = start + offsets
        mask = index < count
        values = tl.load(input + index * stride, mask=mask, other=0)
        accumulator += values
    total = tl.sum(accumulator, axis=0)
    tl.store(output, total)


def sum_strided(partials, count, stride, dtype):
    if partials.device.type != "cuda":
        return None
    real_dtype = getattr(torch, str(dtype).split(".")[-1], torch.float32)
    if count <= 0:
        return torch.zeros((), dtype=real_dtype, device="cuda")

    output = torch.empty((), dtype=real_dtype, device="cuda")
    block_size = 1
    while block_size < count and block_size < 4096:
        block_size *= 2
    _sum_strided_kernel[(1,)](
        partials,
        output,
        count,
        stride,
        BLOCK=block_size,
        num_warps=8,
    )
    return output


@triton.jit
def _multilabel_margin_loss_kernel(
    input_ptr, target_ptr, out_ptr, batch,
    C: tl.constexpr, C_PAD: tl.constexpr, BLOCK_ROWS: tl.constexpr,
):
    pid = tl.program_id(0)
    rows = pid * BLOCK_ROWS + tl.arange(0, BLOCK_ROWS)
    rmask = rows < batch
    cls = tl.arange(0, C_PAD)
    cvalid = cls < C
    m = rmask[:, None] & cvalid[None, :]
    x = tl.load(input_ptr + rows[:, None] * C + cls[None, :], mask=m, other=0.0).to(tl.float32)
    tgt = tl.load(target_ptr + rows[:, None] * C + cls[None, :], mask=m, other=-1)
    slot = cls
    end = C + 1
    first_neg = tl.min(tl.where(tgt < 0, slot[None, :], end), axis=1)
    active = slot[None, :] < first_neg[:, None]
    match = (tgt[:, :, None] == cls[None, None, :]) & active[:, :, None]
    target_count = tl.sum(tl.where(match, 1.0, 0.0), axis=1)
    is_non_target = cvalid[None, :] & (target_count == 0)
    margin = 1.0 - x[:, :, None] + x[:, None, :]
    hinge = tl.where(margin > 0.0, margin, 0.0)
    weight = target_count[:, :, None] * tl.where(is_non_target[:, None, :], 1.0, 0.0)
    total = tl.sum(tl.sum(weight * hinge, axis=2), axis=1)
    out = total / C
    tl.store(out_ptr + rows, out, mask=rmask)


def multilabel_margin_loss_per_row(input, target, losses):
    if input.device.type != "cuda":
        return False
    if not is_iluvatar_device(input):
        return False
    if input.ndim != 2 or not input.is_contiguous() or not target.is_contiguous():
        return False
    batch, num_classes = input.shape
    c_pad = 1
    while c_pad < num_classes:
        c_pad *= 2
    block_rows = 8
    num_warps = 1
    grid = (triton.cdiv(batch, block_rows),)
    _multilabel_margin_loss_kernel[grid](
        input, target, losses, batch,
        C=num_classes, C_PAD=c_pad, BLOCK_ROWS=block_rows,
        num_warps=num_warps,
    )
    return True


@triton.jit
def _slogdet_kernel(a_ptr, sign_ptr, logdet_ptr, batch, N: tl.constexpr, BN: tl.constexpr):
    # One program per matrix: load the whole NxN block and run Gaussian elimination
    # with partial pivoting. det = product of pivots, with a sign flip per row swap;
    # slogdet returns (sign, log|det|). LU is sequential so this can't use the tile
    # DSL, but cuSOLVER's batched path is launch-heavy for small N, so a per-matrix
    # program wins (1.8-4.2x on small batched matrices).
    pid = tl.program_id(0)
    if pid >= batch:
        return
    rows = tl.arange(0, BN)
    cols = tl.arange(0, BN)
    mask = (rows[:, None] < N) & (cols[None, :] < N)
    base = pid * N * N
    a = tl.load(a_ptr + base + rows[:, None] * N + cols[None, :], mask=mask, other=0.0).to(tl.float32)
    sign = 1.0
    logdet = 0.0
    for k in range(N):
        colk = tl.where(
            (rows >= k) & (rows < N),
            tl.abs(tl.sum(tl.where(cols[None, :] == k, a, 0.0), axis=1)),
            -1.0,
        )
        piv = tl.argmax(colk, axis=0)
        rowk = tl.sum(tl.where(rows[:, None] == k, a, 0.0), axis=0)
        rowp = tl.sum(tl.where(rows[:, None] == piv, a, 0.0), axis=0)
        do_swap = piv != k
        a = tl.where(rows[:, None] == k, rowp[None, :], a)
        a = tl.where(rows[:, None] == piv, rowk[None, :], a)
        sign = tl.where(do_swap, -sign, sign)
        pivval = tl.sum(tl.where((rows[:, None] == k) & (cols[None, :] == k), a, 0.0))
        sign = sign * tl.where(pivval < 0, -1.0, 1.0)
        logdet = logdet + tl.log(tl.abs(pivval))
        colk_vals = tl.sum(tl.where(cols[None, :] == k, a, 0.0), axis=1)
        factor = tl.where(rows > k, colk_vals / pivval, 0.0)
        rowk2 = tl.sum(tl.where(rows[:, None] == k, a, 0.0), axis=0)
        a = a - factor[:, None] * rowk2[None, :] * tl.where(rows[:, None] > k, 1.0, 0.0)
    tl.store(sign_ptr + pid, sign)
    tl.store(logdet_ptr + pid, logdet)


def slogdet_batched(input, sign, logdet):
    if input.device.type != "cuda":
        return False
    # f64 LU in-kernel is unreliable on the vendor GPUs (limited double support); only
    # f32 takes the kernel. Iluvatar lacks the f64 path entirely (handled by wrapper).
    if input.dtype != torch.float32:
        return False
    if input.ndim != 3 or not input.is_contiguous():
        return False
    batch, n, n2 = input.shape
    if n != n2 or n == 0:
        return False
    bn = 1
    while bn < n:
        bn *= 2
    grid = (batch,)
    _slogdet_kernel[grid](input, sign, logdet, batch, N=n, BN=bn)
    return True


@triton.jit
def _pixel_unshuffle_kernel(
    in_ptr,
    out_ptr,
    N,
    C,
    H,
    W,
    stride_n,
    stride_c,
    stride_h,
    stride_w,
    R: tl.constexpr,
    RPAD: tl.constexpr,
    BLOCK_H: tl.constexpr,
    BLOCK_W: tl.constexpr,
):
    # One program owns a (BLOCK_H, BLOCK_W) tile of the output (h, w) plane for a
    # fixed (n, c, i) where i in [0, R) is the vertical sub-pixel offset. It reads
    # input row (h*R + i) cols (w*R + j) for j in [0, R): along the innermost axis
    # the read advances by stride_w (== 1 for contiguous input), so loads coalesce;
    # the R sub-columns j are gathered with a small RPAD-wide inner axis. Writes go
    # to contiguous output channels (c*R*R + i*R + j) at plane offset h*W + w, so
    # stores coalesce along w. Index decode is per-program, not per-element.
    pid_row = tl.program_id(0)
    pid_hw = tl.program_id(1)
    num_hb = (H + BLOCK_H - 1) // BLOCK_H
    hb = pid_hw % num_hb
    wb = pid_hw // num_hb

    i = pid_row % R
    t = pid_row // R
    c = t % C
    n = t // C

    h = hb * BLOCK_H + tl.arange(0, BLOCK_H)
    w = wb * BLOCK_W + tl.arange(0, BLOCK_W)
    j = tl.arange(0, RPAD)

    in_row = h * R + i
    base_in = n * stride_n + c * stride_c
    in_off = (
        base_in
        + in_row[:, None, None] * stride_h
        + (w[None, :, None] * R + j[None, None, :]) * stride_w
    )
    mask = (h[:, None, None] < H) & (w[None, :, None] < W) & (j[None, None, :] < R)
    val = tl.load(in_ptr + in_off, mask=mask, other=0)

    channels = C * R * R
    plane = H * W
    out_ch = n * channels + c * R * R + i * R + j
    out_off = out_ch[None, None, :] * plane + h[:, None, None] * W + w[None, :, None]
    tl.store(out_ptr + out_off, val, mask=mask)


@triton.jit
def _pixel_unshuffle_flat_kernel(
    in_ptr,
    out_ptr,
    n_elements,
    C,
    H,
    W,
    channels,
    stride_n,
    stride_c,
    stride_h,
    stride_w,
    R: tl.constexpr,
    BLOCK: tl.constexpr,
):
    # Flat 1D grid over the contiguous output: each program writes a BLOCK run of
    # output elements (perfectly coalesced) and decodes each output index back to
    # the input gather offset. Output channel ch = c*R*R + i*R + j, so j,i,c peel
    # off ch; the input element is row (h*R+i), col (w*R+j). For small tensors this
    # one-launch flat form beats the tiled kernel's larger grid.
    pid = tl.program_id(0)
    o = pid * BLOCK + tl.arange(0, BLOCK)
    mask = o < n_elements
    w = o % W
    t = o // W
    h = t % H
    t = t // H
    ch = t % channels
    n = t // channels
    j = ch % R
    t2 = ch // R
    i = t2 % R
    c = t2 // R
    in_off = (
        n.to(tl.int64) * stride_n
        + c.to(tl.int64) * stride_c
        + (h * R + i).to(tl.int64) * stride_h
        + (w * R + j).to(tl.int64) * stride_w
    )
    val = tl.load(in_ptr + in_off, mask=mask)
    tl.store(out_ptr + o, val, mask=mask)


_PIXEL_UNSHUFFLE_DTYPES = (
    torch.float16,
    torch.bfloat16,
    torch.float32,
    torch.float64,
    torch.int16,
    torch.int32,
    torch.int64,
)


def pixel_unshuffle_fast_path(input, output, downscale_factor):
    # Reads strided input directly (handles non-contiguous without a pre-copy) and
    # writes contiguous output; both sides coalesce, unlike the 6D permute gather.
    # Two kernels by measured win on H100: contiguous input uses the flat 1D-grid
    # kernel (one launch, best for small/medium tensors); non-contiguous input uses
    # the (BLOCK_H, BLOCK_W) tiled kernel, whose 2D tiling coalesces the strided
    # transpose-style read far better (e.g. 2.7x vs 1.4x on f16 transposed).
    if input.device.type != "cuda":
        return False
    if input.ndim != 4:
        return False
    if input.dtype not in _PIXEL_UNSHUFFLE_DTYPES:
        return False
    if not output.is_contiguous():
        return False

    r = downscale_factor
    N, C, Hr, Wr = input.shape
    if r <= 0 or Hr % r != 0 or Wr % r != 0:
        return False
    H, W = Hr // r, Wr // r
    if H == 0 or W == 0:
        return False

    channels = C * r * r
    stride_n, stride_c, stride_h, stride_w = input.stride()

    # Flat 1D kernel only for contiguous, low-R input: its per-element index decode
    # is cheap there and the one-launch form helps small/medium tensors. High R
    # (e.g. r=8) makes the decode heavy, so those keep the tiled kernel which
    # computes the sub-pixel row offset once per program.
    if input.is_contiguous() and r <= 4:
        block_size = 2048
        num_warps = 8 if input.element_size() >= 8 else 4
        grid = (triton.cdiv(output.numel(), block_size),)
        _pixel_unshuffle_flat_kernel[grid](
            input,
            output,
            output.numel(),
            C,
            H,
            W,
            channels,
            stride_n,
            stride_c,
            stride_h,
            stride_w,
            R=r,
            BLOCK=block_size,
            num_warps=num_warps,
        )
        return True

    rpad = 1
    while rpad < r:
        rpad *= 2

    # Block heuristic from a measured sweep on H100: wide W favours a tall W-block
    # with single-row H tiles; narrower W favours fatter H blocks to keep occupancy.
    if W >= 512:
        block_h, block_w, num_warps = 1, 256, 4
    elif W >= 256:
        block_h, block_w, num_warps = 8, 128, 8
    else:
        block_h, block_w, num_warps = 16, 128, 4

    grid = (
        N * C * r,
        triton.cdiv(H, block_h) * triton.cdiv(W, block_w),
    )
    _pixel_unshuffle_kernel[grid](
        input,
        output,
        N,
        C,
        H,
        W,
        stride_n,
        stride_c,
        stride_h,
        stride_w,
        R=r,
        RPAD=rpad,
        BLOCK_H=block_h,
        BLOCK_W=block_w,
        num_warps=num_warps,
    )
    return True


@triton.jit
def _feature_alpha_dropout_kernel(
    in_ptr,
    out_ptr,
    spatial,
    seed,
    p,
    a,
    b,
    dropped,
    BLOCK: tl.constexpr,
):
    # feature_alpha_dropout masks whole channels. The grid is (N*C, spatial_blocks):
    # one philox draw keyed on the channel id (program_id(0)) decides keep/drop for
    # the entire channel, so there is no per-element division and a single fused
    # launch replaces the rand + compare + cast + apply chain. Kept channels get the
    # affine a*x+b; dropped channels saturate to the constant `dropped`.
    pid_c = tl.program_id(0)
    pid_s = tl.program_id(1)
    offs = pid_s * BLOCK + tl.arange(0, BLOCK)
    mask = offs < spatial
    base = pid_c.to(tl.int64) * spatial
    x = tl.load(in_ptr + base + offs, mask=mask, other=0.0)
    keep = tl.rand(seed, pid_c) > p
    affine = a * x.to(tl.float32) + b
    value = tl.where(keep, affine, dropped)
    tl.store(out_ptr + base + offs, value.to(x.dtype), mask=mask)


def feature_alpha_dropout_fast_path(input, output, p, seed, a, b, dropped):
    if input.device.type != "cuda":
        return False
    if input.ndim < 2:
        return False
    if not input.is_contiguous() or not output.is_contiguous():
        return False
    if input.dtype not in (torch.float16, torch.bfloat16, torch.float32, torch.float64):
        return False

    n_channels = input.shape[0] * input.shape[1]
    if n_channels == 0:
        return True
    spatial = 1
    for size in input.shape[2:]:
        spatial *= size
    if spatial == 0:
        return True

    block_size = 1024
    num_warps = 4
    grid = (n_channels, triton.cdiv(spatial, block_size))
    _feature_alpha_dropout_kernel[grid](
        input,
        output,
        spatial,
        seed,
        float(p),
        float(a),
        float(b),
        float(dropped),
        BLOCK=block_size,
        num_warps=num_warps,
    )
    return True


@triton.jit
def _flip_nd_kernel(
    in_ptr,
    out_ptr,
    n_elements,
    sz0,
    sz1,
    sz2,
    sz3,
    sz4,
    st0,
    st1,
    st2,
    st3,
    st4,
    fl0: tl.constexpr,
    fl1: tl.constexpr,
    fl2: tl.constexpr,
    fl3: tl.constexpr,
    fl4: tl.constexpr,
    NDIM: tl.constexpr,
    BLOCK: tl.constexpr,
):
    # Output is contiguous, so the flat output offset is the row-major mixed-radix
    # encoding of the coords. Decode coords from the trailing axis inward, mirror
    # (size-1-coord) the flagged axes, and gather through the input strides. This
    # reads non-contiguous input directly, replacing the contiguous() pre-copy.
    pid = tl.program_id(0)
    out_off = pid * BLOCK + tl.arange(0, BLOCK)
    mask = out_off < n_elements
    rem = out_off
    in_off = tl.zeros((BLOCK,), dtype=tl.int64)
    if NDIM >= 5:
        c4 = rem % sz4
        rem = rem // sz4
        c4 = tl.where(fl4, sz4 - 1 - c4, c4)
        in_off += c4.to(tl.int64) * st4
    if NDIM >= 4:
        c3 = rem % sz3
        rem = rem // sz3
        c3 = tl.where(fl3, sz3 - 1 - c3, c3)
        in_off += c3.to(tl.int64) * st3
    if NDIM >= 3:
        c2 = rem % sz2
        rem = rem // sz2
        c2 = tl.where(fl2, sz2 - 1 - c2, c2)
        in_off += c2.to(tl.int64) * st2
    if NDIM >= 2:
        c1 = rem % sz1
        rem = rem // sz1
        c1 = tl.where(fl1, sz1 - 1 - c1, c1)
        in_off += c1.to(tl.int64) * st1
    c0 = tl.where(fl0, sz0 - 1 - rem, rem)
    in_off += c0.to(tl.int64) * st0
    value = tl.load(in_ptr + in_off, mask=mask)
    tl.store(out_ptr + out_off, value, mask=mask)


@triton.jit
def _flip_2d_kernel(
    in_ptr,
    out_ptr,
    rows,
    cols,
    stride_r,
    stride_c,
    FLIP_R: tl.constexpr,
    FLIP_C: tl.constexpr,
    BLOCK_R: tl.constexpr,
    BLOCK_C: tl.constexpr,
):
    # 2D tile so both the strided read and the contiguous write coalesce; this is
    # the transpose-bound case (flip along the non-contiguous axis) where the 1D
    # kernel degrades to a scattered gather.
    pid_r = tl.program_id(0)
    pid_c = tl.program_id(1)
    rs = pid_r * BLOCK_R + tl.arange(0, BLOCK_R)
    cs = pid_c * BLOCK_C + tl.arange(0, BLOCK_C)
    mask = (rs[:, None] < rows) & (cs[None, :] < cols)
    ir = tl.where(FLIP_R, rows - 1 - rs, rs)
    ic = tl.where(FLIP_C, cols - 1 - cs, cs)
    in_off = ir[:, None].to(tl.int64) * stride_r + ic[None, :].to(tl.int64) * stride_c
    value = tl.load(in_ptr + in_off, mask=mask)
    out_off = rs[:, None].to(tl.int64) * cols + cs[None, :].to(tl.int64)
    tl.store(out_ptr + out_off, value, mask=mask)


_FLIP_DTYPES = (
    torch.float16,
    torch.bfloat16,
    torch.float32,
    torch.float64,
    torch.int16,
    torch.int32,
    torch.int64,
)


def _collapse_flip_dims(sizes, strides, flags):
    # Merge adjacent axes that share a flip flag and are memory-adjacent
    # (stride[i] == stride[i+1] * size[i+1]). Two adjacent flipped axes collapse
    # because (Si-1-i)*Sj + (Sj-1-j) == Si*Sj - 1 - (i*Sj + j); likewise two
    # unflipped axes merge into one linear run. This shrinks the per-element index
    # decode (e.g. contiguous fliplr 4D -> 3 effective axes, flip[0,1] -> 1 axis).
    merged_sizes = [sizes[0]]
    merged_strides = [strides[0]]
    merged_flags = [flags[0]]
    for i in range(1, len(sizes)):
        if flags[i] == merged_flags[-1] and merged_strides[-1] == strides[i] * sizes[i]:
            merged_sizes[-1] *= sizes[i]
            merged_strides[-1] = strides[i]
        else:
            merged_sizes.append(sizes[i])
            merged_strides.append(strides[i])
            merged_flags.append(flags[i])
    return merged_sizes, merged_strides, merged_flags


def flip_into(input, output, dims):
    # Strided flip: output is contiguous and input is read through its own strides,
    # so non-contiguous input pays no contiguous() pre-copy. Adjacent axes are
    # collapsed first to minimise the index decode; a 2D fast path then coalesces
    # the transpose-bound case, and the rest use the N-D (<=5 after collapsing)
    # kernel.
    if input.device.type != "cuda":
        return False
    if not output.is_contiguous():
        return False
    if input.dtype not in _FLIP_DTYPES:
        return False
    ndim = input.ndim
    if ndim == 0 or input.numel() == 0:
        return False

    sizes, strides, flags = _collapse_flip_dims(
        list(input.shape),
        list(input.stride()),
        [d in dims for d in range(ndim)],
    )
    ndim = len(sizes)
    if ndim > 5:
        return False

    if ndim == 2:
        rows, cols = sizes
        stride_r, stride_c = strides
        block_r, block_c, num_warps = 16, 128, 4
        grid = (triton.cdiv(rows, block_r), triton.cdiv(cols, block_c))
        _flip_2d_kernel[grid](
            input,
            output,
            rows,
            cols,
            stride_r,
            stride_c,
            FLIP_R=flags[0],
            FLIP_C=flags[1],
            BLOCK_R=block_r,
            BLOCK_C=block_c,
            num_warps=num_warps,
        )
        return True

    sizes = sizes + [1] * (5 - ndim)
    strides = strides + [0] * (5 - ndim)
    flags = flags + [False] * (5 - ndim)

    block_size = 1024
    num_warps = 4
    if input.element_size() <= 2:
        block_size = 2048
    elif input.element_size() >= 8:
        block_size = 512
    grid = (triton.cdiv(input.numel(), block_size),)
    _flip_nd_kernel[grid](
        input,
        output,
        input.numel(),
        sizes[0],
        sizes[1],
        sizes[2],
        sizes[3],
        sizes[4],
        strides[0],
        strides[1],
        strides[2],
        strides[3],
        strides[4],
        fl0=flags[0],
        fl1=flags[1],
        fl2=flags[2],
        fl3=flags[3],
        fl4=flags[4],
        NDIM=ndim,
        BLOCK=block_size,
        num_warps=num_warps,
    )
    return True


@triton.jit
def _sum_strided_kernel(
    in_ptr,
    out_ptr,
    n_elements,
    stride,
    BLOCK: tl.constexpr,
):
    # Grid-stride accumulation of n strided fp32 elements into per-program partials,
    # each atomically added to a single fp32 accumulator. Caller zeroes out_ptr.
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n_elements
    value = tl.load(in_ptr + offs.to(tl.int64) * stride, mask=mask, other=0.0)
    tl.atomic_add(out_ptr, tl.sum(value, axis=0))


def sum_strided(tensor, n_elements, stride, dtype=torch.float32):
    if tensor.device.type != "cuda":
        return None
    if n_elements <= 0:
        return torch.zeros((), dtype=dtype, device=tensor.device)

    accumulator = torch.zeros((), dtype=torch.float32, device=tensor.device)
    block_size = 1024
    num_warps = 4
    grid = (triton.cdiv(n_elements, block_size),)
    _sum_strided_kernel[grid](
        tensor,
        accumulator,
        n_elements,
        stride,
        BLOCK=block_size,
        num_warps=num_warps,
    )
    return accumulator.to(dtype)


@triton.jit
def _mse_sum_fused_kernel(
    in_ptr,
    tgt_ptr,
    out_ptr,
    n_elements,
    BLOCK: tl.constexpr,
):
    # Single-pass fused squared-difference + reduction: each program reads a BLOCK
    # chunk of both flat buffers, accumulates sum((x-t)^2) in fp32, and atomic-adds
    # its partial to one fp32 accumulator. Replaces the 2-stage row-reduce + strided
    # sum chain (one launch, one read of each operand). Caller zeroes out_ptr.
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n_elements
    x = tl.load(in_ptr + offs, mask=mask, other=0.0).to(tl.float32)
    t = tl.load(tgt_ptr + offs, mask=mask, other=0.0).to(tl.float32)
    diff = x - t
    tl.atomic_add(out_ptr, tl.sum(diff * diff, axis=0))


def mse_sum_fused(input, target):
    # Returns the fp32 scalar sum of squared differences for contiguous, same-shape
    # operands, or None if ineligible (caller keeps its existing reduction path).
    if input.device.type != "cuda":
        return None
    if input.dtype not in (torch.float16, torch.bfloat16, torch.float32, torch.float64):
        return None
    if input.shape != target.shape:
        return None
    if not input.is_contiguous() or not target.is_contiguous():
        return None

    n_elements = input.numel()
    if n_elements == 0:
        return torch.zeros((), dtype=torch.float32, device=input.device)

    accumulator = torch.zeros((), dtype=torch.float32, device=input.device)
    block_size = 2048
    num_warps = 16
    if input.dtype == torch.float64:
        block_size = 1024
        num_warps = 8
    grid = (triton.cdiv(n_elements, block_size),)
    _mse_sum_fused_kernel[grid](
        input.reshape(-1),
        target.reshape(-1),
        accumulator,
        n_elements,
        BLOCK=block_size,
        num_warps=num_warps,
    )
    return accumulator
