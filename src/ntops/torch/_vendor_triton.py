import functools
import math

import torch
import triton
import triton.language as tl


@functools.cache
def _device_name_for_index(index):
    try:
        return torch.cuda.get_device_name(index)
    except Exception:
        return ""


def _device_name(tensor):
    if not isinstance(tensor, torch.Tensor):
        return ""
    if tensor.device.type != "cuda" or not hasattr(torch, "cuda"):
        return ""
    index = tensor.device.index
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
        pivval = tl.sum(tl.where(cols == k, rowp, 0.0))
        sign = sign * tl.where(pivval < 0, -1.0, 1.0)
        logdet = logdet + tl.log(tl.abs(pivval))
        colk_vals = tl.sum(tl.where(cols[None, :] == k, a, 0.0), axis=1)
        factor = tl.where(rows > k, colk_vals / pivval, 0.0)
        a = a - factor[:, None] * rowp[None, :] * tl.where(rows[:, None] > k, 1.0, 0.0)
    tl.store(sign_ptr + pid, sign)
    tl.store(logdet_ptr + pid, logdet)


def slogdet_batched(input, sign, logdet):
    if input.device.type != "cuda":
        return False
    if not _dtype_matches(input.dtype, (torch.float32,)):
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
    # Sequential NxN LU: 1 warp covers the work, extra warps idle. On MR-V100 that
    # overhead is decisive (N<=32 peaks at 1 warp: batch1024_16x16 0.50->2.7x; N>=48
    # peaks at 4). Other platforms keep Triton's default 4 (already passing).
    if is_iluvatar_device(input):
        num_warps = 1 if n <= 32 else 4
        _slogdet_kernel[grid](input, sign, logdet, batch, N=n, BN=bn, num_warps=num_warps)
    else:
        _slogdet_kernel[grid](input, sign, logdet, batch, N=n, BN=bn)
    return True


# === gumbel_softmax vendor triton ===


@triton.jit(do_not_specialize=["seed"])
def _gumbel_softmax_soft_kernel(
    logits_ptr, out_ptr, rows, cols, seed, tau,
    ROWS_PER_BLOCK: tl.constexpr, BLOCK_COLS: tl.constexpr,
):
    # ROWS_PER_BLOCK rows per program, each row in one BLOCK_COLS tile (the wrapper
    # guarantees BLOCK_COLS >= cols). Row-tiling amortizes launch/occupancy cost for
    # small column counts. Draw the Gumbel noise once and run a single-pass
    # numerically-stable softmax (half the RNG + memory traffic of the NineToothed kernel).
    #
    # Internal math is fp32 for every storage dtype. gumbel_softmax is a stochastic
    # sampler (perf compare=False; correctness only checks the simplex sum to 1e-4), so
    # an fp32 softmax (sum error ~1e-7) is well inside tolerance, and this avoids CoreX
    # fp64 transcendentals (MR-V100: fp64-internal 0.31ms ratio 0.30 vs fp32 0.024ms 4-5x).
    row_base = tl.program_id(0) * ROWS_PER_BLOCK
    rows_off = row_base + tl.arange(0, ROWS_PER_BLOCK)
    cols_off = tl.arange(0, BLOCK_COLS)
    row_mask = rows_off[:, None] < rows
    col_mask = cols_off[None, :] < cols
    mask = row_mask & col_mask
    idx = rows_off[:, None] * cols + cols_off[None, :]
    x = tl.load(logits_ptr + idx, mask=mask, other=0.0).to(tl.float32)
    u = tl.rand(seed, idx)
    g = -tl.log(-tl.log(u + 1e-20) + 1e-20)
    val = (x + g) / tau
    val = tl.where(mask, val, float("-inf"))
    m = tl.max(val, axis=1)
    e = tl.exp(val - m[:, None])
    denom = tl.sum(e, axis=1)
    res = e / denom[:, None]
    tl.store(out_ptr + idx, res.to(out_ptr.dtype.element_ty), mask=mask)


def _rows_per_block(cols_pow2):
    if cols_pow2 <= 16:
        return 64
    if cols_pow2 <= 32:
        return 32
    if cols_pow2 <= 64:
        return 16
    if cols_pow2 <= 128:
        return 4
    return 1


def gumbel_softmax_soft(logits, output, seed, tau):
    if logits.device.type != "cuda":
        return False
    if logits.dtype not in (torch.float32, torch.float64):
        return False
    if logits.ndim != 2 or not logits.is_contiguous() or not output.is_contiguous():
        return False
    rows, cols = logits.shape
    if rows == 0 or cols == 0:
        return False
    block = 1
    while block < cols:
        block *= 2
    if block > 8192:
        return False
    rpb = _rows_per_block(block)
    tile = rpb * block
    if tile <= 256:
        num_warps = 2
    elif tile <= 1024:
        num_warps = 4
    else:
        num_warps = 8
    _gumbel_softmax_soft_kernel[(triton.cdiv(rows, rpb),)](
        logits, output, rows, cols, int(seed), float(tau),
        ROWS_PER_BLOCK=rpb, BLOCK_COLS=block,
        num_warps=num_warps,
    )
    return True


@triton.jit
def _gumbel_hard_one_hot_kernel(
    soft_ptr, hard_ptr, rows, cols,
    ROWS_PER_BLOCK: tl.constexpr, BLOCK_COLS: tl.constexpr,
):
    # Per-row straight-through one-hot: argmax of the soft sample, write 1 at the
    # argmax and 0 elsewhere. Fuses argmax + zero-fill + scatter (3 launches in the
    # torch path) into a single launch; row-tiled for small column counts.
    row_base = tl.program_id(0) * ROWS_PER_BLOCK
    rows_off = row_base + tl.arange(0, ROWS_PER_BLOCK)
    cols_off = tl.arange(0, BLOCK_COLS)
    row_mask = rows_off[:, None] < rows
    col_mask = cols_off[None, :] < cols
    mask = row_mask & col_mask
    idx = rows_off[:, None] * cols + cols_off[None, :]
    x = tl.load(soft_ptr + idx, mask=mask, other=float("-inf")).to(tl.float32)
    am = tl.argmax(x, axis=1)
    onehot = tl.where(cols_off[None, :] == am[:, None], 1.0, 0.0)
    tl.store(hard_ptr + idx, onehot.to(hard_ptr.dtype.element_ty), mask=mask)


def gumbel_hard_one_hot(soft, hard):
    if soft.device.type != "cuda":
        return False
    if soft.dtype not in (torch.float32, torch.float64):
        return False
    if soft.ndim != 2 or not soft.is_contiguous() or not hard.is_contiguous():
        return False
    rows, cols = soft.shape
    if rows == 0 or cols == 0:
        return False
    block = 1
    while block < cols:
        block *= 2
    if block > 8192:
        return False
    rpb = _rows_per_block(block)
    tile = rpb * block
    num_warps = 2 if tile <= 256 else (4 if tile <= 1024 else 8)
    _gumbel_hard_one_hot_kernel[(triton.cdiv(rows, rpb),)](
        soft, hard, rows, cols,
        ROWS_PER_BLOCK=rpb, BLOCK_COLS=block, num_warps=num_warps,
    )
    return True


# === heaviside vendor triton ===


@triton.jit
def _heaviside_kernel(
    in_ptr, val_ptr, out_ptr, n,
    VAL_SCALAR: tl.constexpr, IS_FLOAT: tl.constexpr, BLOCK_SIZE: tl.constexpr,
):
    # Elementwise heaviside over a dense linear span (works for contiguous tensors
    # and for transpose-style strided tensors that share an identical dense layout,
    # since heaviside is position-preserving). values may be a single broadcast
    # scalar (VAL_SCALAR) read once. torch.heaviside: x>0 -> 1, x==0 -> values,
    # x<0 or NaN -> 0.
    offs = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < n
    x = tl.load(in_ptr + offs, mask=mask, other=0)
    if VAL_SCALAR:
        v = tl.load(val_ptr)
    else:
        v = tl.load(val_ptr + offs, mask=mask, other=0)
    if IS_FLOAT:
        xc = x.to(tl.float32)
        gt = xc > 0.0
        eq = xc == 0.0
    else:
        gt = x > 0
        eq = x == 0
    one = tl.full((BLOCK_SIZE,), 1, out_ptr.dtype.element_ty)
    zero = tl.full((BLOCK_SIZE,), 0, out_ptr.dtype.element_ty)
    res = tl.where(gt, one, tl.where(eq, v.to(out_ptr.dtype.element_ty), zero))
    tl.store(out_ptr + offs, res, mask=mask)


_HEAVISIDE_FLOAT = (torch.float16, torch.bfloat16, torch.float32, torch.float64)
_HEAVISIDE_INT = (torch.int16, torch.int32, torch.int64)


def _dtype_matches(dtype, reference_set):
    if dtype in reference_set:
        return True
    name = str(dtype).rpartition(".")[2]
    return any(str(r).rpartition(".")[2] == name for r in reference_set)


def _dense_offset0(tensor):
    try:
        if tensor.storage_offset() != 0:
            return False
    except Exception:
        return False
    try:
        return tensor.numel() == tensor.untyped_storage().size() // tensor.element_size()
    except Exception:
        return False


def heaviside_fast_path(input, values, out, result_dtype):
    if input.device.type != "cuda":
        return False
    if not _dtype_matches(result_dtype, _HEAVISIDE_FLOAT) and not _dtype_matches(result_dtype, _HEAVISIDE_INT):
        return False
    if input.dtype != result_dtype or values.dtype != result_dtype:
        return False

    n = out.numel()
    if n == 0:
        return True

    val_scalar = values.numel() == 1

    if input.is_contiguous() and out.is_contiguous() and input.numel() == n:
        if val_scalar:
            val_t = values.reshape(1)[:1]
        elif values.is_contiguous() and values.numel() == n:
            val_t = values
        else:
            return False
    elif (
        not val_scalar
        and tuple(input.shape) == tuple(values.shape) == tuple(out.shape)
        and input.stride() == values.stride() == out.stride()
        and _dense_offset0(input) and _dense_offset0(values) and _dense_offset0(out)
    ):
        val_t = values
    else:
        return False

    block_size = 2048
    num_warps = 8
    if result_dtype == torch.float64:
        block_size = 1024
    elif result_dtype == torch.int64:
        block_size = 1024
    grid = (triton.cdiv(n, block_size),)
    _heaviside_kernel[grid](
        input, val_t, out, n,
        VAL_SCALAR=val_scalar, IS_FLOAT=(_dtype_matches(result_dtype, _HEAVISIDE_FLOAT)),
        BLOCK_SIZE=block_size, num_warps=num_warps,
    )
    return True


# === slice_scatter vendor triton ===


@triton.jit
def _slice_scatter_kernel(
    in_ptr, src_ptr, out_ptr,
    outer, dim_size, inner, src_dim_size, lo, step, n,
    BLOCK_SIZE: tl.constexpr,
):
    # Single coalesced pass over the contiguous output: each element is either copied
    # from the slice source (when its coordinate along `dim` lands inside
    # [lo, lo+step*src_dim_size) on a step boundary) or from the base input. Replaces
    # the two-copy (whole-input copy + strided as_strided view copy) path, halving the
    # write traffic and keeping the output store fully coalesced.
    offs = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < n
    inner_idx = offs % inner
    tmp = offs // inner
    coord = tmp % dim_size
    outer_idx = tmp // dim_size

    rel = coord - lo
    in_slice = (coord >= lo) & (rel < src_dim_size * step) & (rel % step == 0)
    src_coord = rel // step
    src_flat = (outer_idx * src_dim_size + src_coord) * inner + inner_idx

    base_val = tl.load(in_ptr + offs, mask=mask, other=0)
    src_val = tl.load(src_ptr + src_flat, mask=mask & in_slice, other=0)
    res = tl.where(in_slice, src_val, base_val)
    tl.store(out_ptr + offs, res, mask=mask)


def slice_scatter_into(input, src, output, dim, lo, step, length):
    if input.device.type != "cuda":
        return False
    if not input.is_contiguous() or not src.is_contiguous() or not output.is_contiguous():
        return False
    if input.dtype != src.dtype or input.dtype != output.dtype:
        return False
    n = output.numel()
    if n == 0:
        return True
    shape = input.shape
    dim_size = shape[dim]
    inner = 1
    for s in shape[dim + 1:]:
        inner *= s
    outer = 1
    for s in shape[:dim]:
        outer *= s
    block_size = 2048
    num_warps = 4
    if input.dtype in (torch.float64, torch.int64):
        block_size = 1024
    grid = (triton.cdiv(n, block_size),)
    _slice_scatter_kernel[grid](
        input, src, output,
        outer, dim_size, inner, length, lo, step, n,
        BLOCK_SIZE=block_size, num_warps=num_warps,
    )
    return True
