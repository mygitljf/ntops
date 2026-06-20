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
    if not hasattr(tensor, "device") or not hasattr(tensor.device, "type"):
        return ""
    if tensor.device.type != "cuda":
        return ""
    index = getattr(tensor.device, "index", None)
    if index is None:
        if hasattr(torch, "cuda"):
            try:
                index = torch.cuda.current_device()
            except Exception:
                return ""
        else:
            return ""
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


# ---------------------------------------------------------------------------
# sum_strided: finish a two-stage row reduction.
#
# kl_div / count_nonzero fuse a pointwise term with a per-row partial reduction
# (the NineToothed reduction arrangement writes the row total into column 0 of a
# (rows, cols) buffer).  This kernel sums those `count` row-totals -- read at a
# fixed element `stride` -- into a single scalar, keeping the whole reduction on
# the GPU (no torch.sum fallback).  Accumulation dtype matches the partial buffer
# (fp32 / fp64 for kl_div, int64 for count_nonzero).
# ---------------------------------------------------------------------------


@triton.jit
def _sum_strided_kernel(
    in_ptr,
    out_ptr,
    count,
    stride,
    scale,
    apply_scale: tl.constexpr,
    BLOCK: tl.constexpr,
):
    acc = tl.zeros((BLOCK,), dtype=in_ptr.dtype.element_ty)
    base = tl.arange(0, BLOCK)
    for start in tl.range(0, count, BLOCK):
        idx = start + base
        mask = idx < count
        vals = tl.load(in_ptr + idx * stride, mask=mask, other=0)
        acc += vals
    total = tl.sum(acc, axis=0)
    if apply_scale:
        total *= scale
    tl.store(out_ptr, total)


def sum_strided(partials, count, stride, dtype, scale=None):
    if partials.device.type != "cuda":
        return None
    if count <= 0:
        try:
            return torch.zeros((), dtype=dtype, device=partials.device)
        except (TypeError, RuntimeError):
            import infinicore
            return infinicore.zeros([], dtype=dtype, device=partials.device)

    try:
        out = torch.empty((), dtype=dtype, device=partials.device)
    except (TypeError, RuntimeError):
        import infinicore
        out = infinicore.empty([], dtype=dtype, device=partials.device)
    # One program reduces the whole (small) partial chain; the row count is
    # numel // cols, i.e. a few thousand at most for our shapes, so a single
    # block with a bounded inner loop keeps launch overhead negligible.
    block = 1
    while block < count and block < 4096:
        block *= 2
    _sum_strided_kernel[(1,)](
        partials,
        out,
        count,
        stride,
        1.0 if scale is None else scale,
        apply_scale=scale is not None,
        BLOCK=block,
        num_warps=8,
    )
    return out


# ---------------------------------------------------------------------------
# count_nonzero_reduce: 2D-tiled nonzero count over the last logical axis.
#
# The NineToothed reduction arrangement runs one program per reduced row, so a
# leading-dim reduction (dim=0) feeds it a transposed view whose row elements sit
# at stride=ncols -> each program strided-reads a whole row, fully non-coalesced
# (0.17x on MetaX C500). It also writes the row total into every column of an
# (M, N) int64 buffer (write amplification). This kernel instead tiles MB adjacent
# rows per program: the contiguous row axis vectorizes the load (coalesced), and
# only M int64 results are written, fixing both dim0 (strided read) and inner-dim
# (write amplification) regressions on MetaX without an A-class native fallback.
# ---------------------------------------------------------------------------


@triton.jit
def _count_nonzero_2d_kernel(
    in_ptr, out_ptr, M, N, row_stride, col_stride, MB: tl.constexpr, NB: tl.constexpr
):
    pid = tl.program_id(0)
    m = pid * MB + tl.arange(0, MB)
    mmask = m < M
    acc = tl.zeros((MB,), dtype=tl.int32)
    for n0 in tl.range(0, N, NB):
        n = n0 + tl.arange(0, NB)
        nmask = n < N
        ptr = in_ptr + m[:, None] * row_stride + n[None, :] * col_stride
        v = tl.load(ptr, mask=mmask[:, None] & nmask[None, :], other=0)
        acc += tl.sum((v != 0).to(tl.int32), axis=1)
    tl.store(out_ptr + m, acc.to(tl.int64), mask=mmask)


@triton.jit
def _count_nonzero_col_kernel(in_ptr, out_ptr, R, C, RB: tl.constexpr, CB: tl.constexpr):
    # Reduce the leading R rows, keep the C trailing contiguous columns. Output (C)
    # is tiny while R is huge (e.g. reduce 65536 rows, keep 256 cols), so one program
    # per column slab serializes the whole R reduction. Split-K instead: a 2D grid
    # (col slab, row slab) lets each program reduce RB rows of CB contiguous (stride-1,
    # coalesced) columns and atomic-add its int32 partial, parallelizing the dominant
    # R axis. Per-column count <= R fits int32 (matches the row kernel accumulator).
    pid_c = tl.program_id(0)
    pid_r = tl.program_id(1)
    c = pid_c * CB + tl.arange(0, CB)
    cmask = c < C
    r = pid_r * RB + tl.arange(0, RB)
    rmask = r < R
    v = tl.load(
        in_ptr + r[:, None] * C + c[None, :],
        mask=rmask[:, None] & cmask[None, :],
        other=0,
    )
    partial = tl.sum((v != 0).to(tl.int32), axis=0)
    tl.atomic_add(out_ptr + c, partial, mask=cmask)


def count_nonzero_reduce(flat):
    # Only the CoreX/MetaX path needs this: NVIDIA H100 already meets threshold on
    # the proven NineToothed reduction, so it stays byte-identical there.
    if not is_corex_or_metax_device(flat):
        return None
    if flat.device.type != "cuda" or flat.ndim != 2:
        return None

    m, n = flat.shape
    try:
        out = torch.empty(m, dtype=torch.int64, device=flat.device)
    except (TypeError, RuntimeError):
        import infinicore
        out = infinicore.empty([m], dtype=infinicore.int64, device=flat.device)
    mb = 64
    nb = 256
    grid = (triton.cdiv(m, mb),)
    _count_nonzero_2d_kernel[grid](
        flat, out, m, n, flat.stride(0), flat.stride(1),
        MB=mb, NB=nb, num_warps=8,
    )
    return out


def count_nonzero_col_reduce(flat):
    # flat is contiguous (R, C); reduce axis 0, keep the C trailing columns.
    if not is_corex_or_metax_device(flat):
        return None
    if flat.device.type != "cuda" or flat.ndim != 2 or not flat.is_contiguous():
        return None

    r, c = flat.shape
    infini_fallback = False
    try:
        out = torch.zeros(c, dtype=torch.int32, device=flat.device)
    except (TypeError, RuntimeError):
        import infinicore
        out = infinicore.zeros([c], dtype=infinicore.int32, device=flat.device)
        infini_fallback = True
    cb = 128
    rb = 128
    grid = (triton.cdiv(c, cb), triton.cdiv(r, rb))
    _count_nonzero_col_kernel[grid](
        flat, out, r, c, RB=rb, CB=cb, num_warps=8,
    )
    if infini_fallback:
        return out
    return out.to(torch.int64)


# ---------------------------------------------------------------------------
# combinations_fast: index-unranking kernels for r=2 and r=3.
#
# Each output row index p is mapped directly to its tuple of source indices by
# triangular (r=2) / tetrahedral (r=3) unranking with an exact int64 binary
# search -- no float sqrt, so it is correct for any n.  This replaces torch's
# native generator with a single coalesced-write kernel.
# ---------------------------------------------------------------------------


@triton.jit
def _combinations_r2_kernel(
    values_ptr, out_ptr, n, total, REPL: tl.constexpr, NITERS: tl.constexpr, BLOCK: tl.constexpr
):
    p = (tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)).to(tl.int64)
    mask = p < total
    n64 = tl.full((BLOCK,), n, tl.int64)

    lo = tl.zeros((BLOCK,), tl.int64)
    hi = n64 - 1
    for _ in tl.static_range(NITERS):
        mid = (lo + hi + 1) // 2
        if REPL:
            pref = mid * (2 * n64 - mid + 1) // 2
        else:
            pref = mid * (2 * n64 - 1 - mid) // 2
        take = pref <= p
        lo = tl.where(take, mid, lo)
        hi = tl.where(take, hi, mid - 1)
    i = lo
    if REPL:
        prefi = i * (2 * n64 - i + 1) // 2
        j = p - prefi + i
    else:
        prefi = i * (2 * n64 - 1 - i) // 2
        j = p - prefi + i + 1

    vi = tl.load(values_ptr + i, mask=mask, other=0)
    vj = tl.load(values_ptr + j, mask=mask, other=0)
    tl.store(out_ptr + p * 2, vi, mask=mask)
    tl.store(out_ptr + p * 2 + 1, vj, mask=mask)


@triton.jit
def _combinations_r3_kernel(
    values_ptr, out_ptr, n, total, REPL: tl.constexpr, NITERS: tl.constexpr, BLOCK: tl.constexpr
):
    p = (tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)).to(tl.int64)
    mask = p < total
    n64 = tl.full((BLOCK,), n, tl.int64)

    # total tetrahedral count: C3(n) (no-repl) or C3(n+2) (with-repl), where
    # C3(m) = m*(m-1)*(m-2)/6 (the product is 0 for m < 3, so no guard needed).
    if REPL:
        tot = (n64 + 2) * (n64 + 1) * n64 // 6
    else:
        tot = n64 * (n64 - 1) * (n64 - 2) // 6

    # first index i: largest i with prefix(i) <= p, prefix(i) = tot - C3(rem),
    # rem = (n - i) (+2 for repl).
    lo = tl.zeros((BLOCK,), tl.int64)
    hi = n64 - 1
    for _ in tl.static_range(NITERS):
        mid = (lo + hi + 1) // 2
        if REPL:
            rem = n64 - mid + 2
        else:
            rem = n64 - mid
        c3 = rem * (rem - 1) * (rem - 2) // 6
        pref = tot - c3
        take = pref <= p
        lo = tl.where(take, mid, lo)
        hi = tl.where(take, hi, mid - 1)
    i = lo
    if REPL:
        remi = n64 - i + 2
    else:
        remi = n64 - i
    c3i = remi * (remi - 1) * (remi - 2) // 6
    r1 = p - (tot - c3i)

    # second / third indices unrank as an r=2 problem over M items, base offset.
    if REPL:
        M = n64 - i
        base = i
    else:
        M = n64 - 1 - i
        base = i + 1

    lo2 = tl.zeros((BLOCK,), tl.int64)
    hi2 = M - 1
    for _ in tl.static_range(NITERS):
        mid = (lo2 + hi2 + 1) // 2
        if REPL:
            pref2 = mid * (2 * M - mid + 1) // 2
        else:
            pref2 = mid * (2 * M - 1 - mid) // 2
        take = pref2 <= r1
        lo2 = tl.where(take, mid, lo2)
        hi2 = tl.where(take, hi2, mid - 1)
    jl = lo2
    if REPL:
        pref2j = jl * (2 * M - jl + 1) // 2
        kl = r1 - pref2j + jl
    else:
        pref2j = jl * (2 * M - 1 - jl) // 2
        kl = r1 - pref2j + jl + 1
    j = base + jl
    k = base + kl

    vi = tl.load(values_ptr + i, mask=mask, other=0)
    vj = tl.load(values_ptr + j, mask=mask, other=0)
    vk = tl.load(values_ptr + k, mask=mask, other=0)
    tl.store(out_ptr + p * 3, vi, mask=mask)
    tl.store(out_ptr + p * 3 + 1, vj, mask=mask)
    tl.store(out_ptr + p * 3 + 2, vk, mask=mask)


@triton.jit
def _combinations_r4_kernel(
    values_ptr, out_ptr, n, total, REPL: tl.constexpr, NITERS: tl.constexpr, BLOCK: tl.constexpr
):
    p = (tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)).to(tl.int64)
    mask = p < total
    n64 = tl.full((BLOCK,), n, tl.int64)

    # r=4 unranks recursively: pick first index i by tetra-of-4 prefix counts,
    # C4(m) = m*(m-1)*(m-2)*(m-3)/24, then unrank the remaining 3 as an r=3 problem
    # over M items (exact int64 binary search at every level, no float sqrt).
    if REPL:
        tot = (n64 + 3) * (n64 + 2) * (n64 + 1) * n64 // 24
    else:
        tot = n64 * (n64 - 1) * (n64 - 2) * (n64 - 3) // 24

    lo = tl.zeros((BLOCK,), tl.int64)
    hi = n64 - 1
    for _ in tl.static_range(NITERS):
        mid = (lo + hi + 1) // 2
        rem = n64 - mid + 3 if REPL else n64 - mid
        c4 = rem * (rem - 1) * (rem - 2) * (rem - 3) // 24
        take = (tot - c4) <= p
        lo = tl.where(take, mid, lo)
        hi = tl.where(take, hi, mid - 1)
    i = lo
    remi = n64 - i + 3 if REPL else n64 - i
    c4i = remi * (remi - 1) * (remi - 2) * (remi - 3) // 24
    r1 = p - (tot - c4i)

    if REPL:
        M = n64 - i
        base = i
    else:
        M = n64 - 1 - i
        base = i + 1

    if REPL:
        tot3 = (M + 2) * (M + 1) * M // 6
    else:
        tot3 = M * (M - 1) * (M - 2) // 6
    lo = tl.zeros((BLOCK,), tl.int64)
    hi = M - 1
    for _ in tl.static_range(NITERS):
        mid = (lo + hi + 1) // 2
        rem = M - mid + 2 if REPL else M - mid
        c3 = rem * (rem - 1) * (rem - 2) // 6
        take = (tot3 - c3) <= r1
        lo = tl.where(take, mid, lo)
        hi = tl.where(take, hi, mid - 1)
    j0 = lo
    remj = M - j0 + 2 if REPL else M - j0
    c3j = remj * (remj - 1) * (remj - 2) // 6
    r2 = r1 - (tot3 - c3j)

    if REPL:
        M2 = M - j0
        base2 = base + j0
    else:
        M2 = M - 1 - j0
        base2 = base + j0 + 1

    lo = tl.zeros((BLOCK,), tl.int64)
    hi = M2 - 1
    for _ in tl.static_range(NITERS):
        mid = (lo + hi + 1) // 2
        if REPL:
            pref = mid * (2 * M2 - mid + 1) // 2
        else:
            pref = mid * (2 * M2 - 1 - mid) // 2
        take = pref <= r2
        lo = tl.where(take, mid, lo)
        hi = tl.where(take, hi, mid - 1)
    k0 = lo
    if REPL:
        l0 = r2 - k0 * (2 * M2 - k0 + 1) // 2 + k0
    else:
        l0 = r2 - k0 * (2 * M2 - 1 - k0) // 2 + k0 + 1

    idx0 = i
    idx1 = base + j0
    idx2 = base2 + k0
    idx3 = base2 + l0

    v0 = tl.load(values_ptr + idx0, mask=mask, other=0)
    v1 = tl.load(values_ptr + idx1, mask=mask, other=0)
    v2 = tl.load(values_ptr + idx2, mask=mask, other=0)
    v3 = tl.load(values_ptr + idx3, mask=mask, other=0)
    tl.store(out_ptr + p * 4, v0, mask=mask)
    tl.store(out_ptr + p * 4 + 1, v1, mask=mask)
    tl.store(out_ptr + p * 4 + 2, v2, mask=mask)
    tl.store(out_ptr + p * 4 + 3, v3, mask=mask)


def _comb_niters(n):
    iters = 1
    bound = 1
    while bound < max(n, 2):
        bound *= 2
        iters += 1
    return iters


def combinations_fast(values, output, r, with_replacement):
    if values.device.type != "cuda":
        return False
    if r not in (2, 3, 4):
        return False
    if not values.is_contiguous() or not output.is_contiguous():
        return False
    if values.dtype != output.dtype:
        return False

    n = values.numel()
    total = output.shape[0]
    if total == 0:
        return True

    niters = _comb_niters(n)
    block = 256
    grid = (triton.cdiv(total, block),)

    if r == 2:
        _combinations_r2_kernel[grid](
            values, output, n, total,
            REPL=bool(with_replacement), NITERS=niters, BLOCK=block,
            num_warps=4,
        )
    elif r == 3:
        _combinations_r3_kernel[grid](
            values, output, n, total,
            REPL=bool(with_replacement), NITERS=niters, BLOCK=block,
            num_warps=4,
        )
    else:
        _combinations_r4_kernel[grid](
            values, output, n, total,
            REPL=bool(with_replacement), NITERS=niters, BLOCK=block,
            num_warps=4,
        )
    return True


# ---------------------------------------------------------------------------
# gram_fast: split-K Gram matrix (N @ N^T) for few-row, wide-column matrices.
#
# corrcoef on a (R, K) matrix with R << K (e.g. 8 x 65536) makes the Gram matmul
# output tiny (R x R) but the K reduction huge.  A standard tiled matmul assigns
# the whole R x R output to one program, serializing the entire K reduction.
# Here each program reduces one BK-wide K slab into a partial R x R Gram and
# atomic-adds it, so the K axis is parallelized across programs (split-K).  The
# tile (R_PAD x BK) is kept <= 16384 elements to stay within shared memory for
# tl.dot.  fp32 accumulation with allow_tf32=False keeps full f32 precision.
# ---------------------------------------------------------------------------


@triton.jit
def _gram_splitk_kernel(n_ptr, out_ptr, R, K, R_PAD: tl.constexpr, BK: tl.constexpr):
    pid = tl.program_id(0)
    k0 = pid * BK
    rows = tl.arange(0, R_PAD)
    ks = tl.arange(0, BK)
    rmask = rows < R
    kmask = (k0 + ks) < K
    tile = tl.load(
        n_ptr + rows[:, None] * K + (k0 + ks)[None, :],
        mask=rmask[:, None] & kmask[None, :],
        other=0.0,
    )
    part = tl.dot(tile, tl.trans(tile), allow_tf32=False)
    tl.atomic_add(
        out_ptr + rows[:, None] * R + rows[None, :],
        part,
        mask=rmask[:, None] & rmask[None, :],
    )


def _gram_rpad_bk(rows, elem_cap=16384):
    # elem_cap bounds the R_PAD x BK tile so tl.dot's operands fit in shared memory.
    # NVIDIA H100 (228 KB smem) tolerates a 16384-element (64 KB) tile; MetaX C500
    # caps shared memory at 64 KB, and the extra row-sum scratch in the fused kernel
    # pushes a 64 KB tile to 65792 bytes (OutOfResources), so callers pass a smaller
    # cap there to keep the kernel compilable without an A-class native fallback.
    r_pad = 16
    while r_pad < rows:
        r_pad *= 2
    bk = max(64, min(2048, elem_cap // r_pad))
    block = 64
    while block * 2 <= bk:
        block *= 2
    return r_pad, block


def gram_fast(normalized):
    if normalized.device.type != "cuda":
        return None
    if normalized.dtype != torch.float32:
        return None
    if normalized.ndim != 2 or not normalized.is_contiguous():
        return None

    rows, k = normalized.shape
    # The split-K win only matters when K dominates a small output; for square-ish
    # or tall matrices the autotuned mm is already better, so defer to it there.
    if k < 4 * rows or k < 2048:
        return None

    r_pad, bk = _gram_rpad_bk(rows)
    out = torch.zeros((rows, rows), dtype=torch.float32, device=normalized.device)
    grid = (triton.cdiv(k, bk),)
    _gram_splitk_kernel[grid](
        normalized, out, rows, k,
        R_PAD=r_pad, BK=bk, num_warps=8,
    )
    return out


@triton.jit
def _sumsq_kernel(x_ptr, out_ptr, K, BK: tl.constexpr):
    pid = tl.program_id(0)
    ks = pid * BK + tl.arange(0, BK)
    mask = ks < K
    v = tl.load(x_ptr + ks, mask=mask, other=0.0)
    tl.atomic_add(out_ptr + 0, tl.sum(v))
    tl.atomic_add(out_ptr + 1, tl.sum(v * v))


def corrcoef_single_row(values):
    # A single observation row reduces corrcoef to a scalar: 1.0 when the row has
    # nonzero variance, NaN otherwise (matching torch). One split-K sum/sum-of-
    # squares pass replaces the full normalize+gram launches for this degenerate
    # but launch-bound shape.
    if values.device.type != "cuda":
        return None
    if values.dtype != torch.float32 or not values.is_contiguous():
        return None

    k = values.numel()
    if k < 2048:
        return None

    acc = torch.zeros(2, dtype=torch.float32, device=values.device)
    bk = 2048
    grid = (triton.cdiv(k, bk),)
    _sumsq_kernel[grid](values, acc, k, BK=bk, num_warps=8)
    mean = acc[0] / k
    ss = acc[1] - k * mean * mean
    return torch.where(
        ss > 0,
        torch.ones((), dtype=torch.float32, device=values.device),
        torch.full((), float("nan"), dtype=torch.float32, device=values.device),
    )


# ---------------------------------------------------------------------------
# corrcoef_fused: single-pass corrcoef for few-row, very-wide matrices.
#
# The two-stage (normalize -> gram_fast) path reads X twice and pays two launches,
# which dominates for tiny-R / huge-K matrices (e.g. 8 x 65536) that are launch
# bound.  Here one split-K pass reads X once, accumulating both the row sums and
# the raw Gram (X @ X^T); a tiny finalize kernel turns those into the correlation
# matrix: cov = G - K * mean_i*mean_j, corr = cov / sqrt(var_i*var_j), clamped to
# [-1, 1] with NaN preserved for zero-variance rows (matching torch).  The
# finalize tile is R_PAD x R_PAD in one program, so this is gated to small R.
# ---------------------------------------------------------------------------


@triton.jit
def _corr_stats_gram_kernel(x_ptr, sum_ptr, gram_ptr, R, K, R_PAD: tl.constexpr, BK: tl.constexpr):
    pid = tl.program_id(0)
    k0 = pid * BK
    rows = tl.arange(0, R_PAD)
    ks = tl.arange(0, BK)
    rmask = rows < R
    kmask = (k0 + ks) < K
    tile = tl.load(
        x_ptr + rows[:, None] * K + (k0 + ks)[None, :],
        mask=rmask[:, None] & kmask[None, :],
        other=0.0,
    )
    tl.atomic_add(sum_ptr + rows, tl.sum(tile, axis=1), mask=rmask)
    g = tl.dot(tile, tl.trans(tile), allow_tf32=False)
    tl.atomic_add(
        gram_ptr + rows[:, None] * R + rows[None, :],
        g,
        mask=rmask[:, None] & rmask[None, :],
    )


@triton.jit
def _corr_finalize_kernel(graw_ptr, sum_ptr, out_ptr, R, K, R_PAD: tl.constexpr):
    rows = tl.arange(0, R_PAD)
    cols = tl.arange(0, R_PAD)
    rmask = rows < R
    cmask = cols < R
    m = rmask[:, None] & cmask[None, :]
    g = tl.load(graw_ptr + rows[:, None] * R + cols[None, :], mask=m, other=0.0)
    si = tl.load(sum_ptr + rows, mask=rmask, other=0.0)
    sj = tl.load(sum_ptr + cols, mask=cmask, other=0.0)
    cov = g - (si / K)[:, None] * (sj / K)[None, :] * K
    var = tl.sum(tl.where(rows[:, None] == cols[None, :], cov, 0.0), axis=1)
    # inject NaN at the variance source so a zero-variance row's whole row/column
    # becomes NaN (torch's behaviour); IEEE min/max would otherwise drop it.
    inv = tl.where(var > 0.0, 1.0 / tl.sqrt(var), float("nan"))
    res = cov * inv[:, None] * inv[None, :]
    clamped = tl.minimum(tl.maximum(res, -1.0), 1.0)
    final = tl.where(res == res, clamped, res)
    tl.store(out_ptr + rows[:, None] * R + cols[None, :], final, mask=m)


def corrcoef_fused(matrix):
    if matrix.device.type != "cuda":
        return None
    if matrix.dtype != torch.float32 or matrix.ndim != 2 or not matrix.is_contiguous():
        return None

    # CoreX (Iluvatar) Triton miscompiles a kernel that fuses tl.sum(axis=1) and
    # tl.dot over the same tile: _corr_stats_gram_kernel aborts with an MLIR
    # "cast<Ty>() argument of incompatible type" assertion at codegen. Routing
    # Iluvatar to the two-stage normalize + gram_fast (split-K dot, no in-kernel
    # tl.sum) path keeps a pure-kernel route — gram_fast always fires here because
    # k >= 8192 >= 4*rows >= 2048 — and is fast (1.97x / 3.53x on the two affected
    # shapes). This is a B-class capability route between kernels, not an A-class
    # PyTorch/native performance fallback.
    if is_iluvatar_device(matrix):
        return None

    rows, k = matrix.shape
    # Gate to the launch-bound few-row / wide-K regime: the single-pass read beats
    # normalize+gram only when R is tiny and K is huge. Outside this, the finalize
    # RxR tile and atomic contention make it slower than the two-stage path.
    if rows > 16 or k < 8192:
        return None

    # MetaX C500 caps shared memory at 64 KB; a 16384-element f32 tile plus the
    # row-sum scratch needs 65792 bytes there (OutOfResources). Halving the tile
    # element budget keeps the same split-K kernel compilable on MetaX while
    # leaving the larger NVIDIA tile untouched.
    elem_cap = 8192 if is_metax_device(matrix) else 16384
    r_pad, bk = _gram_rpad_bk(rows, elem_cap=elem_cap)
    rsum = torch.zeros(rows, dtype=torch.float32, device=matrix.device)
    graw = torch.zeros((rows, rows), dtype=torch.float32, device=matrix.device)
    _corr_stats_gram_kernel[(triton.cdiv(k, bk),)](
        matrix, rsum, graw, rows, k,
        R_PAD=r_pad, BK=bk, num_warps=8,
    )
    out = torch.empty((rows, rows), dtype=torch.float32, device=matrix.device)
    _corr_finalize_kernel[(1,)](graw, rsum, out, rows, k, R_PAD=r_pad)
    return out
