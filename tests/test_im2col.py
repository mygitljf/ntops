import pytest
import torch
import torch.nn.functional as F

import ntops
from tests.skippers import skip_if_cuda_not_available


def _make_input(shape, dtype, noncontiguous=False):
    if dtype == torch.bool:
        input = torch.randint(0, 2, shape, dtype=dtype, device="cuda")
    elif dtype.is_floating_point:
        input = torch.randn(shape, dtype=dtype, device="cuda")
    else:
        input = torch.randint(-10, 10, shape, dtype=dtype, device="cuda")

    if noncontiguous:
        input = input.transpose(2, 3)

    return input


def _pair(value):
    if isinstance(value, int):
        return value, value
    value = tuple(value)
    if len(value) == 1:
        return value[0], value[0]
    return value


def _reference_im2col(input, kernel_size, dilation=1, padding=0, stride=1):
    kernel_size = _pair(kernel_size)
    dilation = _pair(dilation)
    padding = _pair(padding)
    stride = _pair(stride)

    kernel_h, kernel_w = kernel_size
    dilation_h, dilation_w = dilation
    padding_h, padding_w = padding
    stride_h, stride_w = stride
    n, c, h, w = input.shape
    out_h = (h + 2 * padding_h - dilation_h * (kernel_h - 1) - 1) // stride_h + 1
    out_w = (w + 2 * padding_w - dilation_w * (kernel_w - 1) - 1) // stride_w + 1

    padded = F.pad(input, (padding_w, padding_w, padding_h, padding_h))
    blocks = []
    for r in range(kernel_h):
        for s in range(kernel_w):
            blocks.append(
                padded[
                    :,
                    :,
                    r * dilation_h : r * dilation_h + out_h * stride_h : stride_h,
                    s * dilation_w : s * dilation_w + out_w * stride_w : stride_w,
                ]
            )

    return torch.stack(blocks, dim=2).reshape(n, c * kernel_h * kernel_w, out_h * out_w)


@skip_if_cuda_not_available
@pytest.mark.parametrize(
    "dtype, rtol, atol",
    (
        (torch.float32, 1e-5, 1e-5),
        (torch.float16, 1e-3, 1e-3),
        (torch.bfloat16, 1e-2, 1e-2),
        (torch.float64, 1e-8, 1e-8),
        (torch.bool, 0, 0),
        (torch.int32, 0, 0),
        (torch.int64, 0, 0),
    ),
)
@pytest.mark.parametrize("noncontiguous", (False, True))
@pytest.mark.parametrize(
    "shape, kernel_size, dilation, padding, stride",
    (
        ((2, 3, 8, 8), (3, 3), 1, 0, 1),
        ((1, 4, 10, 12), (5, 3), 1, 1, (2, 1)),
        ((2, 2, 16, 16), (4, 4), 1, 0, (4, 4)),
        ((3, 6, 7, 9), (3, 2), 1, 0, (1, 1)),
        ((1, 8, 9, 11), (2, 3), 1, 1, (1, 2)),
        ((2, 5, 12, 6), (3, 3), 2, (2, 1), (2, 1)),
    ),
)
def test_im2col_matches_torch(
    shape,
    kernel_size,
    dilation,
    padding,
    stride,
    noncontiguous,
    dtype,
    rtol,
    atol,
):
    input = _make_input(shape, dtype, noncontiguous=noncontiguous)

    output = ntops.torch.im2col(
        input,
        kernel_size=kernel_size,
        dilation=dilation,
        padding=padding,
        stride=stride,
    )
    reference = _reference_im2col(
        input,
        kernel_size=kernel_size,
        dilation=dilation,
        padding=padding,
        stride=stride,
    )

    if dtype == torch.bool or not dtype.is_floating_point:
        assert torch.equal(output, reference)
    else:
        assert torch.allclose(output, reference, rtol=rtol, atol=atol)


@skip_if_cuda_not_available
def test_im2col_rejects_empty_output_shape():
    input = torch.randn((1, 1, 2, 2), device="cuda")

    with pytest.raises(RuntimeError):
        ntops.torch.im2col(input, kernel_size=(3, 3), padding=0, stride=1)
