import pytest
import torch
import torch.nn.functional as F

import ntops
from tests.skippers import skip_if_cuda_not_available


def _assert_close(output, reference, rtol=1e-3, atol=1e-3):
    assert output.shape == reference.shape
    assert output.dtype == reference.dtype
    if reference.dtype.is_floating_point:
        assert torch.allclose(output, reference, rtol=rtol, atol=atol, equal_nan=True)
    else:
        assert torch.equal(output, reference)


@skip_if_cuda_not_available
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32, torch.float64])
@pytest.mark.parametrize("shape", [(1,), (7,), (3, 5), (2, 3, 4), (8, 1)])
def test_mse_loss_none(shape, dtype):
    input = torch.randn(shape, dtype=dtype, device="cuda")
    target = torch.randn(shape, dtype=dtype, device="cuda")
    output = ntops.torch.mse_loss(input, target, reduction="none")
    reference = F.mse_loss(input, target, reduction="none")
    _assert_close(output, reference)


@skip_if_cuda_not_available
@pytest.mark.parametrize("reduction", ["mean", "sum"])
@pytest.mark.parametrize("dtype", [torch.float16, torch.float32, torch.float64])
def test_mse_loss_reduction(reduction, dtype):
    input = torch.randn((128, 256), dtype=dtype, device="cuda")
    target = torch.randn((128, 256), dtype=dtype, device="cuda")
    output = ntops.torch.mse_loss(input, target, reduction=reduction)
    reference = F.mse_loss(input, target, reduction=reduction)
    assert torch.allclose(output, reference, rtol=2e-3, atol=2e-3)


@skip_if_cuda_not_available
def test_mse_loss_non_contiguous():
    input = torch.randn((5, 7), dtype=torch.float32, device="cuda").t()
    target = torch.randn((5, 7), dtype=torch.float32, device="cuda").t()
    output = ntops.torch.mse_loss(input, target, reduction="none")
    reference = F.mse_loss(input, target, reduction="none")
    _assert_close(output, reference, rtol=1e-4, atol=1e-4)


@skip_if_cuda_not_available
def test_mse_loss_broadcast():
    input = torch.randn((4, 1), dtype=torch.float32, device="cuda")
    target = torch.randn((1, 6), dtype=torch.float32, device="cuda")
    output = ntops.torch.mse_loss(input, target, reduction="none")
    reference = F.mse_loss(input, target, reduction="none")
    _assert_close(output, reference, rtol=1e-4, atol=1e-4)
