import pytest
import torch
import torch.nn.functional as F

import ntops
from tests.skippers import skip_if_cuda_not_available


def _assert_close(output, reference, rtol=1e-3, atol=1e-3):
    assert output.shape == reference.shape
    assert output.dtype == reference.dtype
    assert torch.allclose(output, reference, rtol=rtol, atol=atol, equal_nan=True)


@skip_if_cuda_not_available
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
@pytest.mark.parametrize("shape", [(8, 5), (7,), (4, 3, 6), (16, 1)])
def test_kl_div_none(shape, dtype):
    input = torch.randn(shape, dtype=dtype, device="cuda").log_softmax(-1)
    target = torch.randn(shape, dtype=dtype, device="cuda").softmax(-1)
    output = ntops.torch.kl_div(input, target, reduction="none")
    reference = F.kl_div(input, target, reduction="none")
    tol = 2e-2 if dtype == torch.bfloat16 else 2e-3
    _assert_close(output, reference, rtol=tol, atol=tol)


@skip_if_cuda_not_available
@pytest.mark.parametrize("reduction", ["mean", "sum", "batchmean"])
@pytest.mark.parametrize("dtype", [torch.float16, torch.float32])
def test_kl_div_reduction(reduction, dtype):
    input = torch.randn((64, 32), dtype=dtype, device="cuda").log_softmax(-1)
    target = torch.randn((64, 32), dtype=dtype, device="cuda").softmax(-1)
    output = ntops.torch.kl_div(input, target, reduction=reduction)
    reference = F.kl_div(input, target, reduction=reduction)
    assert torch.allclose(output, reference, rtol=2e-3, atol=2e-3)


@skip_if_cuda_not_available
def test_kl_div_log_target():
    input = torch.randn((32, 16), device="cuda").log_softmax(-1)
    target = torch.randn((32, 16), device="cuda").log_softmax(-1)
    output = ntops.torch.kl_div(input, target, reduction="none", log_target=True)
    reference = F.kl_div(input, target, reduction="none", log_target=True)
    _assert_close(output, reference, rtol=1e-4, atol=1e-4)


@skip_if_cuda_not_available
def test_kl_div_non_contiguous():
    input = torch.randn((5, 7), device="cuda").log_softmax(-1).t()
    target = torch.randn((5, 7), device="cuda").softmax(-1).t()
    output = ntops.torch.kl_div(input, target, reduction="none")
    reference = F.kl_div(input, target, reduction="none")
    _assert_close(output, reference, rtol=1e-4, atol=1e-4)
