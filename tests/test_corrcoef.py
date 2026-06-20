import pytest
import torch

import ntops
from tests.skippers import skip_if_cuda_not_available


@skip_if_cuda_not_available
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("shape", [(5, 100), (3, 50), (10, 200), (2, 8)])
def test_corrcoef_2d(shape, dtype):
    input = torch.randn(shape, dtype=dtype, device="cuda")
    output = ntops.torch.corrcoef(input)
    # CPU is the exact ground truth (CoreX has no f64 gemm, so an on-device reference
    # would crash there for double).
    reference = torch.corrcoef(input.cpu())
    assert output.shape == reference.shape
    assert output.dtype == reference.dtype
    assert torch.allclose(output.cpu(), reference, rtol=1e-4, atol=1e-4, equal_nan=True)


@skip_if_cuda_not_available
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_corrcoef_1d(dtype):
    input = torch.randn(100, dtype=dtype, device="cuda")
    output = ntops.torch.corrcoef(input)
    reference = torch.corrcoef(input.cpu())
    assert output.shape == reference.shape
    assert torch.allclose(output.cpu(), reference, rtol=1e-4, atol=1e-4, equal_nan=True)


@skip_if_cuda_not_available
def test_corrcoef_non_contiguous():
    input = torch.randn((100, 5), dtype=torch.float32, device="cuda").t()
    output = ntops.torch.corrcoef(input)
    reference = torch.corrcoef(input.cpu())
    assert torch.allclose(output.cpu(), reference, rtol=1e-4, atol=1e-4, equal_nan=True)
