import pytest
import torch

import ntops
from tests.skippers import skip_if_cuda_not_available


@skip_if_cuda_not_available
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("shape", [(3, 3), (5, 5), (4, 6, 6), (2, 3, 8, 8)])
def test_slogdet(shape, dtype):
    input = torch.randn(shape, dtype=dtype, device="cuda")
    sign, logabsdet = ntops.torch.slogdet(input)
    # CPU is the exact ground truth (some vendor GPUs lack correct f64 linalg).
    ref_sign, ref_logabsdet = torch.linalg.slogdet(input.cpu())
    assert sign.shape == ref_sign.shape
    assert logabsdet.shape == ref_logabsdet.shape
    assert torch.allclose(sign.cpu(), ref_sign, rtol=1e-4, atol=1e-4)
    assert torch.allclose(logabsdet.cpu(), ref_logabsdet, rtol=1e-4, atol=1e-4, equal_nan=True)


@skip_if_cuda_not_available
def test_slogdet_singular():
    input = torch.zeros((4, 4), dtype=torch.float64, device="cuda")
    sign, logabsdet = ntops.torch.slogdet(input)
    ref_sign, ref_logabsdet = torch.linalg.slogdet(input.cpu())
    assert torch.allclose(sign.cpu(), ref_sign)
    assert torch.allclose(logabsdet.cpu(), ref_logabsdet, equal_nan=True)
