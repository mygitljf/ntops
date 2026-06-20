import pytest
import torch
import torch.nn.functional as F

import ntops
from tests.skippers import skip_if_cuda_not_available


@skip_if_cuda_not_available
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("shape", [(8, 5), (4, 10), (2, 3, 6)])
def test_gumbel_softmax_soft_simplex(shape, dtype):
    logits = torch.randn(shape, dtype=dtype, device="cuda")
    output = ntops.torch.gumbel_softmax(logits, tau=1.0, hard=False)
    assert output.shape == logits.shape
    assert torch.all(output >= 0)
    sums = output.sum(dim=-1)
    assert torch.allclose(sums, torch.ones_like(sums), rtol=1e-4, atol=1e-4)


@skip_if_cuda_not_available
@pytest.mark.parametrize("dim", [-1, 0, 1])
def test_gumbel_softmax_hard_one_hot(dim):
    logits = torch.randn((6, 8, 4), dtype=torch.float32, device="cuda")
    output = ntops.torch.gumbel_softmax(logits, tau=1.0, hard=True, dim=dim)
    assert output.shape == logits.shape
    sums = output.sum(dim=dim)
    assert torch.allclose(sums, torch.ones_like(sums), rtol=1e-4, atol=1e-4)
    nonzero_per_slice = (output != 0).sum(dim=dim)
    assert torch.all(nonzero_per_slice == 1)


@skip_if_cuda_not_available
def test_gumbel_softmax_low_tau_statistics():
    torch.manual_seed(0)
    logits = torch.tensor([[2.0, 1.0, 0.1, -1.0]], device="cuda").repeat(4096, 1)
    counts = torch.zeros(4, device="cuda")
    for _ in range(20):
        counts += ntops.torch.gumbel_softmax(logits, tau=0.5, hard=True).sum(0)
    # The highest-logit class must be selected most frequently.
    assert counts.argmax().item() == 0
