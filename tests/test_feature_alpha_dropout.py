import pytest
import torch
import torch.nn.functional as F

import ntops
from tests.skippers import skip_if_cuda_not_available


@skip_if_cuda_not_available
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("shape", [(2, 4, 8, 8), (8, 16, 4, 4), (4, 8, 16)])
def test_feature_alpha_dropout_eval_identity(shape, dtype):
    input = torch.randn(shape, dtype=dtype, device="cuda")
    output = ntops.torch.feature_alpha_dropout(input, p=0.5, training=False)
    assert torch.equal(output, input)


@skip_if_cuda_not_available
def test_feature_alpha_dropout_p0_identity():
    input = torch.randn((2, 4, 8, 8), dtype=torch.float32, device="cuda")
    output = ntops.torch.feature_alpha_dropout(input, p=0.0, training=True)
    assert torch.equal(output, input)


@skip_if_cuda_not_available
def test_feature_alpha_dropout_shape_preserved():
    input = torch.randn((4, 64, 8, 8), dtype=torch.float32, device="cuda")
    output = ntops.torch.feature_alpha_dropout(input, p=0.5, training=True)
    assert output.shape == input.shape
    assert output.dtype == input.dtype


@skip_if_cuda_not_available
def test_feature_alpha_dropout_statistics():
    input = torch.randn((256, 256, 4, 4), dtype=torch.float32, device="cuda")
    p = 0.5
    nt = ntops.torch.feature_alpha_dropout(input, p=p, training=True)
    ref = F.feature_alpha_dropout(input, p=p, training=True)
    # Match torch mean/std preservation property within statistical tolerance.
    assert abs(nt.mean().item() - ref.mean().item()) < 0.05
    assert abs(nt.std().item() - ref.std().item()) < 0.1
