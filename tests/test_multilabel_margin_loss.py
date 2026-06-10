import pytest
import torch
import torch.nn.functional as F

import ntops
from tests.skippers import skip_if_cuda_not_available


def _assert_close(output, reference, rtol=2e-3, atol=2e-3):
    assert output.shape == reference.shape
    assert output.dtype == reference.dtype
    assert torch.allclose(output, reference, rtol=rtol, atol=atol, equal_nan=True)


def _make_target(batch, classes):
    target = torch.full((batch, classes), -1, dtype=torch.long, device="cuda")
    for row in range(batch):
        target[row, 0] = row % classes
        if classes > 2:
            target[row, 1] = (row + 2) % classes
    return target


@skip_if_cuda_not_available
@pytest.mark.parametrize("dtype", [torch.float16, torch.float32, torch.float64])
@pytest.mark.parametrize("shape", [(1, 4), (2, 4), (3, 5), (4, 8)])
@pytest.mark.parametrize("reduction", ["none", "sum", "mean"])
def test_multilabel_margin_loss_shapes(dtype, shape, reduction):
    input = torch.randn(shape, dtype=dtype, device="cuda")
    target = _make_target(shape[0], shape[1])
    output = ntops.torch.multilabel_margin_loss(input, target, reduction=reduction)
    reference = F.multilabel_margin_loss(input, target, reduction=reduction)
    _assert_close(output, reference)


@skip_if_cuda_not_available
@pytest.mark.parametrize("reduction", ["none", "sum", "mean"])
def test_multilabel_margin_loss_1d(reduction):
    input = torch.randn((5,), dtype=torch.float32, device="cuda")
    target = torch.tensor([0, 2, -1, -1, -1], dtype=torch.long, device="cuda")
    output = ntops.torch.multilabel_margin_loss(input, target, reduction=reduction)
    reference = F.multilabel_margin_loss(input, target, reduction=reduction)
    _assert_close(output, reference)


@skip_if_cuda_not_available
def test_multilabel_margin_loss_non_contiguous():
    input = torch.randn((5, 3), dtype=torch.float32, device="cuda").t()
    target = _make_target(5, 3).t()
    output = ntops.torch.multilabel_margin_loss(input, target, reduction="none")
    reference = F.multilabel_margin_loss(input, target, reduction="none")
    _assert_close(output, reference)


@skip_if_cuda_not_available
def test_multilabel_margin_loss_integer_promotes():
    input = torch.randint(-2, 4, (2, 4), dtype=torch.int32, device="cuda")
    target = _make_target(2, 4)
    output = ntops.torch.multilabel_margin_loss(input, target, reduction="mean")
    reference = F.multilabel_margin_loss(input.to(torch.float32), target, reduction="mean")
    _assert_close(output, reference)


@skip_if_cuda_not_available
def test_multilabel_margin_loss_legacy_reduction():
    input = torch.randn((2, 4), dtype=torch.float32, device="cuda")
    target = _make_target(2, 4)
    output = ntops.torch.multilabel_margin_loss(input, target, size_average=False, reduce=True)
    reference = F.multilabel_margin_loss(input, target, size_average=False, reduce=True)
    _assert_close(output, reference)


@skip_if_cuda_not_available
def test_multilabel_margin_loss_rejects_target_dtype():
    input = torch.randn((2, 4), dtype=torch.float32, device="cuda")
    target = torch.zeros((2, 4), dtype=torch.int32, device="cuda")
    with pytest.raises(RuntimeError):
        ntops.torch.multilabel_margin_loss(input, target)


@skip_if_cuda_not_available
@pytest.mark.parametrize("reduction", ["none", "sum", "mean"])
def test_multilabel_margin_loss_prefix_stop(reduction):
    input = torch.tensor([[0.1, 0.2, 0.3, 0.4]], dtype=torch.float32, device="cuda")
    target = torch.tensor([[3, -1, 1, -1]], dtype=torch.long, device="cuda")
    output = ntops.torch.multilabel_margin_loss(input, target, reduction=reduction)
    reference = F.multilabel_margin_loss(input, target, reduction=reduction)
    _assert_close(output, reference)


@skip_if_cuda_not_available
@pytest.mark.parametrize("reduction", ["none", "sum", "mean"])
def test_multilabel_margin_loss_duplicate_targets(reduction):
    input = torch.randn((3, 6), dtype=torch.float32, device="cuda")
    target = torch.tensor(
        [[1, 1, -1, -1, -1, -1], [0, 3, 3, -1, -1, -1], [2, 2, 2, -1, -1, -1]],
        dtype=torch.long,
        device="cuda",
    )
    output = ntops.torch.multilabel_margin_loss(input, target, reduction=reduction)
    reference = F.multilabel_margin_loss(input, target, reduction=reduction)
    _assert_close(output, reference)


@skip_if_cuda_not_available
@pytest.mark.parametrize("dtype", [torch.float16, torch.float32, torch.float64])
def test_multilabel_margin_loss_all_classes_targeted(dtype):
    input = torch.randn((4, 8), dtype=dtype, device="cuda")
    target = torch.arange(8, dtype=torch.long, device="cuda").expand(4, 8).contiguous()
    output = ntops.torch.multilabel_margin_loss(input, target, reduction="none")
    reference = F.multilabel_margin_loss(input, target, reduction="none")
    _assert_close(output, reference)
