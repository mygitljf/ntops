import pytest
import torch
import torch.nn.functional as F

import ntops
from tests.skippers import skip_if_cuda_not_available


def _is_iluvatar():
    if not torch.cuda.is_available():
        return False
    return "Iluvatar" in torch.cuda.get_device_name(torch.cuda.current_device())


def _assert_pool(output, indices, reference, reference_indices, rtol=2e-3, atol=2e-3, check_indices=True):
    assert output.shape == reference.shape
    assert output.dtype == reference.dtype
    assert indices.shape == reference_indices.shape
    assert indices.dtype == reference_indices.dtype
    assert torch.allclose(output, reference, rtol=rtol, atol=atol, equal_nan=True)
    if check_indices:
        assert torch.equal(indices, reference_indices)


@skip_if_cuda_not_available
@pytest.mark.parametrize("dtype", [torch.float16, torch.float32, torch.float64])
@pytest.mark.parametrize(
    "shape, kernel_size, output_size",
    [((1, 1, 4, 4), 2, (2, 2)), ((2, 3, 6, 5), (2, 2), (3, 3)), ((3, 5, 5), 2, (2, 2))],
)
def test_fractional_max_pool2d_shapes(dtype, shape, kernel_size, output_size):
    input = torch.randn(shape, dtype=dtype, device="cuda")
    n_batch = 1 if input.ndim == 3 else input.shape[0]
    random_samples = torch.full((n_batch, input.shape[-3], 2), 0.5, dtype=dtype, device="cuda")
    output, indices = ntops.torch.fractional_max_pool2d(
        input,
        kernel_size,
        output_size=output_size,
        _random_samples=random_samples,
        return_indices=True,
    )
    reference, reference_indices = F.fractional_max_pool2d(
        input,
        kernel_size,
        output_size=output_size,
        _random_samples=random_samples,
        return_indices=True,
    )
    _assert_pool(output, indices, reference, reference_indices, check_indices=not (_is_iluvatar() and dtype == torch.float64))


@skip_if_cuda_not_available
def test_fractional_max_pool2d_output_ratio():
    input = torch.randn((1, 2, 8, 6), dtype=torch.float32, device="cuda")
    random_samples = torch.full((1, 2, 2), 0.25, dtype=torch.float32, device="cuda")
    output, indices = ntops.torch.fractional_max_pool2d(
        input, 2, output_ratio=(0.5, 0.5), _random_samples=random_samples, return_indices=True
    )
    reference, reference_indices = F.fractional_max_pool2d(
        input, 2, output_ratio=(0.5, 0.5), _random_samples=random_samples, return_indices=True
    )
    _assert_pool(output, indices, reference, reference_indices)


@skip_if_cuda_not_available
def test_fractional_max_pool2d_non_contiguous():
    input = torch.randn((1, 2, 5, 6), dtype=torch.float32, device="cuda").transpose(2, 3)
    random_samples = torch.full((1, 2, 2), 0.75, dtype=torch.float32, device="cuda")
    output, indices = ntops.torch.fractional_max_pool2d(
        input, 2, output_size=(2, 2), _random_samples=random_samples, return_indices=True
    )
    reference, reference_indices = F.fractional_max_pool2d(
        input, 2, output_size=(2, 2), _random_samples=random_samples, return_indices=True
    )
    _assert_pool(output, indices, reference, reference_indices)


@skip_if_cuda_not_available
def test_fractional_max_pool2d_return_output_only():
    input = torch.randn((1, 1, 4, 4), dtype=torch.float32, device="cuda")
    random_samples = torch.full((1, 1, 2), 0.5, dtype=torch.float32, device="cuda")
    output = ntops.torch.fractional_max_pool2d(
        input, 2, output_size=(2, 2), _random_samples=random_samples
    )
    reference = F.fractional_max_pool2d(input, 2, output_size=(2, 2), _random_samples=random_samples)
    assert torch.allclose(output, reference, rtol=2e-3, atol=2e-3)


@skip_if_cuda_not_available
def test_fractional_max_pool2d_rejects_bad_random_samples():
    input = torch.randn((1, 1, 4, 4), dtype=torch.float32, device="cuda")
    random_samples = torch.empty((1, 2, 2), dtype=torch.float32, device="cuda")
    with pytest.raises(RuntimeError):
        ntops.torch.fractional_max_pool2d(input, 2, output_size=(2, 2), _random_samples=random_samples)
