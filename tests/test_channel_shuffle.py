import pytest
import torch

import ntops
from tests.skippers import skip_if_cuda_not_available


def _make_input(shape, dtype):
    if dtype == torch.bool:
        return torch.randint(0, 2, shape, dtype=torch.bool, device="cuda")
    if dtype.is_floating_point:
        return torch.randn(shape, dtype=dtype, device="cuda")
    return torch.randint(-100, 100, shape, dtype=dtype, device="cuda")


def _assert_equal(output, reference):
    assert output.shape == reference.shape
    assert output.dtype == reference.dtype
    if reference.dtype.is_floating_point:
        assert torch.allclose(output, reference, rtol=2e-3, atol=2e-3, equal_nan=True)
    else:
        assert torch.equal(output, reference)


def _reference_channel_shuffle(input, groups):
    channels = input.size(1)
    return (
        input.view(input.size(0), groups, channels // groups, *input.shape[2:])
        .transpose(1, 2)
        .reshape(input.shape)
    )


@skip_if_cuda_not_available
@pytest.mark.parametrize(
    "dtype",
    [
        torch.float32,
        torch.float16,
        torch.float64,
        torch.bfloat16,
        torch.int32,
        torch.int64,
        torch.bool,
    ],
)
@pytest.mark.parametrize(
    "shape, groups",
    [
        ((2, 4, 3), 2),
        ((2, 4, 3, 5), 2),
        ((2, 8, 4, 4), 4),
        ((2, 3, 6, 3, 5), 3),
    ],
)
def test_channel_shuffle_basic(dtype, shape, groups):
    input = _make_input(shape, dtype)
    output = ntops.torch.channel_shuffle(input, groups)
    reference = torch.channel_shuffle(input, groups)
    _assert_equal(output, reference)


@skip_if_cuda_not_available
@pytest.mark.parametrize("groups", [1, 4])
def test_channel_shuffle_identity_groups(groups):
    input = torch.randn((2, 4, 3, 5), dtype=torch.float32, device="cuda")
    output = ntops.torch.channel_shuffle(input, groups)
    reference = torch.channel_shuffle(input, groups)
    _assert_equal(output, reference)


@skip_if_cuda_not_available
def test_channel_shuffle_noncontiguous_transposed_input():
    input = torch.randn((2, 5, 4, 3), dtype=torch.float32, device="cuda").transpose(2, 3)
    assert not input.is_contiguous()
    output = ntops.torch.channel_shuffle(input, 5)
    reference = _reference_channel_shuffle(input, 5)
    _assert_equal(output, reference)


@skip_if_cuda_not_available
def test_channel_shuffle_noncontiguous_permuted_input():
    input = torch.randn((2, 3, 4, 5), dtype=torch.float32, device="cuda").permute(0, 2, 1, 3)
    assert not input.is_contiguous()
    output = ntops.torch.channel_shuffle(input, 2)
    reference = _reference_channel_shuffle(input, 2)
    _assert_equal(output, reference)


@skip_if_cuda_not_available
def test_channel_shuffle_channels_last_input():
    input = torch.randn((2, 4, 3, 5), dtype=torch.float32, device="cuda")
    input = input.contiguous(memory_format=torch.channels_last)
    assert not input.is_contiguous()
    output = ntops.torch.channel_shuffle(input, 2)
    reference = torch.channel_shuffle(input, 2)
    _assert_equal(output, reference)


@skip_if_cuda_not_available
def test_channel_shuffle_zero_batch():
    input = torch.randn((0, 4, 3, 5), dtype=torch.float32, device="cuda")
    output = ntops.torch.channel_shuffle(input, 2)
    reference = torch.channel_shuffle(input, 2)
    _assert_equal(output, reference)


@skip_if_cuda_not_available
def test_channel_shuffle_out():
    input = torch.randn((2, 4, 3, 5), dtype=torch.float32, device="cuda")
    out = torch.empty_like(input)
    result = ntops.torch.channel_shuffle(input, 2, out=out)
    reference = torch.channel_shuffle(input, 2)
    assert result is out
    _assert_equal(out, reference)


@skip_if_cuda_not_available
@pytest.mark.parametrize("shape", [(4, 3), (4,), ()])
def test_channel_shuffle_rejects_low_rank(shape):
    input = _make_input(shape, torch.float32)
    with pytest.raises(RuntimeError):
        ntops.torch.channel_shuffle(input, 1)


@skip_if_cuda_not_available
@pytest.mark.parametrize("groups", [0, -1])
def test_channel_shuffle_rejects_nonpositive_groups(groups):
    input = torch.randn((2, 4, 3), dtype=torch.float32, device="cuda")
    with pytest.raises(RuntimeError):
        ntops.torch.channel_shuffle(input, groups)


@skip_if_cuda_not_available
@pytest.mark.parametrize("groups", [True, False, 2.0, "2"])
def test_channel_shuffle_rejects_non_integer_groups(groups):
    input = torch.randn((2, 4, 3), dtype=torch.float32, device="cuda")
    with pytest.raises(TypeError):
        ntops.torch.channel_shuffle(input, groups)


@skip_if_cuda_not_available
def test_channel_shuffle_rejects_nondivisible_channels():
    input = torch.randn((2, 5, 3), dtype=torch.float32, device="cuda")
    with pytest.raises(RuntimeError):
        ntops.torch.channel_shuffle(input, 2)
