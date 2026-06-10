import functools
import logging

import torch

import ntops
from ntops.torch import _vendor_triton
from ntops.torch.utils import _cached_make, _prepare_out

_logger = logging.getLogger(__name__)


def _validate_channel_shuffle(input, groups):
    if not isinstance(groups, int) or isinstance(groups, bool):
        raise TypeError(
            "channel_shuffle(): argument 'groups' (position 2) must be int, "
            f"not {type(groups).__name__}"
        )
    if input.ndim <= 2:
        raise RuntimeError(
            "channel_shuffle expects input with > 2 dims, but got input with "
            f"sizes {list(input.shape)}"
        )
    if groups <= 0:
        raise RuntimeError(
            "Number of groups to divide channels in must be positive. "
            f"Value of groups:{groups}"
        )

    channels = input.size(1)
    if channels % groups != 0:
        raise RuntimeError(
            "Number of channels must be divisible by groups. "
            f"Got {channels} channels and {groups} groups."
        )


@functools.cache
def _get_kernel(ndim, groups):
    return _cached_make(
        ntops.kernels.channel_shuffle.premake,
        ndim,
        groups=groups,
        block_size=ntops.kernels.channel_shuffle.BLOCK_SIZE,
        num_warps=1,
        max_num_configs=1,
    )


def channel_shuffle(input, groups, *, out=None):
    _validate_channel_shuffle(input, groups)

    if out is None and input.is_contiguous() and groups in (1, input.size(1)):
        return input

    if input.is_contiguous() and input.dtype == torch.float64 and _vendor_triton.is_iluvatar_device(input):
        channels = input.size(1)
        result = (
            input.view(input.size(0), groups, channels // groups, *input.shape[2:])
            .transpose(1, 2)
            .reshape(input.shape)
        )
        if out is not None:
            out.copy_(result)
            return out
        return result

    output = _prepare_out(out, input.shape, input.dtype, input.device)
    if input.numel() == 0:
        return output
    try:
        if _vendor_triton.channel_shuffle_fast_path(input, output, groups):
            return output
    except Exception:
        pass
    try:
        _get_kernel(input.ndim, groups)(input, output)
    except FileNotFoundError:
        _logger.warning(
            "channel_shuffle: kernel compilation failed, "
            "falling back to view-based implementation"
        )
        channels = input.size(1)
        result = (
            input.view(input.size(0), groups, channels // groups, *input.shape[2:])
            .transpose(1, 2)
            .reshape(input.shape)
        )
        output.copy_(result)
    return output
