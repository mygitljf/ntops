import functools

import ninetoothed
from ninetoothed import Tensor


BLOCK_SIZE = 2048


def arrangement(input, output, groups=None, block_size=None):
    if block_size is None:
        block_size = ninetoothed.block_size()

    channel_dim = 1
    ndim = input.ndim
    channels_per_group = input.shape[channel_dim] // groups

    tile_shape = tuple(
        channels_per_group if dim == channel_dim else 1
        for dim in range(ndim)
    )
    input_arranged = input.tile(tile_shape)
    input_arranged = input_arranged.ravel()

    squeeze_dims = tuple(ndim + dim for dim in range(ndim) if dim != channel_dim)
    input_arranged = input_arranged.squeeze(squeeze_dims)

    perm = (
        tuple(range(channel_dim))
        + (ndim, channel_dim)
        + tuple(range(channel_dim + 1, ndim))
    )
    input_arranged = input_arranged.permute(perm)
    input_arranged = input_arranged.flatten(
        start_dim=channel_dim,
        end_dim=channel_dim + 2,
    )
    input_arranged = input_arranged.flatten().tile((block_size,))

    output_arranged = output.flatten().tile((block_size,))

    return input_arranged, output_arranged


def application(input, output):
    output = input  # noqa: F841


def premake(ndim, groups=None, dtype=None, block_size=BLOCK_SIZE):
    arrangement_ = functools.partial(
        arrangement,
        groups=groups,
        block_size=block_size,
    )
    tensors = (Tensor(ndim, dtype=dtype), Tensor(ndim, dtype=dtype))
    return arrangement_, application, tensors
