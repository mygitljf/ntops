import functools

from ninetoothed import Tensor


BLOCK_SIZE = 1024


def arrangement(input, output, flip_mask=None, block_size=BLOCK_SIZE):
    # Reverse the flipped dims of the output at the meta level (negative-step
    # slices), then identity-copy: output[..., ::-1, ...] = input. This handles
    # any combination of axes (trailing or not) and arbitrarily large rows,
    # because the trailing dim is tiled in `block_size` chunks rather than
    # required to fit a single tile.
    ndim = input.ndim
    rev = tuple(
        slice(None, None, -1) if flip_mask[d] else slice(None) for d in range(ndim)
    )
    output_rev = output[rev]

    inner = tuple(1 for _ in range(ndim - 1)) + (block_size,)
    input_arranged = input.tile(inner)
    output_arranged = output_rev.tile(inner)

    lead = tuple(range(ndim - 1))
    if lead:
        input_arranged.dtype = input_arranged.dtype.squeeze(lead)
        output_arranged.dtype = output_arranged.dtype.squeeze(lead)

    return input_arranged, output_arranged


def application(input, output):
    output = input  # noqa: F841


def premake(ndim, flip_mask, dtype=None, block_size=BLOCK_SIZE):
    arrangement_ = functools.partial(
        arrangement, flip_mask=tuple(flip_mask), block_size=block_size
    )

    tensors = (
        Tensor(ndim, dtype=dtype, other=0),
        Tensor(ndim, dtype=dtype),
    )

    return arrangement_, application, tensors
