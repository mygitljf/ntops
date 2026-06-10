import functools

import ninetoothed.language as ntl
from ninetoothed import Tensor

from ntops.kernels.element_wise import arrangement


BLOCK_SIZE = 1024


def copy_application(input, output):
    output = input  # noqa: F841


def _atomic_add(src, output, target_offset, valid):
    ntl.atomic_add(
        output.source.data_ptr() + target_offset,
        src,
        sem="relaxed",
        mask=valid,
    )


def scatter_application_1d(index, src, dim, output):
    coord_0 = index
    valid = (src.offsets(0) < src.source.shape[0]) & (coord_0 >= 0) & (coord_0 < output.source.shape[0])
    _atomic_add(src, output, coord_0 * output.source.stride(0), valid)


def scatter_application_2d(index, src, dim, output):
    coord_0 = ntl.where(dim == 0, index, src.offsets(0))
    coord_1 = ntl.where(dim == 1, index, src.offsets(1))
    valid = (
        (src.offsets(0) < src.source.shape[0])
        & (src.offsets(1) < src.source.shape[1])
        & (coord_0 >= 0)
        & (coord_0 < output.source.shape[0])
        & (coord_1 >= 0)
        & (coord_1 < output.source.shape[1])
    )
    target_offset = coord_0 * output.source.stride(0) + coord_1 * output.source.stride(1)
    _atomic_add(src, output, target_offset, valid)


def scatter_application_3d(index, src, dim, output):
    coord_0 = ntl.where(dim == 0, index, src.offsets(0))
    coord_1 = ntl.where(dim == 1, index, src.offsets(1))
    coord_2 = ntl.where(dim == 2, index, src.offsets(2))
    valid = (
        (src.offsets(0) < src.source.shape[0])
        & (src.offsets(1) < src.source.shape[1])
        & (src.offsets(2) < src.source.shape[2])
        & (coord_0 >= 0)
        & (coord_0 < output.source.shape[0])
        & (coord_1 >= 0)
        & (coord_1 < output.source.shape[1])
        & (coord_2 >= 0)
        & (coord_2 < output.source.shape[2])
    )
    target_offset = (
        coord_0 * output.source.stride(0)
        + coord_1 * output.source.stride(1)
        + coord_2 * output.source.stride(2)
    )
    _atomic_add(src, output, target_offset, valid)


def scatter_application_4d(index, src, dim, output):
    coord_0 = ntl.where(dim == 0, index, src.offsets(0))
    coord_1 = ntl.where(dim == 1, index, src.offsets(1))
    coord_2 = ntl.where(dim == 2, index, src.offsets(2))
    coord_3 = ntl.where(dim == 3, index, src.offsets(3))
    valid = (
        (src.offsets(0) < src.source.shape[0])
        & (src.offsets(1) < src.source.shape[1])
        & (src.offsets(2) < src.source.shape[2])
        & (src.offsets(3) < src.source.shape[3])
        & (coord_0 >= 0)
        & (coord_0 < output.source.shape[0])
        & (coord_1 >= 0)
        & (coord_1 < output.source.shape[1])
        & (coord_2 >= 0)
        & (coord_2 < output.source.shape[2])
        & (coord_3 >= 0)
        & (coord_3 < output.source.shape[3])
    )
    target_offset = (
        coord_0 * output.source.stride(0)
        + coord_1 * output.source.stride(1)
        + coord_2 * output.source.stride(2)
        + coord_3 * output.source.stride(3)
    )
    _atomic_add(src, output, target_offset, valid)


def scatter_application_5d(index, src, dim, output):
    coord_0 = ntl.where(dim == 0, index, src.offsets(0))
    coord_1 = ntl.where(dim == 1, index, src.offsets(1))
    coord_2 = ntl.where(dim == 2, index, src.offsets(2))
    coord_3 = ntl.where(dim == 3, index, src.offsets(3))
    coord_4 = ntl.where(dim == 4, index, src.offsets(4))
    valid = (
        (src.offsets(0) < src.source.shape[0])
        & (src.offsets(1) < src.source.shape[1])
        & (src.offsets(2) < src.source.shape[2])
        & (src.offsets(3) < src.source.shape[3])
        & (src.offsets(4) < src.source.shape[4])
        & (coord_0 >= 0)
        & (coord_0 < output.source.shape[0])
        & (coord_1 >= 0)
        & (coord_1 < output.source.shape[1])
        & (coord_2 >= 0)
        & (coord_2 < output.source.shape[2])
        & (coord_3 >= 0)
        & (coord_3 < output.source.shape[3])
        & (coord_4 >= 0)
        & (coord_4 < output.source.shape[4])
    )
    target_offset = (
        coord_0 * output.source.stride(0)
        + coord_1 * output.source.stride(1)
        + coord_2 * output.source.stride(2)
        + coord_3 * output.source.stride(3)
        + coord_4 * output.source.stride(4)
    )
    _atomic_add(src, output, target_offset, valid)


_SCATTER_APPLICATIONS = {
    1: scatter_application_1d,
    2: scatter_application_2d,
    3: scatter_application_3d,
    4: scatter_application_4d,
    5: scatter_application_5d,
}


def premake_copy(ndim, dtype=None, block_size=BLOCK_SIZE):
    arrangement_ = functools.partial(arrangement, block_size=block_size)
    tensors = (Tensor(ndim, dtype=dtype), Tensor(ndim, dtype=dtype))
    return arrangement_, copy_application, tensors


def premake_scatter(ndim, dtype=None, block_size=BLOCK_SIZE):
    if ndim not in _SCATTER_APPLICATIONS:
        raise NotImplementedError("scatter_add supports tensors with 1 to 5 dimensions")
    arrangement_ = functools.partial(arrangement, block_size=block_size)
    tensors = (
        Tensor(ndim, dtype=None, other=0),
        Tensor(ndim, dtype=dtype, other=0),
        Tensor(0, dtype=None, constexpr=True),
        Tensor(ndim, dtype=dtype),
    )
    return arrangement_, _SCATTER_APPLICATIONS[ndim], tensors
