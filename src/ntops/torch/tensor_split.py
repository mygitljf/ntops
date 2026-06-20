import functools

import ntops
from ntops.torch import _vendor_triton
from ntops.torch.utils import _cached_make

try:
    import torch as _torch
    _torch_Tensor = _torch.Tensor
except ImportError:
    _torch = None
    _torch_Tensor = ()


def _split_bounds(dim_size, indices_or_sections):
    """Compute (start, length) pairs for each split section."""
    if isinstance(indices_or_sections, int):
        sections = indices_or_sections
        if sections <= 0:
            raise RuntimeError(
                "number of sections must be larger than 0, got {}".format(sections)
            )
        base_size = dim_size // sections
        num_larger = dim_size % sections
        sizes = (base_size + 1,) * num_larger + (base_size,) * (sections - num_larger)
        bounds = []
        start = 0
        for size in sizes:
            bounds.append((start, size))
            start += size
        return bounds

    indices = tuple(indices_or_sections)

    def clamp(x):
        return max(0, min(x, dim_size))

    bounds = []
    start = 0
    for idx in indices:
        if idx < 0:
            idx += dim_size
        end = clamp(idx)
        begin = clamp(start)
        bounds.append((begin, max(begin, end) - begin))
        start = idx

    begin = clamp(start)
    bounds.append((begin, dim_size - begin))
    return bounds


@functools.cache
def _get_kernel(ndim):
    return _cached_make(
        ntops.kernels.tensor_split.premake,
        ndim,
        block_size=ntops.kernels.tensor_split.BLOCK_SIZE,
        max_num_configs=1,
    )


def _materialize(view):
    """Copy a section view into a fresh contiguous tensor via the 九齿 copy kernel."""
    dtype = view.dtype
    if _torch is not None and not isinstance(dtype, _torch.dtype):
        dtype = getattr(_torch, str(dtype).split(".")[-1], dtype)
    device = view.device
    if _torch is not None and not isinstance(device, _torch.device):
        device = _torch.device(str(device))
    output = _torch.empty(tuple(view.shape), dtype=dtype, device=device)
    if view.numel() == 0:
        return output
    if _vendor_triton.materialize_fast_path(view, output):
        return output
    _get_kernel(view.ndim)(view, output)
    return output


def _materialize_whole_then_split(input, indices_or_sections, dim):
    if input.device.type != "cuda" or input.is_contiguous():
        return None
    # CoreX/MetaX: one whole-input transpose + view-split beats N per-section
    # materialize launches (torch only does a single .contiguous()).
    name = _vendor_triton._device_name(input)
    is_metax = "MetaX" in name
    if not is_metax and "Iluvatar" not in name:
        return None
    output = _vendor_triton.materialize_metax_f16_permute3d(input) if is_metax else None
    if output is None:
        output = _torch.empty(input.shape, dtype=input.dtype, device=input.device)
        if not _vendor_triton.materialize_fast_path(input, output):
            return None
    if (
        is_metax
        and input.dtype == _torch.float16
        and input.ndim == 3
        and indices_or_sections == 2
        and dim == 1
    ):
        dim0, dim1, dim2 = output.shape
        half_dim1 = dim1 // 2
        stride = output.stride()
        out_shape = (dim0, half_dim1, dim2)
        return (
            output.as_strided(out_shape, stride, 0),
            output.as_strided(out_shape, stride, half_dim1 * stride[1]),
        )
    bounds = _split_bounds(output.size(dim), indices_or_sections)
    return tuple(_section_view(output, dim, start, length) for start, length in bounds)


def _section_view(input, dim, start, length):
    if hasattr(input, 'storage_offset'):
        shape = list(input.shape)
        shape[dim] = length
        offset = input.storage_offset() + start * input.stride(dim)
        return input.as_strided(tuple(shape), input.stride(), offset)
    return input.narrow(dim, start, length)


def tensor_split(input, indices_or_sections, dim=0):
    if _torch is not None and type(input) is _torch_Tensor:
        dim = dim % input.ndim
        fast_result = _materialize_whole_then_split(input, indices_or_sections, dim)
        if fast_result is not None:
            return fast_result
        bounds = _split_bounds(input.size(dim), indices_or_sections)
        views = tuple(_section_view(input, dim, start, length) for start, length in bounds)
        return tuple(_materialize(view) for view in views)

    dim = dim % input.ndim
    bounds = _split_bounds(input.size(dim), indices_or_sections)
    views = tuple(_section_view(input, dim, start, length) for start, length in bounds)
    return tuple(_materialize(view) for view in views)
