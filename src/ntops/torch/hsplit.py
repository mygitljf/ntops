import torch

import ntops
from ntops.torch.utils import _cached_make


def _to_nt(torch_dtype):
    import ninetoothed

    mapping = {
        torch.float16: ninetoothed.float16,
        torch.bfloat16: ninetoothed.bfloat16,
        torch.float32: ninetoothed.float32,
        torch.float64: ninetoothed.float64,
        torch.int8: ninetoothed.int8,
        torch.int16: ninetoothed.int16,
        torch.int32: ninetoothed.int32,
        torch.int64: ninetoothed.int64,
        torch.uint8: ninetoothed.uint8,
    }
    return mapping.get(torch_dtype)


def _segment_bounds(size, indices_or_sections):
    # torch.hsplit segmentation semantics (tensor_split along the horizontal dim).
    if isinstance(indices_or_sections, int):
        sections = indices_or_sections
        if sections <= 0:
            raise RuntimeError(
                f"number of sections must be larger than 0, got {sections}"
            )
        if size % sections != 0:
            raise RuntimeError(
                "torch.hsplit attempted to split along the horizontal dimension, "
                f"but the size of the dimension {size} is not divisible by the split "
                f"size {sections}!"
            )
        step = size // sections
        return [(i * step, (i + 1) * step) for i in range(sections)]

    indices = [i + size if i < 0 else i for i in indices_or_sections]
    bounds = []
    prev = 0
    for index in indices:
        cur = max(0, min(index, size))
        bounds.append((min(prev, size), cur if cur >= prev else prev))
        prev = cur
    bounds.append((min(prev, size), size if size >= prev else prev))
    return bounds


def _materialize(view):
    # C 类: torch.hsplit returns zero-copy views; here each segment is materialized
    # with the ntops.kernels.hsplit copy kernel.
    output = torch.empty(view.shape, dtype=view.dtype, device=view.device)
    nt_dtype = _to_nt(view.dtype)
    if view.numel() == 0:
        return output
    if nt_dtype is None:
        raise NotImplementedError(f"hsplit kernel does not support {view.dtype}")

    kernel = _cached_make(
        ntops.kernels.hsplit.premake,
        view.ndim,
        dtype=nt_dtype,
        block_size=ntops.kernels.hsplit.BLOCK_SIZE,
        num_warps=4,
        max_num_configs=1,
    )
    kernel(view, output)
    return output


def _section_view(input, dim, start, length):
    return input.narrow(dim, start, length)


def hsplit(input, indices_or_sections):
    if input.ndim < 1:
        raise RuntimeError(
            "torch.hsplit requires a tensor with at least 1 dimension, but got a "
            f"tensor with {input.ndim} dimensions!"
        )

    dim = 0 if input.ndim == 1 else 1
    size = input.shape[dim]

    outputs = []
    for lo, hi in _segment_bounds(size, indices_or_sections):
        view = _section_view(input, dim, lo, hi - lo)
        outputs.append(_materialize(view))

    return tuple(outputs)
