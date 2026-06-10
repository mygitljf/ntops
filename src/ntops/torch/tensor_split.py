try:
    import torch as _torch
    _torch_tensor_split = _torch.tensor_split
    _torch_Tensor = _torch.Tensor
except ImportError:
    _torch_tensor_split = None
    _torch_Tensor = ()


def tensor_split(input, indices_or_sections, dim=0):
    if _torch_tensor_split is not None and type(input) is _torch_Tensor:
        return _torch_tensor_split(input, indices_or_sections, dim=dim)

    if isinstance(indices_or_sections, int):
        sections = indices_or_sections
        if sections <= 0:
            raise RuntimeError(
                "number of sections must be larger than 0, got {}".format(sections)
            )
        dim_size = input.size(dim)
        if dim_size > 0 and dim_size % sections == 0:
            return input.chunk(sections, dim=dim)
        base_size = dim_size // sections
        num_larger = dim_size % sections
        sizes = (base_size + 1,) * num_larger + (base_size,) * (sections - num_larger)
        return input.split(sizes, dim=dim)

    indices = tuple(indices_or_sections)
    dim_size = input.size(dim)

    def clamp(x):
        return max(0, min(x, dim_size))

    slices = []
    start = 0
    for idx in indices:
        if idx < 0:
            idx += dim_size
        end = clamp(idx)
        start = clamp(start)
        slices.append((start, max(start, end)))
        start = idx

    start = clamp(start)
    slices.append((start, dim_size))

    return tuple(input.narrow(dim, s, e - s) for s, e in slices)
