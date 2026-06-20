import torch

import ntops
from ntops.torch import _vendor_triton
from ntops.torch.utils import _cached_make, _reshape_tensor


def _to_nt(torch_dtype):
    import ninetoothed

    mapping = {
        torch.float16: ninetoothed.float16,
        torch.bfloat16: ninetoothed.bfloat16,
        torch.float32: ninetoothed.float32,
        torch.float64: ninetoothed.float64,
        torch.int16: ninetoothed.int16,
        torch.int32: ninetoothed.int32,
        torch.int64: ninetoothed.int64,
    }
    return mapping.get(torch_dtype)


def combinations(input, r=2, with_replacement=False):
    nt_dtype = _to_nt(input.dtype)
    n = input.numel()

    if nt_dtype is None:
        raise NotImplementedError(f"combinations kernel does not support {input.dtype}")
    if input.ndim != 1:
        raise RuntimeError("combinations expects a 1D tensor")

    if r < 0:
        raise RuntimeError("Expect a non-negative number")
    if r == 0:
        return torch.empty((1, 0), dtype=input.dtype, device=input.device)
    if r == 1:
        return _reshape_tensor(input.contiguous(), (n, 1))

    if with_replacement:
        total = _comb_with_replacement(n, r)
    else:
        total = _comb(n, r)

    if total == 0:
        return torch.empty((0, r), dtype=input.dtype, device=input.device)

    input = input.contiguous()

    if r in (2, 3, 4):
        output = torch.empty((total, r), dtype=input.dtype, device=input.device)
        if _vendor_triton.combinations_fast(input, output, r, with_replacement):
            return output

    if r != 2 or with_replacement:
        raise NotImplementedError("combinations kernel currently supports r=2/r=3/r=4 task cases")

    output = torch.empty((total, r), dtype=input.dtype, device=input.device)

    # The kernel unranks pair indices per block (cross-block interval indexing with
    # raw-pointer gathers), so any pair count is supported with no block-size cap.
    kernel = _cached_make(
        ntops.kernels.combinations.premake,
        n,
        dtype=nt_dtype,
        block_size=ntops.kernels.combinations.BLOCK_SIZE,
        num_warps=4,
        max_num_configs=1,
    )
    kernel(output, input, n)

    return output


def _comb(n, r):
    if r > n:
        return 0
    numer = 1
    denom = 1
    for i in range(1, r + 1):
        numer *= n - r + i
        denom *= i
    return numer // denom


def _comb_with_replacement(n, r):
    if n == 0 and r > 0:
        return 0
    return _comb(n + r - 1, r)
