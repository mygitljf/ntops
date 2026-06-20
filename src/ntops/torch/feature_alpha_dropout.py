import torch

import ntops
from ntops.torch import _vendor_triton
from ntops.torch.utils import _cached_make, _clone, _to_nt


_ALPHA = -1.7580993408473766


def _kernel_eligible(input, training, p, inplace):
    return (
        training
        and p != 0
        and not inplace
        and _to_nt(input.dtype) is not None
        and input.ndim >= 2
        and input.is_contiguous()
    )


def feature_alpha_dropout(input, p=0.5, training=True, inplace=False):
    if not _kernel_eligible(input, training, p, inplace):
        if not training or p == 0:
            if inplace:
                return input
            return _clone(input)
        if inplace:
            raise NotImplementedError("feature_alpha_dropout inplace=True needs an in-place kernel")
        if _to_nt(input.dtype) is None:
            raise NotImplementedError(f"feature_alpha_dropout kernel does not support {input.dtype}")
        if input.ndim < 2:
            raise NotImplementedError("feature_alpha_dropout kernel requires rank >= 2")
        input = input.contiguous()

    import random

    nt_dtype = _to_nt(input.dtype)
    alpha = _ALPHA
    a = ((1.0 - p) * (1.0 + p * alpha * alpha)) ** -0.5
    b = -a * p * alpha
    dropped = a * alpha + b
    spatial = input[0, 0].numel() if input.ndim > 2 else 1

    output = torch.empty_like(input)
    seed = random.randrange(0, 2**31)

    if _vendor_triton.feature_alpha_dropout_fast_path(
        input, output, float(p), seed, float(a), float(b), float(dropped)
    ):
        return output

    kernel = _cached_make(
        ntops.kernels.feature_alpha_dropout.premake,
        input.ndim,
        dtype=nt_dtype,
        block_size=ntops.kernels.feature_alpha_dropout.BLOCK_SIZE,
        num_warps=4,
        max_num_configs=1,
    )
    kernel(input, float(p), seed, int(spatial), float(a), float(b), float(dropped), output)
    return output
