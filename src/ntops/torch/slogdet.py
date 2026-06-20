import torch

from ntops.torch._vendor_triton import is_iluvatar_device, slogdet_batched

_rt = __import__("torch")


# Matrices wider than this fall back to cuSOLVER (D 类): the in-kernel NxN block
# (padded to a power of two) grows quadratically, so large N exceeds one tile.
_MAX_KERNEL_N = 64


def _cpu_slogdet(input):
    # B 类: CoreX GPU f64 linalg is unsupported -- single-matrix LU silently returns 0,
    # batched LU (cublasDgetrfBatched) raises CUBLAS_STATUS_NOT_SUPPORTED. CPU is the
    # only correct path for double precision on this device.
    # ntops: capability-fallback - CoreX has no usable GPU f64 LU/GEMM path.
    sign, logabsdet = _rt.linalg.slogdet(input.cpu())
    return _rt.return_types.slogdet((sign.to(input.device), logabsdet.to(input.device)))


def slogdet(input):
    if input.dtype == torch.float64 and is_iluvatar_device(input):
        return _cpu_slogdet(input)

    # f32 square matrices up to N=64 run the per-matrix LU kernel on all platforms;
    # single matrices and higher-rank batches are reshaped to (-1, N, N).
    if (
        input.dtype == torch.float32
        and input.ndim >= 2
        and input.shape[-1] == input.shape[-2]
        and 0 < input.shape[-1] <= _MAX_KERNEL_N
    ):
        n = input.shape[-1]
        batched = input.contiguous().view(input.numel() // (n * n), n, n)
        batch = batched.shape[0]
        sign = torch.empty([batch], dtype=torch.float32, device=input.device)
        logabsdet = torch.empty([batch], dtype=torch.float32, device=input.device)
        if slogdet_batched(batched, sign, logabsdet):
            return _rt.return_types.slogdet(
                (sign.view(input.shape[:-2]), logabsdet.view(input.shape[:-2]))
            )

    # Remaining paths keep cuSOLVER: N > _MAX_KERNEL_N (D 类, tile cap) and f64
    # (B 类: no reliable in-kernel f64 LU / no f64 gemm on CoreX).
    # ntops: capability-fallback - current vendor LU kernel only supports f32 N<=64.
    return _rt.linalg.slogdet(
        input._torch_ref if hasattr(input, "_torch_ref") and input._torch_ref is not None else input
    )
