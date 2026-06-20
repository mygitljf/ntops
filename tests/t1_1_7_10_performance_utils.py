from dataclasses import dataclass
import math

import torch
import torch.nn.functional as F

import ntops
from tests.skippers import skip_if_cuda_not_available


PERF_THRESHOLD = 0.90
WARMUP = 10
ITERATIONS = 30


@dataclass(frozen=True)
class PerfCase:
    op_name: str
    case_name: str
    make_pair: object
    rtol: float = 2e-3
    atol: float = 2e-3
    compare: bool = True


def _time_cuda(fn, warmup=WARMUP, iterations=ITERATIONS):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iterations):
        fn()
    end.record()
    end.synchronize()
    return start.elapsed_time(end) / iterations


def _assert_outputs_match(output, reference, rtol=2e-3, atol=2e-3):
    if isinstance(reference, (tuple, list)):
        assert len(output) == len(reference)
        for lhs, rhs in zip(output, reference):
            _assert_outputs_match(lhs, rhs, rtol=rtol, atol=atol)
        return
    assert output.shape == reference.shape
    assert output.dtype == reference.dtype
    if reference.dtype.is_floating_point:
        assert torch.allclose(output, reference, rtol=rtol, atol=atol, equal_nan=True)
    else:
        assert torch.equal(output, reference)


def _assert_shapes_match(output, reference):
    if isinstance(reference, (tuple, list)):
        assert len(output) == len(reference)
        for lhs, rhs in zip(output, reference):
            _assert_shapes_match(lhs, rhs)
        return
    assert output.shape == reference.shape
    assert output.dtype == reference.dtype


def _rand(shape, dtype):
    if dtype.is_floating_point:
        return torch.randn(shape, dtype=dtype, device="cuda")
    return torch.randint(-128, 128, shape, dtype=dtype, device="cuda")


# ---- feature_alpha_dropout (eval mode = deterministic identity, comparable) ----


def _make_feature_alpha_dropout(shape, dtype, training=False, p=0.5):
    def make_pair():
        input = _rand(shape, dtype)
        return (
            lambda: ntops.torch.feature_alpha_dropout(input, p=p, training=training),
            lambda: F.feature_alpha_dropout(input, p=p, training=training),
        )

    return make_pair


# ---- pixel_unshuffle ----


def _make_pixel_unshuffle(shape, dtype, factor, noncontig=False):
    def make_pair():
        input = _rand(shape, dtype)
        if noncontig:
            input = input.transpose(-1, -2)
        return (
            lambda: ntops.torch.pixel_unshuffle(input, factor),
            lambda: F.pixel_unshuffle(input, factor),
        )

    return make_pair


# ---- mse_loss ----


def _make_mse_loss(shape, dtype, reduction="none", noncontig=False):
    def make_pair():
        input = torch.randn(shape, dtype=dtype, device="cuda")
        target = torch.randn(shape, dtype=dtype, device="cuda")
        if noncontig:
            input = input.t()
            target = target.t()
        return (
            lambda: ntops.torch.mse_loss(input, target, reduction=reduction),
            lambda: F.mse_loss(input, target, reduction=reduction),
        )

    return make_pair


# ---- flip / fliplr ----


def _make_flip(shape, dtype, dims, noncontig=False):
    def make_pair():
        input = _rand(shape, dtype)
        if noncontig:
            input = input.transpose(-1, -2)
        return (
            lambda: ntops.torch.flip(input, dims),
            lambda: torch.flip(input, dims),
        )

    return make_pair


def _make_fliplr(shape, dtype):
    def make_pair():
        input = _rand(shape, dtype)
        return (
            lambda: ntops.torch.fliplr(input),
            lambda: torch.fliplr(input),
        )

    return make_pair


# ---- gumbel_softmax (hard=True path is comparable up to RNG; compare simplex props
#      through statistics is unstable for ratio, so we use soft path shapes for timing) ----


def _make_gumbel_softmax(shape, dtype, hard=False, tau=1.0, dim=-1):
    def make_pair():
        logits = torch.randn(shape, dtype=dtype, device="cuda")
        return (
            lambda: ntops.torch.gumbel_softmax(logits, tau=tau, hard=hard, dim=dim),
            lambda: F.gumbel_softmax(logits, tau=tau, hard=hard, dim=dim),
        )

    return make_pair


# ---- slice_scatter ----


def _make_slice_scatter(shape, dtype, dim, start, end, step=1):
    def make_pair():
        input = _rand(shape, dtype)
        indices = torch.arange(
            0 if start is None else start,
            input.size(dim) if end is None else end,
            step,
            device="cuda",
        )
        src = input.index_select(dim, indices).clone()
        return (
            lambda: ntops.torch.slice_scatter(input, src, dim=dim, start=start, end=end, step=step),
            lambda: torch.slice_scatter(input, src, dim=dim, start=start, end=end, step=step),
        )

    return make_pair


# ---- slogdet ----


def _make_slogdet(shape, dtype):
    def make_pair():
        input = torch.randn(shape, dtype=dtype, device="cuda")
        return (
            lambda: ntops.torch.slogdet(input),
            lambda: torch.linalg.slogdet(input),
        )

    return make_pair


# ---- heaviside ----


def _make_heaviside(shape, dtype, noncontig=False, broadcast=False):
    def make_pair():
        input = _rand(shape, dtype)
        if broadcast:
            values = torch.randn((1,), dtype=dtype, device="cuda") if dtype.is_floating_point else torch.randint(0, 4, (1,), dtype=dtype, device="cuda")
        else:
            values = _rand(shape, dtype)
        if noncontig:
            input = input.t()
            if not broadcast:
                values = values.t()
        return (
            lambda: ntops.torch.heaviside(input, values),
            lambda: torch.heaviside(input, values),
        )

    return make_pair


# ---- hsplit ----


def _make_hsplit(shape, dtype, ios):
    def make_pair():
        input = _rand(shape, dtype)
        return (
            lambda: ntops.torch.hsplit(input, ios),
            lambda: torch.hsplit(input, ios),
        )

    return make_pair


_LARGE = 1 << 24
_MID = 1 << 23


_PERF_CASES = [
    # ===================== T1-1-7 =====================
    # feature_alpha_dropout (training mode = real RNG+mask work; RNG so compare=False)
    PerfCase("feature_alpha_dropout", "f32_n8c32_3d", _make_feature_alpha_dropout((8, 32, 4096), torch.float32, training=True), compare=False),
    PerfCase("feature_alpha_dropout", "f32_n16c64_4d", _make_feature_alpha_dropout((16, 64, 64, 64), torch.float32, training=True), compare=False),
    PerfCase("feature_alpha_dropout", "f32_n32c32_4d", _make_feature_alpha_dropout((32, 32, 64, 64), torch.float32, training=True), compare=False),
    PerfCase("feature_alpha_dropout", "f32_n32c64_4d", _make_feature_alpha_dropout((32, 64, 64, 64), torch.float32, training=True), compare=False),
    PerfCase("feature_alpha_dropout", "f32_n4c128_4d", _make_feature_alpha_dropout((4, 128, 64, 64), torch.float32, training=True), compare=False),
    PerfCase("feature_alpha_dropout", "f32_n64c16_4d", _make_feature_alpha_dropout((64, 16, 64, 64), torch.float32, training=True), compare=False),
    PerfCase("feature_alpha_dropout", "f32_n8c16_5d", _make_feature_alpha_dropout((8, 16, 16, 32, 32), torch.float32, training=True), compare=False),
    PerfCase("feature_alpha_dropout", "f64_n16c32_4d", _make_feature_alpha_dropout((16, 32, 64, 64), torch.float64, training=True), compare=False),
    PerfCase("feature_alpha_dropout", "f64_n8c32_4d", _make_feature_alpha_dropout((8, 32, 48, 48), torch.float64, training=True), compare=False),
    PerfCase("feature_alpha_dropout", "f16_n16c64_4d", _make_feature_alpha_dropout((16, 64, 64, 64), torch.float16, training=True), compare=False),
    PerfCase("feature_alpha_dropout", "f16_n32c32_4d", _make_feature_alpha_dropout((32, 32, 64, 64), torch.float16, training=True), compare=False),
    PerfCase("feature_alpha_dropout", "f32_n8c256_4d", _make_feature_alpha_dropout((8, 256, 32, 32), torch.float32, training=True), compare=False),
    PerfCase("feature_alpha_dropout", "f32_n16c64_p03", _make_feature_alpha_dropout((16, 64, 64, 64), torch.float32, training=True, p=0.3), compare=False),
    PerfCase("feature_alpha_dropout", "f32_n16c64_p07", _make_feature_alpha_dropout((16, 64, 64, 64), torch.float32, training=True, p=0.7), compare=False),
    PerfCase("feature_alpha_dropout", "f32_n4c64_3d", _make_feature_alpha_dropout((4, 64, 8192), torch.float32, training=True), compare=False),
    # pixel_unshuffle
    PerfCase("pixel_unshuffle", "f32_n8c8_512_r2", _make_pixel_unshuffle((8, 8, 512, 512), torch.float32, 2)),
    PerfCase("pixel_unshuffle", "f16_n8c8_512_r2", _make_pixel_unshuffle((8, 8, 512, 512), torch.float16, 2)),
    PerfCase("pixel_unshuffle", "bf16_n8c8_512_r2", _make_pixel_unshuffle((8, 8, 512, 512), torch.bfloat16, 2)),
    PerfCase("pixel_unshuffle", "f32_n4c16_512_r4", _make_pixel_unshuffle((4, 16, 512, 512), torch.float32, 4)),
    PerfCase("pixel_unshuffle", "f16_n4c16_512_r4", _make_pixel_unshuffle((4, 16, 512, 512), torch.float16, 4)),
    PerfCase("pixel_unshuffle", "f32_n16c4_256_r2", _make_pixel_unshuffle((16, 4, 256, 256), torch.float32, 2)),
    PerfCase("pixel_unshuffle", "f32_n8c8_512_r4", _make_pixel_unshuffle((8, 8, 512, 512), torch.float32, 4)),
    PerfCase("pixel_unshuffle", "f32_n2c32_1024_r2", _make_pixel_unshuffle((2, 32, 1024, 1024), torch.float32, 2)),
    PerfCase("pixel_unshuffle", "f32_n16c8_256_r2", _make_pixel_unshuffle((16, 8, 256, 256), torch.float32, 2)),
    PerfCase("pixel_unshuffle", "f64_n4c8_256_r2", _make_pixel_unshuffle((4, 8, 256, 256), torch.float64, 2)),
    PerfCase("pixel_unshuffle", "f32_n8c16_512_r8", _make_pixel_unshuffle((8, 16, 512, 512), torch.float32, 8)),
    PerfCase("pixel_unshuffle", "i32_n8c8_512_r2", _make_pixel_unshuffle((8, 8, 512, 512), torch.int32, 2)),
    PerfCase("pixel_unshuffle", "f32_n32c4_128_r2", _make_pixel_unshuffle((32, 4, 128, 128), torch.float32, 2)),
    PerfCase("pixel_unshuffle", "f32_noncontig_r2", _make_pixel_unshuffle((8, 8, 512, 512), torch.float32, 2, noncontig=True)),
    PerfCase("pixel_unshuffle", "f16_noncontig_r2", _make_pixel_unshuffle((8, 8, 512, 512), torch.float16, 2, noncontig=True)),
    # mse_loss (none = elementwise kernel; mean/sum = native reduction)
    PerfCase("mse_loss", "f32_large_none_1d", _make_mse_loss((_LARGE,), torch.float32, "none")),
    PerfCase("mse_loss", "f16_large_none_1d", _make_mse_loss((_LARGE,), torch.float16, "none")),
    PerfCase("mse_loss", "bf16_large_none_1d", _make_mse_loss((_LARGE,), torch.bfloat16, "none")),
    PerfCase("mse_loss", "f64_large_none_1d", _make_mse_loss((_MID,), torch.float64, "none")),
    PerfCase("mse_loss", "f32_large_none_2d", _make_mse_loss((4096, 4096), torch.float32, "none")),
    PerfCase("mse_loss", "f32_large_none_3d", _make_mse_loss((256, 256, 256), torch.float32, "none")),
    PerfCase("mse_loss", "f32_mid_none_1d", _make_mse_loss((_MID,), torch.float32, "none")),
    PerfCase("mse_loss", "f16_mid_none_1d", _make_mse_loss((_MID,), torch.float16, "none")),
    PerfCase("mse_loss", "f32_large_mean", _make_mse_loss((_LARGE,), torch.float32, "mean")),
    PerfCase("mse_loss", "f32_large_sum", _make_mse_loss((_LARGE,), torch.float32, "sum")),
    PerfCase("mse_loss", "f16_large_mean", _make_mse_loss((_LARGE,), torch.float16, "mean")),
    PerfCase("mse_loss", "f16_large_sum", _make_mse_loss((_LARGE,), torch.float16, "sum")),
    PerfCase("mse_loss", "f64_large_mean", _make_mse_loss((_MID,), torch.float64, "mean")),
    PerfCase("mse_loss", "bf16_large_mean", _make_mse_loss((_LARGE,), torch.bfloat16, "mean")),
    PerfCase("mse_loss", "f32_noncontig_none", _make_mse_loss((4096, 4096), torch.float32, "none", noncontig=True)),
    PerfCase("mse_loss", "f16_noncontig_none", _make_mse_loss((4096, 4096), torch.float16, "none", noncontig=True)),
    # flip
    PerfCase("flip", "f32_large_2d_dim01", _make_flip((4096, 4096), torch.float32, [0, 1])),
    PerfCase("flip", "f16_large_2d_dim01", _make_flip((4096, 4096), torch.float16, [0, 1])),
    PerfCase("flip", "bf16_large_2d_dim01", _make_flip((4096, 4096), torch.bfloat16, [0, 1])),
    PerfCase("flip", "f32_large_2d_dim0", _make_flip((4096, 4096), torch.float32, [0])),
    PerfCase("flip", "f32_large_2d_dim1", _make_flip((4096, 4096), torch.float32, [1])),
    PerfCase("flip", "f32_large_3d_dim2", _make_flip((256, 256, 256), torch.float32, [2])),
    PerfCase("flip", "f32_large_3d_dim0", _make_flip((256, 256, 256), torch.float32, [0])),
    PerfCase("flip", "f32_large_3d_dim12", _make_flip((256, 256, 256), torch.float32, [1, 2])),
    PerfCase("flip", "f16_large_3d_dim012", _make_flip((256, 256, 256), torch.float16, [0, 1, 2])),
    PerfCase("flip", "f32_large_1d", _make_flip((_LARGE,), torch.float32, [0])),
    PerfCase("flip", "f16_large_1d", _make_flip((_LARGE,), torch.float16, [0])),
    PerfCase("flip", "i64_large_1d", _make_flip((_LARGE,), torch.int64, [0])),
    PerfCase("flip", "i32_large_2d_dim01", _make_flip((4096, 4096), torch.int32, [0, 1])),
    PerfCase("flip", "f64_large_2d_dim1", _make_flip((2048, 2048), torch.float64, [1])),
    PerfCase("flip", "f32_noncontig_dim0", _make_flip((4096, 4096), torch.float32, [0], noncontig=True)),
    PerfCase("flip", "f16_noncontig_dim1", _make_flip((4096, 4096), torch.float16, [1], noncontig=True)),
    # fliplr
    PerfCase("fliplr", "f32_large_2d", _make_fliplr((4096, 4096), torch.float32)),
    PerfCase("fliplr", "f16_large_2d", _make_fliplr((4096, 4096), torch.float16)),
    PerfCase("fliplr", "bf16_large_2d", _make_fliplr((4096, 4096), torch.bfloat16)),
    PerfCase("fliplr", "f64_large_2d", _make_fliplr((2048, 2048), torch.float64)),
    PerfCase("fliplr", "i32_large_2d", _make_fliplr((4096, 4096), torch.int32)),
    PerfCase("fliplr", "i64_large_2d", _make_fliplr((4096, 4096), torch.int64)),
    PerfCase("fliplr", "f32_large_3d", _make_fliplr((256, 256, 256), torch.float32)),
    PerfCase("fliplr", "f16_large_3d", _make_fliplr((256, 256, 256), torch.float16)),
    PerfCase("fliplr", "f64_large_3d", _make_fliplr((256, 256, 128), torch.float64)),
    PerfCase("fliplr", "f32_large_2d_rect", _make_fliplr((8192, 2048), torch.float32)),
    PerfCase("fliplr", "f32_large_2d_tall", _make_fliplr((2048, 8192), torch.float32)),
    PerfCase("fliplr", "f32_4d", _make_fliplr((64, 64, 64, 64), torch.float32)),
    PerfCase("fliplr", "f16_4d", _make_fliplr((64, 64, 64, 64), torch.float16)),
    PerfCase("fliplr", "i32_large_3d", _make_fliplr((256, 256, 256), torch.int32)),
    PerfCase("fliplr", "f32_large_2d_8192", _make_fliplr((8192, 8192), torch.float32)),
    # ===================== T1-1-10 =====================
    # gumbel_softmax (RNG so compare=False)
    PerfCase("gumbel_softmax", "f32_4096x256_soft", _make_gumbel_softmax((4096, 256), torch.float32, hard=False), compare=False),
    PerfCase("gumbel_softmax", "f32_4096x256_hard", _make_gumbel_softmax((4096, 256), torch.float32, hard=True), compare=False),
    PerfCase("gumbel_softmax", "f32_16384x64_soft", _make_gumbel_softmax((16384, 64), torch.float32, hard=False), compare=False),
    PerfCase("gumbel_softmax", "f32_16384x64_hard", _make_gumbel_softmax((16384, 64), torch.float32, hard=True), compare=False),
    PerfCase("gumbel_softmax", "f32_65536x16_soft", _make_gumbel_softmax((65536, 16), torch.float32, hard=False), compare=False),
    PerfCase("gumbel_softmax", "f32_65536x16_hard", _make_gumbel_softmax((65536, 16), torch.float32, hard=True), compare=False),
    PerfCase("gumbel_softmax", "f32_8192x128_soft", _make_gumbel_softmax((8192, 128), torch.float32, hard=False), compare=False),
    PerfCase("gumbel_softmax", "f32_8192x128_hard", _make_gumbel_softmax((8192, 128), torch.float32, hard=True), compare=False),
    PerfCase("gumbel_softmax", "f32_4096x512_soft", _make_gumbel_softmax((4096, 512), torch.float32, hard=False), compare=False),
    PerfCase("gumbel_softmax", "f32_1024x1024_soft", _make_gumbel_softmax((1024, 1024), torch.float32, hard=False), compare=False),
    PerfCase("gumbel_softmax", "f32_32768x32_soft", _make_gumbel_softmax((32768, 32), torch.float32, hard=False), compare=False),
    PerfCase("gumbel_softmax", "f32_2048x256_tau05", _make_gumbel_softmax((2048, 256), torch.float32, hard=False, tau=0.5), compare=False),
    PerfCase("gumbel_softmax", "f64_4096x128_soft", _make_gumbel_softmax((4096, 128), torch.float64, hard=False), compare=False),
    PerfCase("gumbel_softmax", "f64_4096x128_hard", _make_gumbel_softmax((4096, 128), torch.float64, hard=True), compare=False),
    PerfCase("gumbel_softmax", "f64_8192x64_soft", _make_gumbel_softmax((8192, 64), torch.float64, hard=False), compare=False),
    # slice_scatter
    PerfCase("slice_scatter", "f32_large_dim1_half", _make_slice_scatter((4096, 4096), torch.float32, 1, 0, 2048)),
    PerfCase("slice_scatter", "f32_large_dim0_half", _make_slice_scatter((4096, 4096), torch.float32, 0, 0, 2048)),
    PerfCase("slice_scatter", "f16_large_dim1_half", _make_slice_scatter((4096, 4096), torch.float16, 1, 0, 2048)),
    PerfCase("slice_scatter", "f16_large_dim0_half", _make_slice_scatter((4096, 4096), torch.float16, 0, 0, 2048)),
    PerfCase("slice_scatter", "bf16_large_dim1_half", _make_slice_scatter((4096, 4096), torch.bfloat16, 1, 0, 2048)),
    PerfCase("slice_scatter", "f32_large_dim1_quarter", _make_slice_scatter((4096, 4096), torch.float32, 1, 0, 1024)),
    PerfCase("slice_scatter", "f32_large_dim0_quarter", _make_slice_scatter((4096, 4096), torch.float32, 0, 0, 1024)),
    PerfCase("slice_scatter", "f32_3d_dim2", _make_slice_scatter((256, 256, 256), torch.float32, 2, 0, 128)),
    PerfCase("slice_scatter", "f32_3d_dim1", _make_slice_scatter((256, 256, 256), torch.float32, 1, 0, 128)),
    PerfCase("slice_scatter", "f32_3d_dim0", _make_slice_scatter((256, 256, 256), torch.float32, 0, 0, 128)),
    PerfCase("slice_scatter", "f16_3d_dim2", _make_slice_scatter((256, 256, 256), torch.float16, 2, 0, 128)),
    PerfCase("slice_scatter", "f64_mid_dim1", _make_slice_scatter((2048, 2048), torch.float64, 1, 0, 1024)),
    PerfCase("slice_scatter", "i64_large_dim1", _make_slice_scatter((4096, 4096), torch.int64, 1, 0, 2048)),
    PerfCase("slice_scatter", "i32_large_dim0", _make_slice_scatter((4096, 4096), torch.int32, 0, 0, 2048)),
    PerfCase("slice_scatter", "f32_large_dim1_step2", _make_slice_scatter((4096, 4096), torch.float32, 1, 0, 4096, 2)),
    # slogdet
    PerfCase("slogdet", "f32_batch256_64x64", _make_slogdet((256, 64, 64), torch.float32)),
    PerfCase("slogdet", "f32_batch128_64x64", _make_slogdet((128, 64, 64), torch.float32)),
    PerfCase("slogdet", "f32_batch512_32x32", _make_slogdet((512, 32, 32), torch.float32)),
    PerfCase("slogdet", "f32_batch64_128x128", _make_slogdet((64, 128, 128), torch.float32)),
    PerfCase("slogdet", "f32_batch64_96x96", _make_slogdet((64, 96, 96), torch.float32)),
    PerfCase("slogdet", "f32_batch256_48x48", _make_slogdet((256, 48, 48), torch.float32)),
    PerfCase("slogdet", "f32_batch1024_16x16", _make_slogdet((1024, 16, 16), torch.float32)),
    PerfCase("slogdet", "f32_single_512x512", _make_slogdet((512, 512), torch.float32)),
    PerfCase("slogdet", "f32_single_1024x1024", _make_slogdet((1024, 1024), torch.float32)),
    PerfCase("slogdet", "f32_single_256x256", _make_slogdet((256, 256), torch.float32)),
    PerfCase("slogdet", "f64_batch64_64x64", _make_slogdet((64, 64, 64), torch.float64)),
    PerfCase("slogdet", "f64_batch128_48x48", _make_slogdet((128, 48, 48), torch.float64)),
    PerfCase("slogdet", "f64_batch32_96x96", _make_slogdet((32, 96, 96), torch.float64)),
    PerfCase("slogdet", "f64_single_512x512", _make_slogdet((512, 512), torch.float64)),
    PerfCase("slogdet", "f64_batch64_128x128", _make_slogdet((64, 128, 128), torch.float64)),
    # heaviside (f32 contiguous = kernel; others native)
    PerfCase("heaviside", "f32_large_1d", _make_heaviside((_LARGE,), torch.float32)),
    PerfCase("heaviside", "f32_large_2d", _make_heaviside((4096, 4096), torch.float32)),
    PerfCase("heaviside", "f32_large_3d", _make_heaviside((256, 256, 256), torch.float32)),
    PerfCase("heaviside", "f32_mid_1d", _make_heaviside((_MID,), torch.float32)),
    PerfCase("heaviside", "f16_large_1d", _make_heaviside((_LARGE,), torch.float16)),
    PerfCase("heaviside", "f16_large_2d", _make_heaviside((4096, 4096), torch.float16)),
    PerfCase("heaviside", "f16_mid_1d", _make_heaviside((_MID,), torch.float16)),
    PerfCase("heaviside", "bf16_large_1d", _make_heaviside((_LARGE,), torch.bfloat16)),
    PerfCase("heaviside", "f64_large_1d", _make_heaviside((_MID,), torch.float64)),
    PerfCase("heaviside", "i32_large_1d", _make_heaviside((_LARGE,), torch.int32)),
    PerfCase("heaviside", "i64_large_1d", _make_heaviside((_LARGE,), torch.int64)),
    PerfCase("heaviside", "i16_large_1d", _make_heaviside((_LARGE,), torch.int16)),
    PerfCase("heaviside", "f32_broadcast", _make_heaviside((4096, 4096), torch.float32, broadcast=True)),
    PerfCase("heaviside", "f16_broadcast", _make_heaviside((4096, 4096), torch.float16, broadcast=True)),
    PerfCase("heaviside", "f32_noncontig", _make_heaviside((4096, 4096), torch.float32, noncontig=True)),
    # hsplit
    PerfCase("hsplit", "f32_large_2d_2", _make_hsplit((4096, 4096), torch.float32, 2)),
    PerfCase("hsplit", "f32_large_2d_4", _make_hsplit((4096, 4096), torch.float32, 4)),
    PerfCase("hsplit", "f32_large_2d_8", _make_hsplit((4096, 4096), torch.float32, 8)),
    PerfCase("hsplit", "f32_large_1d_2", _make_hsplit((_LARGE,), torch.float32, 2)),
    PerfCase("hsplit", "f32_large_1d_4", _make_hsplit((_LARGE,), torch.float32, 4)),
    PerfCase("hsplit", "f16_large_2d_2", _make_hsplit((4096, 4096), torch.float16, 2)),
    PerfCase("hsplit", "f16_large_2d_4", _make_hsplit((4096, 4096), torch.float16, 4)),
    PerfCase("hsplit", "f16_large_1d_4", _make_hsplit((_LARGE,), torch.float16, 4)),
    PerfCase("hsplit", "i64_large_2d_2", _make_hsplit((4096, 4096), torch.int64, 2)),
    PerfCase("hsplit", "i32_large_2d_4", _make_hsplit((4096, 4096), torch.int32, 4)),
    PerfCase("hsplit", "f64_large_2d_2", _make_hsplit((2048, 2048), torch.float64, 2)),
    PerfCase("hsplit", "bf16_large_2d_2", _make_hsplit((4096, 4096), torch.bfloat16, 2)),
    PerfCase("hsplit", "f32_3d_2", _make_hsplit((256, 256, 256), torch.float32, 2)),
    PerfCase("hsplit", "f32_large_2d_list", _make_hsplit((4096, 4096), torch.float32, [1024, 3072])),
    PerfCase("hsplit", "f16_3d_4", _make_hsplit((256, 256, 256), torch.float16, 4)),
]


def perf_cases_for(op_name):
    return [case for case in _PERF_CASES if case.op_name == op_name]


def all_op_names():
    seen = []
    for case in _PERF_CASES:
        if case.op_name not in seen:
            seen.append(case.op_name)
    return seen


@skip_if_cuda_not_available
def run_perf_case(case):
    ntops_call, torch_call = case.make_pair()
    ntops_output = ntops_call()
    reference = torch_call()
    if case.compare:
        _assert_outputs_match(ntops_output, reference, rtol=case.rtol, atol=case.atol)
    else:
        _assert_shapes_match(ntops_output, reference)

    ntops_ms = _time_cuda(ntops_call)
    torch_ms = _time_cuda(torch_call)
    ratio = torch_ms / ntops_ms if ntops_ms > 0 else float("inf")
    print(
        f"{case.op_name}/{case.case_name}: ntops={ntops_ms:.4f} ms, "
        f"torch={torch_ms:.4f} ms, torch/ntops={ratio:.3f}x"
    )
    return ratio


@skip_if_cuda_not_available
def geomean_for(op_name):
    ratios = [run_perf_case(case) for case in perf_cases_for(op_name)]
    geomean = math.prod(ratios) ** (1.0 / len(ratios))
    print(f"{op_name}_geomean: torch/ntops={geomean:.3f}x over {len(ratios)} cases")
    return geomean
