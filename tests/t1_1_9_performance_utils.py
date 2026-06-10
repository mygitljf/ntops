from dataclasses import dataclass

import torch
import torch.nn.functional as F

import ntops
from tests.skippers import skip_if_cuda_not_available


PERF_THRESHOLD = 0.95
WARMUP = 10
ITERATIONS = 30


@dataclass(frozen=True)
class PerfCase:
    op_name: str
    case_name: str
    make_pair: object
    rtol: float = 2e-3
    atol: float = 2e-3


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
    if isinstance(reference, tuple):
        assert isinstance(output, tuple)
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


def _rand_float(shape, dtype):
    return torch.randn(shape, dtype=dtype, device="cuda")


def _rand_frac(shape, dtype):
    return torch.randn(shape, dtype=dtype, device="cuda") * 16 - 8


def _rand_index(shape, dim_size):
    return torch.randint(0, dim_size, shape, dtype=torch.long, device="cuda")


def _make_frac(shape, dtype, out=False, noncontig=False, permute=False):
    def make_pair():
        input = _rand_frac(shape, dtype)
        if noncontig:
            input = input.t()
        if permute:
            input = input.permute(2, 0, 1)
        if not out:
            return lambda: ntops.torch.frac(input), lambda: torch.frac(input)
        nt_out = torch.empty_like(input)
        th_out = torch.empty_like(input)
        return lambda: ntops.torch.frac(input, out=nt_out), lambda: torch.frac(input, out=th_out)

    return make_pair


def _make_scatter(shape, dim, dtype, out=False, noncontig=False):
    def make_pair():
        input = _rand_float(shape, dtype) if dtype.is_floating_point else torch.randint(-64, 64, shape, dtype=dtype, device="cuda")
        src = _rand_float(shape, dtype) if dtype.is_floating_point else torch.randint(-64, 64, shape, dtype=dtype, device="cuda")
        index = _rand_index(shape, shape[dim])
        if noncontig:
            input = input.transpose(0, 1).contiguous().transpose(0, 1)
            src = src.transpose(0, 1).contiguous().transpose(0, 1)
            index = index.transpose(0, 1).contiguous().transpose(0, 1)
        if not out:
            return (
                lambda: ntops.torch.scatter_add(input, dim, index, src),
                lambda: torch.scatter_add(input, dim, index, src),
            )
        nt_out = torch.empty_like(input)
        th_out = torch.empty_like(input)
        return (
            lambda: ntops.torch.scatter_add(input, dim, index, src, out=nt_out),
            lambda: torch.scatter_add(input, dim, index, src, out=th_out),
        )

    return make_pair


def _target(batch, classes):
    target = torch.full((batch, classes), -1, dtype=torch.long, device="cuda")
    target[:, 0] = torch.arange(batch, dtype=torch.long, device="cuda") % classes
    if classes > 2:
        target[:, 1] = (torch.arange(batch, dtype=torch.long, device="cuda") + 2) % classes
    if classes > 4:
        target[:, 2] = (torch.arange(batch, dtype=torch.long, device="cuda") + 4) % classes
    return target


def _make_multilabel(batch, classes, dtype, reduction="mean", noncontig=False):
    def make_pair():
        input = torch.randn((batch, classes), dtype=dtype, device="cuda")
        target = _target(batch, classes)
        if noncontig:
            input = input.t()
            target = target.t()
        return (
            lambda: ntops.torch.multilabel_margin_loss(input, target, reduction=reduction),
            lambda: F.multilabel_margin_loss(input, target, reduction=reduction),
        )

    return make_pair


def _make_pool2d(shape, dtype, kernel_size, output_size, sample=0.5, noncontig=False, return_indices=True):
    def make_pair():
        input = torch.randn(shape, dtype=dtype, device="cuda")
        if noncontig:
            input = input.transpose(-1, -2)
        n_batch = 1 if input.ndim == 3 else input.shape[0]
        random_samples = torch.full(
            (n_batch, input.shape[-3], 2),
            sample,
            dtype=dtype,
            device="cuda",
        )
        return (
            lambda: ntops.torch.fractional_max_pool2d(
                input,
                kernel_size,
                output_size=output_size,
                _random_samples=random_samples,
                return_indices=return_indices,
            ),
            lambda: F.fractional_max_pool2d(
                input,
                kernel_size,
                output_size=output_size,
                _random_samples=random_samples,
                return_indices=return_indices,
            ),
        )

    return make_pair


def _make_pool3d(shape, dtype, kernel_size, output_size, sample=0.5, noncontig=False, return_indices=True):
    def make_pair():
        input = torch.randn(shape, dtype=dtype, device="cuda")
        if noncontig:
            input = input.transpose(-1, -2)
        n_batch = 1 if input.ndim == 4 else input.shape[0]
        random_samples = torch.full(
            (n_batch, input.shape[-4], 3),
            sample,
            dtype=dtype,
            device="cuda",
        )
        return (
            lambda: ntops.torch.fractional_max_pool3d(
                input,
                kernel_size,
                output_size=output_size,
                _random_samples=random_samples,
                return_indices=return_indices,
            ),
            lambda: F.fractional_max_pool3d(
                input,
                kernel_size,
                output_size=output_size,
                _random_samples=random_samples,
                return_indices=return_indices,
            ),
        )

    return make_pair


_PERF_CASES = [
    PerfCase("frac", "f16_large_1d", _make_frac((1 << 24,), torch.float16)),
    PerfCase("frac", "f32_large_1d", _make_frac((1 << 24,), torch.float32)),
    PerfCase("frac", "f64_large_1d", _make_frac((1 << 23,), torch.float64)),
    PerfCase("frac", "bf16_large_1d", _make_frac((1 << 24,), torch.bfloat16)),
    PerfCase("frac", "f32_large_2d", _make_frac((4096, 4096), torch.float32)),
    PerfCase("frac", "f16_large_3d", _make_frac((256, 256, 256), torch.float16)),
    PerfCase("frac", "f32_large_3d", _make_frac((256, 256, 256), torch.float32)),
    PerfCase("frac", "f64_mid_3d", _make_frac((256, 256, 128), torch.float64)),
    PerfCase("frac", "f32_out_1d", _make_frac((1 << 24,), torch.float32, out=True)),
    PerfCase("frac", "f16_out_2d", _make_frac((4096, 4096), torch.float16, out=True)),
    PerfCase("frac", "f32_mid_1d", _make_frac((1 << 24,), torch.float32)),
    PerfCase("frac", "bf16_mid_1d", _make_frac((1 << 24,), torch.bfloat16)),
    PerfCase("frac", "f64_mid_1d", _make_frac((1 << 23,), torch.float64)),
    PerfCase("frac", "f32_small_1d", _make_frac((1 << 23,), torch.float32)),
    PerfCase("frac", "f16_small_1d", _make_frac((1 << 24,), torch.float16)),
    PerfCase("frac", "f32_noncontig_2048", _make_frac((4096, 4096), torch.float32, noncontig=True)),
    PerfCase("frac", "f16_noncontig_2048", _make_frac((4096, 4096), torch.float16, noncontig=True)),
    PerfCase("frac", "f64_noncontig_1024", _make_frac((2048, 2048), torch.float64, noncontig=True)),
    PerfCase("frac", "f32_permute3d", _make_frac((256, 256, 256), torch.float32, permute=True)),
    PerfCase("frac", "f32_permute3d_out", _make_frac((256, 256, 256), torch.float32, out=True, permute=True)),
    PerfCase("scatter_add", "f32_large_dim1", _make_scatter((4096, 4096), 1, torch.float32)),
    PerfCase("scatter_add", "f64_mid_dim1", _make_scatter((2048, 2048), 1, torch.float64)),
    PerfCase("scatter_add", "i32_large_dim1", _make_scatter((4096, 4096), 1, torch.int32)),
    PerfCase("scatter_add", "i64_mid_dim1", _make_scatter((2048, 2048), 1, torch.int64)),
    PerfCase("scatter_add", "f32_large_dim0", _make_scatter((4096, 4096), 0, torch.float32)),
    PerfCase("scatter_add", "f32_3d_dim2", _make_scatter((256, 256, 256), 2, torch.float32)),
    PerfCase("scatter_add", "f32_3d_dim0", _make_scatter((256, 256, 256), 0, torch.float32)),
    PerfCase("scatter_add", "f32_out_dim1", _make_scatter((4096, 4096), 1, torch.float32, out=True)),
    PerfCase("scatter_add", "f32_mid_dim1", _make_scatter((3072, 3072), 1, torch.float32)),
    PerfCase("scatter_add", "f32_small_dim1", _make_scatter((2048, 4096), 1, torch.float32)),
    PerfCase("scatter_add", "f32_4d_dim3", _make_scatter((64, 64, 64, 64), 3, torch.float32)),
    PerfCase("scatter_add", "f32_1d", _make_scatter((1 << 24,), 0, torch.float32)),
    PerfCase("multilabel_margin_loss", "f32_1024x16_mean", _make_multilabel(131072, 16, torch.float32, "mean")),
    PerfCase("multilabel_margin_loss", "f16_1024x16_mean", _make_multilabel(131072, 16, torch.float16, "mean")),
    PerfCase("multilabel_margin_loss", "f64_512x16_mean", _make_multilabel(65536, 16, torch.float64, "mean")),
    PerfCase("multilabel_margin_loss", "f32_1024x16_sum", _make_multilabel(131072, 16, torch.float32, "sum")),
    PerfCase("multilabel_margin_loss", "f32_1024x16_none", _make_multilabel(131072, 16, torch.float32, "none")),
    PerfCase("multilabel_margin_loss", "f32_2048x8_mean", _make_multilabel(262144, 8, torch.float32, "mean")),
    PerfCase("multilabel_margin_loss", "f16_2048x8_mean", _make_multilabel(262144, 8, torch.float16, "mean")),
    PerfCase("multilabel_margin_loss", "f32_768x5_mean", _make_multilabel(262144, 5, torch.float32, "mean")),
    PerfCase("multilabel_margin_loss", "f32_768x5_none", _make_multilabel(262144, 5, torch.float32, "none")),
    PerfCase("multilabel_margin_loss", "f32_1d_sum", _make_multilabel(131072, 12, torch.float32, "sum")),
    PerfCase("multilabel_margin_loss", "f32_4096x4_mean", _make_multilabel(262144, 4, torch.float32, "mean")),
    PerfCase("multilabel_margin_loss", "f16_4096x4_mean", _make_multilabel(262144, 4, torch.float16, "mean")),
    PerfCase("fractional_max_pool2d", "f32_n1c16_64x64", _make_pool2d((8, 64, 128, 128), torch.float32, 2, (64, 64))),
    PerfCase("fractional_max_pool2d", "f16_n1c16_64x64", _make_pool2d((8, 64, 128, 128), torch.float16, 2, (64, 64))),
    PerfCase("fractional_max_pool2d", "f64_n1c8_48x48", _make_pool2d((16, 32, 128, 128), torch.float64, 2, (64, 64))),
    PerfCase("fractional_max_pool2d", "f32_n4c8_64x64", _make_pool2d((16, 32, 128, 128), torch.float32, 2, (64, 64))),
    PerfCase("fractional_max_pool2d", "f16_n4c8_64x64", _make_pool2d((16, 32, 128, 128), torch.float16, 2, (64, 64))),
    PerfCase("fractional_max_pool2d", "f32_k3_96x96", _make_pool2d((16, 64, 192, 192), torch.float32, 3, (64, 64))),
    PerfCase("fractional_max_pool2d", "f16_k3_96x96", _make_pool2d((16, 64, 192, 192), torch.float16, 3, (64, 64))),
    PerfCase("fractional_max_pool2d", "f32_rect", _make_pool2d((16, 64, 160, 128), torch.float32, (2, 3), (80, 40))),
    PerfCase("fractional_max_pool2d", "f32_small", _make_pool2d((32, 32, 128, 128), torch.float32, 2, (64, 64))),
    PerfCase("fractional_max_pool2d", "f16_small", _make_pool2d((32, 32, 128, 128), torch.float16, 2, (64, 64))),
    PerfCase("fractional_max_pool2d", "f32_3d_input", _make_pool2d((512, 160, 160), torch.float32, 2, (80, 80))),
    PerfCase("fractional_max_pool2d", "f16_3d_input", _make_pool2d((512, 160, 160), torch.float16, 2, (80, 80))),
    PerfCase("fractional_max_pool2d", "f32_output_only", _make_pool2d((8, 64, 128, 128), torch.float32, 2, (64, 64), return_indices=False)),
    PerfCase("fractional_max_pool2d", "f16_output_only", _make_pool2d((8, 64, 128, 128), torch.float16, 2, (64, 64), return_indices=False)),
    PerfCase("fractional_max_pool2d", "f32_sample025", _make_pool2d((8, 64, 128, 128), torch.float32, 2, (64, 64), sample=0.25)),
    PerfCase("fractional_max_pool2d", "f32_sample075", _make_pool2d((8, 64, 128, 128), torch.float32, 2, (64, 64), sample=0.75)),
    PerfCase("fractional_max_pool2d", "f32_noncontig", _make_pool2d((8, 64, 128, 160), torch.float32, 2, (64, 64), noncontig=True)),
    PerfCase("fractional_max_pool2d", "f16_noncontig", _make_pool2d((8, 64, 128, 160), torch.float16, 2, (64, 64), noncontig=True)),
    PerfCase("fractional_max_pool2d", "f32_large_c32", _make_pool2d((16, 128, 128, 128), torch.float32, 2, (64, 64))),
    PerfCase("fractional_max_pool2d", "f16_large_c32", _make_pool2d((16, 128, 128, 128), torch.float16, 2, (64, 64))),
    PerfCase("fractional_max_pool3d", "f32_n1c4_32cube", _make_pool3d((4, 32, 64, 64, 64), torch.float32, 2, (32, 32, 32))),
    PerfCase("fractional_max_pool3d", "f16_n1c4_32cube", _make_pool3d((4, 32, 64, 64, 64), torch.float16, 2, (32, 32, 32))),
    PerfCase("fractional_max_pool3d", "f64_n1c2_24cube", _make_pool3d((4, 32, 64, 64, 64), torch.float64, 2, (32, 32, 32))),
    PerfCase("fractional_max_pool3d", "f32_n2c4_32cube", _make_pool3d((4, 32, 64, 64, 64), torch.float32, 2, (32, 32, 32))),
    PerfCase("fractional_max_pool3d", "f16_n2c4_32cube", _make_pool3d((4, 32, 64, 64, 64), torch.float16, 2, (32, 32, 32))),
    PerfCase("fractional_max_pool3d", "f32_k3_36cube", _make_pool3d((4, 16, 96, 96, 96), torch.float32, 3, (32, 32, 32))),
    PerfCase("fractional_max_pool3d", "f16_k3_36cube", _make_pool3d((4, 16, 96, 96, 96), torch.float16, 3, (32, 32, 32))),
    PerfCase("fractional_max_pool3d", "f32_rect", _make_pool3d((4, 16, 64, 80, 48), torch.float32, (2, 3, 2), (32, 24, 24))),
    PerfCase("fractional_max_pool3d", "f32_small", _make_pool3d((16, 32, 32, 32, 32), torch.float32, 2, (16, 16, 16))),
    PerfCase("fractional_max_pool3d", "f16_small", _make_pool3d((16, 32, 32, 32, 32), torch.float16, 2, (16, 16, 16))),
    PerfCase("fractional_max_pool3d", "f32_4d_input", _make_pool3d((32, 64, 64, 64), torch.float32, 2, (32, 32, 32))),
    PerfCase("fractional_max_pool3d", "f16_4d_input", _make_pool3d((32, 64, 64, 64), torch.float16, 2, (32, 32, 32))),
    PerfCase("fractional_max_pool3d", "f32_output_only", _make_pool3d((4, 32, 64, 64, 64), torch.float32, 2, (32, 32, 32), return_indices=False)),
    PerfCase("fractional_max_pool3d", "f16_output_only", _make_pool3d((4, 32, 64, 64, 64), torch.float16, 2, (32, 32, 32), return_indices=False)),
    PerfCase("fractional_max_pool3d", "f32_sample025", _make_pool3d((4, 32, 64, 64, 64), torch.float32, 2, (32, 32, 32), sample=0.25)),
    PerfCase("fractional_max_pool3d", "f32_sample075", _make_pool3d((4, 32, 64, 64, 64), torch.float32, 2, (32, 32, 32), sample=0.75)),
    PerfCase("fractional_max_pool3d", "f32_noncontig", _make_pool3d((4, 32, 48, 64, 80), torch.float32, 2, (24, 32, 32), noncontig=True)),
    PerfCase("fractional_max_pool3d", "f16_noncontig", _make_pool3d((4, 32, 48, 64, 80), torch.float16, 2, (24, 32, 32), noncontig=True)),
    PerfCase("fractional_max_pool3d", "f32_c8", _make_pool3d((8, 32, 48, 48, 48), torch.float32, 2, (24, 24, 24))),
    PerfCase("fractional_max_pool3d", "f16_c8", _make_pool3d((8, 32, 48, 48, 48), torch.float16, 2, (24, 24, 24))),
]


def perf_cases_for(op_name):
    return [case for case in _PERF_CASES if case.op_name == op_name]


@skip_if_cuda_not_available
def run_perf_case(case):
    ntops_call, torch_call = case.make_pair()
    ntops_output = ntops_call()
    reference = torch_call()
    _assert_outputs_match(ntops_output, reference, rtol=case.rtol, atol=case.atol)

    ntops_ms = _time_cuda(ntops_call)
    torch_ms = _time_cuda(torch_call)
    ratio = torch_ms / ntops_ms if ntops_ms > 0 else float("inf")
    print(
        f"{case.op_name}/{case.case_name}: ntops={ntops_ms:.4f} ms, "
        f"torch={torch_ms:.4f} ms, torch/ntops={ratio:.3f}x"
    )
    assert ratio >= PERF_THRESHOLD
