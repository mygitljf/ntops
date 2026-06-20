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


# ---- kl_div (none = elementwise kernel; mean/sum/batchmean = native reduction) ----


def _make_kl_div(shape, dtype, reduction="none", log_target=False, noncontig=False):
    def make_pair():
        input = torch.randn(shape, dtype=dtype, device="cuda").log_softmax(-1)
        if log_target:
            target = torch.randn(shape, dtype=dtype, device="cuda").log_softmax(-1)
        else:
            target = torch.randn(shape, dtype=dtype, device="cuda").softmax(-1)
        if noncontig:
            input = input.t()
            target = target.t()
        return (
            lambda: ntops.torch.kl_div(input, target, reduction=reduction, log_target=log_target),
            lambda: F.kl_div(input, target, reduction=reduction, log_target=log_target),
        )

    return make_pair


# ---- combinations ----


def _make_combinations(n, r, dtype, with_replacement=False):
    def make_pair():
        input = torch.arange(n, dtype=dtype, device="cuda")
        return (
            lambda: ntops.torch.combinations(input, r=r, with_replacement=with_replacement),
            lambda: torch.combinations(input, r=r, with_replacement=with_replacement),
        )

    return make_pair


# ---- narrow (pure view) ----


def _make_narrow(shape, dim, start, length, dtype):
    def make_pair():
        if dtype.is_floating_point:
            input = torch.randn(shape, dtype=dtype, device="cuda")
        else:
            input = torch.randint(-128, 128, shape, dtype=dtype, device="cuda")
        return (
            lambda: ntops.torch.narrow(input, dim, start, length),
            lambda: torch.narrow(input, dim, start, length),
        )

    return make_pair


# ---- corrcoef ----


def _native_corrcoef(input):
    # CoreX has no f64 gemm, so the fair native baseline mirrors what ntops does there:
    # fall back to CPU for double when the GPU path is unsupported.
    try:
        return torch.corrcoef(input)
    except RuntimeError:
        return torch.corrcoef(input.cpu()).to(input.device)


def _make_corrcoef(shape, dtype):
    def make_pair():
        input = torch.randn(shape, dtype=dtype, device="cuda")
        return (
            lambda: ntops.torch.corrcoef(input),
            lambda: _native_corrcoef(input),
        )

    return make_pair


# ---- count_nonzero ----


def _make_count_nonzero(shape, dtype, dim=None):
    def make_pair():
        if dtype == torch.bool:
            input = torch.randint(0, 2, shape, device="cuda").bool()
        elif dtype.is_floating_point:
            input = (torch.randn(shape, device="cuda") > 0).to(dtype)
        else:
            input = torch.randint(0, 3, shape, dtype=dtype, device="cuda")
        return (
            lambda: ntops.torch.count_nonzero(input, dim=dim),
            lambda: torch.count_nonzero(input, dim=dim),
        )

    return make_pair


_LARGE = 1 << 24
_MID = 1 << 23


_PERF_CASES = [
    # ===================== kl_div =====================
    PerfCase("kl_div", "f32_large_none_1d", _make_kl_div((_LARGE,), torch.float32, "none")),
    PerfCase("kl_div", "f16_large_none_1d", _make_kl_div((_LARGE,), torch.float16, "none")),
    PerfCase("kl_div", "bf16_large_none_1d", _make_kl_div((_LARGE,), torch.bfloat16, "none"), rtol=2e-2, atol=2e-2),
    PerfCase("kl_div", "f32_large_none_2d", _make_kl_div((4096, 4096), torch.float32, "none")),
    PerfCase("kl_div", "f32_large_none_3d", _make_kl_div((256, 256, 256), torch.float32, "none")),
    PerfCase("kl_div", "f32_mid_none_1d", _make_kl_div((_MID,), torch.float32, "none")),
    PerfCase("kl_div", "f16_mid_none_1d", _make_kl_div((_MID,), torch.float16, "none")),
    PerfCase("kl_div", "f32_logtarget_none_1d", _make_kl_div((_LARGE,), torch.float32, "none", log_target=True)),
    PerfCase("kl_div", "f16_logtarget_none_1d", _make_kl_div((_LARGE,), torch.float16, "none", log_target=True)),
    PerfCase("kl_div", "f32_large_mean", _make_kl_div((_LARGE,), torch.float32, "mean")),
    PerfCase("kl_div", "f32_large_sum", _make_kl_div((_LARGE,), torch.float32, "sum")),
    PerfCase("kl_div", "f32_large_batchmean_2d", _make_kl_div((4096, 4096), torch.float32, "batchmean")),
    PerfCase("kl_div", "f16_large_mean", _make_kl_div((_LARGE,), torch.float16, "mean")),
    PerfCase("kl_div", "f32_noncontig_none", _make_kl_div((4096, 4096), torch.float32, "none", noncontig=True)),
    PerfCase("kl_div", "f16_noncontig_none", _make_kl_div((4096, 4096), torch.float16, "none", noncontig=True)),
    # ===================== combinations =====================
    PerfCase("combinations", "i64_n64_r2", _make_combinations(64, 2, torch.int64)),
    PerfCase("combinations", "i64_n91_r2", _make_combinations(91, 2, torch.int64)),
    PerfCase("combinations", "i64_n128_r2", _make_combinations(128, 2, torch.int64)),
    PerfCase("combinations", "f32_n64_r2", _make_combinations(64, 2, torch.float32)),
    PerfCase("combinations", "f32_n128_r2", _make_combinations(128, 2, torch.float32)),
    PerfCase("combinations", "f16_n128_r2", _make_combinations(128, 2, torch.float16)),
    PerfCase("combinations", "i32_n128_r2", _make_combinations(128, 2, torch.int32)),
    PerfCase("combinations", "f64_n128_r2", _make_combinations(128, 2, torch.float64)),
    PerfCase("combinations", "i64_n46_r2", _make_combinations(46, 2, torch.int64)),
    PerfCase("combinations", "f32_n91_r2", _make_combinations(91, 2, torch.float32)),
    PerfCase("combinations", "i64_n100_r2", _make_combinations(100, 2, torch.int64)),
    PerfCase("combinations", "i64_n256_r3", _make_combinations(256, 3, torch.int64)),
    PerfCase("combinations", "f32_n256_r3", _make_combinations(256, 3, torch.float32)),
    PerfCase("combinations", "i64_n64_r4", _make_combinations(64, 4, torch.int64)),
    PerfCase("combinations", "i64_n2048_r2_repl", _make_combinations(2048, 2, torch.int64, with_replacement=True)),
    PerfCase("combinations", "f32_n2048_r2_repl", _make_combinations(2048, 2, torch.float32, with_replacement=True)),
    PerfCase("combinations", "i64_n512_r3_repl", _make_combinations(512, 3, torch.int64, with_replacement=True)),
    PerfCase("combinations", "f64_n2048_r2", _make_combinations(2048, 2, torch.float64)),
    # ===================== narrow (pure view alias) =====================
    PerfCase("narrow", "f32_large_2d_dim1_half", _make_narrow((4096, 4096), 1, 0, 2048, torch.float32)),
    PerfCase("narrow", "f32_large_2d_dim0_half", _make_narrow((4096, 4096), 0, 0, 2048, torch.float32)),
    PerfCase("narrow", "f32_large_1d_half", _make_narrow((_LARGE,), 0, 0, _MID, torch.float32)),
    PerfCase("narrow", "f16_large_2d_dim1_half", _make_narrow((4096, 4096), 1, 0, 2048, torch.float16)),
    PerfCase("narrow", "f16_large_2d_dim0_half", _make_narrow((4096, 4096), 0, 0, 2048, torch.float16)),
    PerfCase("narrow", "bf16_large_2d_dim1_half", _make_narrow((4096, 4096), 1, 0, 2048, torch.bfloat16)),
    PerfCase("narrow", "f32_3d_dim2", _make_narrow((256, 256, 256), 2, 0, 128, torch.float32)),
    PerfCase("narrow", "f32_3d_dim1", _make_narrow((256, 256, 256), 1, 0, 128, torch.float32)),
    PerfCase("narrow", "f32_3d_dim0", _make_narrow((256, 256, 256), 0, 0, 128, torch.float32)),
    PerfCase("narrow", "i64_large_2d_dim1", _make_narrow((4096, 4096), 1, 0, 2048, torch.int64)),
    PerfCase("narrow", "i32_large_2d_dim0", _make_narrow((4096, 4096), 0, 0, 2048, torch.int32)),
    PerfCase("narrow", "f64_mid_2d_dim1", _make_narrow((2048, 2048), 1, 0, 1024, torch.float64)),
    PerfCase("narrow", "f32_large_2d_dim1_quarter", _make_narrow((4096, 4096), 1, 1024, 1024, torch.float32)),
    PerfCase("narrow", "f32_4d_dim2", _make_narrow((64, 64, 64, 64), 2, 0, 32, torch.float32)),
    PerfCase("narrow", "f16_3d_dim2", _make_narrow((256, 256, 256), 2, 0, 128, torch.float16)),
    # ===================== corrcoef =====================
    PerfCase("corrcoef", "f32_32x4096", _make_corrcoef((32, 4096), torch.float32)),
    PerfCase("corrcoef", "f32_64x4096", _make_corrcoef((64, 4096), torch.float32)),
    PerfCase("corrcoef", "f32_128x2048", _make_corrcoef((128, 2048), torch.float32)),
    PerfCase("corrcoef", "f32_256x1024", _make_corrcoef((256, 1024), torch.float32)),
    PerfCase("corrcoef", "f32_16x16384", _make_corrcoef((16, 16384), torch.float32)),
    PerfCase("corrcoef", "f32_512x512", _make_corrcoef((512, 512), torch.float32)),
    PerfCase("corrcoef", "f64_32x4096", _make_corrcoef((32, 4096), torch.float64)),
    PerfCase("corrcoef", "f64_64x2048", _make_corrcoef((64, 2048), torch.float64)),
    PerfCase("corrcoef", "f64_128x1024", _make_corrcoef((128, 1024), torch.float64)),
    PerfCase("corrcoef", "f32_64x8192", _make_corrcoef((64, 8192), torch.float32)),
    PerfCase("corrcoef", "f32_100x10000", _make_corrcoef((100, 10000), torch.float32)),
    PerfCase("corrcoef", "f32_8x65536", _make_corrcoef((8, 65536), torch.float32)),
    PerfCase("corrcoef", "f32_1d_65536", _make_corrcoef((65536,), torch.float32)),
    PerfCase("corrcoef", "f64_256x512", _make_corrcoef((256, 512), torch.float64)),
    PerfCase("corrcoef", "f32_512x1024", _make_corrcoef((512, 1024), torch.float32)),
    # ===================== count_nonzero =====================
    PerfCase("count_nonzero", "f32_large_1d_all", _make_count_nonzero((_LARGE,), torch.float32)),
    PerfCase("count_nonzero", "f32_large_2d_all", _make_count_nonzero((4096, 4096), torch.float32)),
    PerfCase("count_nonzero", "f32_large_3d_all", _make_count_nonzero((256, 256, 256), torch.float32)),
    PerfCase("count_nonzero", "i32_large_1d_all", _make_count_nonzero((_LARGE,), torch.int32)),
    PerfCase("count_nonzero", "i64_large_1d_all", _make_count_nonzero((_LARGE,), torch.int64)),
    PerfCase("count_nonzero", "bool_large_1d_all", _make_count_nonzero((_LARGE,), torch.bool)),
    PerfCase("count_nonzero", "f16_large_1d_all", _make_count_nonzero((_LARGE,), torch.float16)),
    PerfCase("count_nonzero", "f32_large_2d_dim0", _make_count_nonzero((4096, 4096), torch.float32, dim=0)),
    PerfCase("count_nonzero", "f32_large_2d_dim1", _make_count_nonzero((4096, 4096), torch.float32, dim=1)),
    PerfCase("count_nonzero", "i32_large_2d_dim0", _make_count_nonzero((4096, 4096), torch.int32, dim=0)),
    PerfCase("count_nonzero", "i64_large_2d_dim1", _make_count_nonzero((4096, 4096), torch.int64, dim=1)),
    PerfCase("count_nonzero", "f32_3d_dim2", _make_count_nonzero((256, 256, 256), torch.float32, dim=2)),
    PerfCase("count_nonzero", "f32_3d_dim01", _make_count_nonzero((256, 256, 256), torch.float32, dim=(0, 1))),
    PerfCase("count_nonzero", "i32_mid_1d_all", _make_count_nonzero((_MID,), torch.int32)),
    PerfCase("count_nonzero", "bool_large_2d_dim0", _make_count_nonzero((4096, 4096), torch.bool, dim=0)),
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
