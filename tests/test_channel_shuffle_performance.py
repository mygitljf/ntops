from dataclasses import dataclass

import torch

import ntops
from tests.skippers import skip_if_cuda_not_available


PERF_THRESHOLD = 0.95
WARMUP = 10
ITERATIONS = 30


def _is_integral(dtype):
    return dtype in (
        torch.int8,
        torch.int16,
        torch.int32,
        torch.int64,
        torch.uint8,
        torch.bool,
    )


def _make_tensor(shape, dtype):
    if _is_integral(dtype):
        if dtype == torch.bool:
            return torch.randint(0, 2, shape, dtype=dtype, device="cuda")
        return torch.randint(-100, 100, shape, dtype=dtype, device="cuda")
    return torch.randn(shape, dtype=dtype, device="cuda")


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


def _assert_outputs_match(output, reference):
    if reference.dtype.is_floating_point:
        assert torch.allclose(output, reference, rtol=2e-3, atol=2e-3, equal_nan=True)
    else:
        assert torch.equal(output, reference)


@dataclass(frozen=True)
class PerfCase:
    name: str
    shape: tuple
    groups: int
    dtype: torch.dtype
    layout: str = "contiguous"


_PERF_CASES = [
    PerfCase("nchw_f32_g2", (64, 64, 128, 128), 2, torch.float32),
    PerfCase("nchw_f32_g4", (64, 64, 128, 128), 4, torch.float32),
    PerfCase("nchw_f32_g8", (64, 64, 128, 128), 8, torch.float32),
    PerfCase("nchw_f16_g2", (64, 64, 128, 128), 2, torch.float16),
    PerfCase("nchw_f16_g8", (64, 64, 128, 128), 8, torch.float16),
    PerfCase("nchw_bf16_g4", (32, 64, 128, 128), 4, torch.bfloat16),
    PerfCase("nchw_f64_g2", (16, 64, 128, 128), 2, torch.float64),
    PerfCase("nchw_i32_g4", (32, 64, 128, 128), 4, torch.int32),
    PerfCase("nchw_i64_g2", (16, 64, 128, 128), 2, torch.int64),
    PerfCase("nchw_bool_g2", (64, 64, 128, 128), 2, torch.bool),
    PerfCase("ncl_f32_g2", (256, 64, 8192), 2, torch.float32),
    PerfCase("ncl_f16_g4", (256, 64, 8192), 4, torch.float16),
    PerfCase("ncdhw_f32_g4", (8, 64, 32, 64, 64), 4, torch.float32),
    PerfCase("small_f32_g2", (512, 32, 32, 32), 2, torch.float32),
    PerfCase("small_f16_g4", (512, 32, 32, 32), 4, torch.float16),
    PerfCase("wide_channels_f32_g16", (16, 256, 64, 64), 16, torch.float32),
    PerfCase("narrow_channels_f32_g2", (128, 8, 256, 256), 2, torch.float32),
    PerfCase("prime_tail_f32_g4", (128, 32, 127, 129), 4, torch.float32),
    PerfCase("channels_last_f32_g4", (64, 64, 128, 128), 4, torch.float32, "channels_last"),
    PerfCase("channels_last_f16_g4", (64, 64, 128, 128), 4, torch.float16, "channels_last"),
]


def _make_input(case):
    input = _make_tensor(case.shape, case.dtype)
    if case.layout == "channels_last":
        return input.contiguous(memory_format=torch.channels_last)
    return input


@skip_if_cuda_not_available
def test_channel_shuffle_performance_all():
    failures = []
    ratios = []
    for case in _PERF_CASES:
        input = _make_input(case)

        def ntops_fn():
            return ntops.torch.channel_shuffle(input, case.groups)

        def torch_fn():
            return torch.channel_shuffle(input, case.groups)

        ntops_output = ntops_fn()
        ref_output = torch_fn()
        _assert_outputs_match(ntops_output, ref_output)

        ntops_ms = _time_cuda(ntops_fn)
        torch_ms = _time_cuda(torch_fn)
        ratio = torch_ms / ntops_ms if ntops_ms > 0 else float("inf")
        ratios.append(ratio)

        print(
            f"channel_shuffle/{case.name}: ntops={ntops_ms:.4f} ms, "
            f"torch={torch_ms:.4f} ms, torch/ntops={ratio:.3f}x"
        )

        if ratio < PERF_THRESHOLD:
            failures.append((case.name, ratio))

    if failures:
        fail_str = "; ".join(f"{name}={ratio:.3f}x" for name, ratio in failures)
        assert False, f"Performance below threshold ({PERF_THRESHOLD}): {fail_str}"

    print(
        f"\nAll {len(_PERF_CASES)} cases passed "
        f"(ratio >= {PERF_THRESHOLD}, mean={sum(ratios) / len(ratios):.3f}x)"
    )
