from dataclasses import dataclass

import torch
import torch.nn.functional as F

import ntops
from tests.skippers import skip_if_cuda_not_available


WARMUP = 20
ITERATIONS = 50
PERF_THRESHOLD = 0.95


@dataclass(frozen=True)
class PerfCase:
    name: str
    shape: tuple[int, int, int, int]
    kernel_size: tuple[int, int]
    dilation: tuple[int, int]
    padding: tuple[int, int]
    stride: tuple[int, int]
    dtype: torch.dtype
    noncontiguous: bool = False


def _make_input(shape, dtype, noncontiguous=False):
    if dtype == torch.bool:
        input = torch.randint(0, 2, shape, dtype=dtype, device="cuda")
    elif dtype.is_floating_point:
        input = torch.randn(shape, dtype=dtype, device="cuda")
    else:
        input = torch.randint(0, 100, shape, dtype=dtype, device="cuda")

    if noncontiguous:
        input = input.transpose(2, 3)

    return input


def _profile(fn, warmup=WARMUP, iterations=ITERATIONS):
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


_PERF_CASES = (
    PerfCase("large_f32_8x64x128x128_k3", (8, 64, 128, 128), (3, 3), (1, 1), (1, 1), (1, 1), torch.float32),
    PerfCase("large_f16_8x64x128x128_k3", (8, 64, 128, 128), (3, 3), (1, 1), (1, 1), (1, 1), torch.float16),
    PerfCase("large_bf16_8x64x128x128_k3", (8, 64, 128, 128), (3, 3), (1, 1), (1, 1), (1, 1), torch.bfloat16),
    PerfCase("large_f32_8x128x128x64_k3", (8, 128, 128, 64), (3, 3), (1, 1), (1, 1), (1, 1), torch.float32),
    PerfCase("large_f16_8x128x128x64_k3", (8, 128, 128, 64), (3, 3), (1, 1), (1, 1), (1, 1), torch.float16),
    PerfCase("mid_f32_16x32x64x64_k5_s2", (16, 32, 64, 64), (5, 5), (1, 1), (2, 2), (2, 2), torch.float32),
    PerfCase("mid_f16_16x32x64x64_k5_s2", (16, 32, 64, 64), (5, 5), (1, 1), (2, 2), (2, 2), torch.float16),
    PerfCase("mid_f32_k3_s1", (8, 48, 96, 96), (3, 3), (1, 1), (1, 1), (1, 1), torch.float32),
    PerfCase("mid_f32_dilated", (4, 32, 96, 80), (3, 3), (2, 2), (2, 2), (1, 1), torch.float32),
    PerfCase("mid_f16_dilated", (4, 32, 96, 80), (3, 3), (2, 2), (2, 2), (1, 1), torch.float16),
    PerfCase("small_f32_32x16x32x32_k1", (32, 16, 32, 32), (1, 1), (1, 1), (0, 0), (1, 1), torch.float32),
    PerfCase("small_f16_32x16x32x32_k1", (32, 16, 32, 32), (1, 1), (1, 1), (0, 0), (1, 1), torch.float16),
    PerfCase("small_f32_16x32x16x16_k3_p1", (16, 32, 16, 16), (3, 3), (1, 1), (1, 1), (1, 1), torch.float32),
    PerfCase("small_bool_16x8x64x64_k3", (16, 8, 64, 64), (3, 3), (1, 1), (1, 1), (1, 1), torch.bool),
    PerfCase("small_f32_8x16x8x8_k5_d2", (8, 16, 8, 8), (5, 5), (2, 2), (4, 4), (1, 1), torch.float32),
    PerfCase("noncontig_f32_8x32x96x80_k3", (8, 32, 96, 80), (3, 3), (1, 1), (1, 1), (1, 1), torch.float32, True),
    PerfCase("noncontig_f16_8x32x96x80_k3", (8, 32, 96, 80), (3, 3), (1, 1), (1, 1), (1, 1), torch.float16, True),
    PerfCase("large_f32_4x128x128x128_k3_str2", (4, 128, 128, 128), (3, 3), (1, 1), (1, 1), (2, 2), torch.float32),
    PerfCase("mid_f32_16x32x32x32_k5_p2", (16, 32, 32, 32), (5, 5), (1, 1), (2, 2), (1, 1), torch.float32),
    PerfCase("small_bf16_32x16x32x32_k1", (32, 16, 32, 32), (1, 1), (1, 1), (0, 0), (1, 1), torch.bfloat16),
)


@skip_if_cuda_not_available
def test_im2col_performance_all():
    failures = []
    ratios = []

    for case in _PERF_CASES:
        input = _make_input(case.shape, case.dtype, case.noncontiguous)

        def ntops_fn():
            return ntops.torch.im2col(
                input,
                kernel_size=case.kernel_size,
                dilation=case.dilation,
                padding=case.padding,
                stride=case.stride,
            )

        def torch_fn():
            return F.unfold(
                input,
                kernel_size=case.kernel_size,
                dilation=case.dilation,
                padding=case.padding,
                stride=case.stride,
            )

        output = ntops_fn()
        reference = torch_fn()
        if case.dtype == torch.bool:
            assert torch.equal(output, reference)
        elif case.dtype.is_floating_point:
            assert torch.allclose(output, reference, rtol=2e-3, atol=2e-3)
        else:
            assert torch.equal(output, reference)

        ntops_ms = _profile(ntops_fn)
        torch_ms = _profile(torch_fn)
        ratio = torch_ms / ntops_ms if ntops_ms > 0 else float("inf")
        ratios.append(ratio)

        print(
            f"im2col/{case.name}: ntops={ntops_ms:.4f} ms, "
            f"torch={torch_ms:.4f} ms, torch/ntops={ratio:.3f}x"
        )

        if ratio < PERF_THRESHOLD:
            failures.append((case.name, ratio))

    geomean = torch.tensor(ratios).log().mean().exp().item()
    print(f"\nOverall geomean ratio: {geomean:.3f}x ({len(_PERF_CASES)} cases)")

    if failures:
        failure_text = "; ".join(f"{name}={ratio:.3f}x" for name, ratio in failures)
        assert False, f"im2col performance below {PERF_THRESHOLD}: {failure_text}"
