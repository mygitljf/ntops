from dataclasses import dataclass
from time import perf_counter

import torch

import ntops
from tests.skippers import skip_if_cuda_not_available


_LARGE_NUMEL = 1 << 24
_MID_NUMEL = _LARGE_NUMEL
_MIN_TORCH_SPEED_RATIO = 0.9
_FAIR_MIN_RATIO = 0.7
_TARGET_OVERALL_TORCH_SPEED_RATIO = 1.5


def _is_integral(dtype):
    return dtype in (torch.int8, torch.int16, torch.int32, torch.int64,
                     torch.uint8, torch.bool)


def _make_tensor(shape, dtype, device="cuda"):
    if _is_integral(dtype):
        return torch.randint(-100, 100, shape, dtype=dtype, device=device)
    return torch.randn(shape, dtype=dtype, device=device)


def _make_noncontig_input(side, dtype):
    """Create a non-contiguous transposed tensor."""
    if _is_integral(dtype):
        return torch.randint(-100, 100, (side, side), dtype=dtype, device="cuda").t()
    return torch.randn((side, side), dtype=dtype, device="cuda").t()


def _make_permuted_input(shape, dtype):
    """Create a 3D permuted non-contiguous tensor."""
    return _make_tensor(shape, dtype).permute(2, 0, 1)


@dataclass(frozen=True)
class PerfCase:
    case_name: str
    make_pair: object


def _make_contiguous_split_pair(shape, dtype, sections=2, dim=0):
    """Pair for contiguous tensor split (view-based for both impls)."""
    def make_pair():
        input = _make_tensor(shape, dtype)

        def ntops_fn():
            return ntops.torch.tensor_split(input, sections, dim=dim)

        def torch_fn():
            return torch.tensor_split(input, sections, dim=dim)

        return ntops_fn, torch_fn

    return make_pair


def _make_noncontig_split_pair(side, dtype, sections=2, dim=0):
    """Materialization cost benchmark: ntops view vs torch contiguous()+view."""
    def make_pair():
        input = _make_noncontig_input(side, dtype)

        def ntops_fn():
            return ntops.torch.tensor_split(input, sections, dim=dim)

        def torch_fn():
            contig = input.contiguous()
            return torch.tensor_split(contig, sections, dim=dim)

        return ntops_fn, torch_fn

    return make_pair


def _make_noncontig_fair_split_pair(side, dtype, sections=2, dim=0):
    """Fair comparison: ntops vs torch both on non-contiguous input.

    Unlike _make_noncontig_split_pair (which benchmarks materialization cost
    avoidance), this pair compares both implementations on the exact same
    non-contiguous input — neither side forces a contiguous() copy.
    """
    def make_pair():
        input = _make_noncontig_input(side, dtype)

        def ntops_fn():
            return ntops.torch.tensor_split(input, sections, dim=dim)

        def torch_fn():
            return torch.tensor_split(input, sections, dim=dim)

        return ntops_fn, torch_fn

    return make_pair


def _make_permute3d_split_pair(shape, dtype, sections=2, dim=1):
    """Pair for 3D permuted non-contiguous split."""
    def make_pair():
        input = _make_permuted_input(shape, dtype)

        def ntops_fn():
            return ntops.torch.tensor_split(input, sections, dim=dim)

        def torch_fn():
            contig = input.contiguous()
            return torch.tensor_split(contig, sections, dim=dim)

        return ntops_fn, torch_fn

    return make_pair


_PERF_CASES = [
    PerfCase("f32_noncontig_4096_dim0_split2", _make_noncontig_split_pair(4096, torch.float32, 2, dim=0)),
    PerfCase("f16_noncontig_4096_dim0_split2", _make_noncontig_split_pair(4096, torch.float16, 2, dim=0)),
    PerfCase("f64_noncontig_4096_dim0_split2", _make_noncontig_split_pair(4096, torch.float64, 2, dim=0)),
    PerfCase("i32_noncontig_4096_dim0_split2", _make_noncontig_split_pair(4096, torch.int32, 2, dim=0)),
    PerfCase("i64_noncontig_4096_dim0_split2", _make_noncontig_split_pair(4096, torch.int64, 2, dim=0)),
    PerfCase("f32_noncontig_4096_dim0_split4", _make_noncontig_split_pair(4096, torch.float32, 4, dim=0)),
    PerfCase("f32_noncontig_4096_dim0_split8", _make_noncontig_split_pair(4096, torch.float32, 8, dim=0)),
    PerfCase("f32_noncontig_4096_dim1_split2", _make_noncontig_split_pair(4096, torch.float32, 2, dim=1)),
    PerfCase("f64_noncontig_4096_dim1_split2", _make_noncontig_split_pair(4096, torch.float64, 2, dim=1)),
    PerfCase("f32_permute3d_256x256x128_dim1_split2",
             _make_permute3d_split_pair((256, 256, 128), torch.float32, 2, dim=1)),
    PerfCase("f16_permute3d_256x256x128_dim1_split2",
             _make_permute3d_split_pair((256, 256, 128), torch.float16, 2, dim=1)),
    PerfCase("f32_noncontig_8192x4096_dim0_split2", _make_noncontig_split_pair(8192, torch.float32, 2, dim=0)),
    PerfCase("FAIR_f32_noncontig_4096_dim0_split2", _make_noncontig_fair_split_pair(4096, torch.float32, 2, dim=0)),
    PerfCase("FAIR_f16_noncontig_4096_dim0_split2", _make_noncontig_fair_split_pair(4096, torch.float16, 2, dim=0)),
    PerfCase("FAIR_f64_noncontig_4096_dim0_split2", _make_noncontig_fair_split_pair(4096, torch.float64, 2, dim=0)),
    PerfCase("FAIR_i32_noncontig_4096_dim0_split2", _make_noncontig_fair_split_pair(4096, torch.int32, 2, dim=0)),
    PerfCase("FAIR_f32_noncontig_4096_dim0_split8", _make_noncontig_fair_split_pair(4096, torch.float32, 8, dim=0)),
    PerfCase("FAIR_f32_noncontig_4096_dim1_split2", _make_noncontig_fair_split_pair(4096, torch.float32, 2, dim=1)),
    PerfCase("FAIR_f32_noncontig_2048_dim0_split4", _make_noncontig_fair_split_pair(2048, torch.float32, 4, dim=0)),
    PerfCase("FAIR_i64_noncontig_4096_dim0_split2", _make_noncontig_fair_split_pair(4096, torch.int64, 2, dim=0)),
]


def _time_cuda(fn, warmup=20, iterations=200):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()

    start = perf_counter()
    for _ in range(iterations):
        fn()
    torch.cuda.synchronize()
    end = perf_counter()
    return (end - start) * 1000 / iterations


def _assert_outputs_match(output, reference):
    """Compare tuple outputs from tensor_split."""
    assert len(output) == len(reference)
    for out, ref in zip(output, reference):
        if ref.dtype.is_floating_point:
            assert torch.allclose(out, ref, rtol=2e-3, atol=2e-3, equal_nan=True)
        else:
            assert torch.equal(out, ref)


@skip_if_cuda_not_available
def test_tensor_split_performance_all():
    failures = []
    ratios = []
    for case in _PERF_CASES:
        ntops_fn, torch_fn = case.make_pair()

        ntops_output = ntops_fn()
        ref_output = torch_fn()
        _assert_outputs_match(ntops_output, ref_output)

        ntops_ms = _time_cuda(ntops_fn)
        torch_ms = _time_cuda(torch_fn)
        ratio = torch_ms / ntops_ms if ntops_ms > 0 else float("inf")
        ratios.append(ratio)
        threshold = _FAIR_MIN_RATIO if case.case_name.startswith("FAIR_") else _MIN_TORCH_SPEED_RATIO
        print(
            f"tensor_split/{case.case_name}: ntops={ntops_ms:.4f} ms, "
            f"torch={torch_ms:.4f} ms, torch/ntops={ratio:.3f}x"
        )
        if ratio <= threshold:
            failures.append((case.case_name, ratio))

    if failures:
        fail_str = "; ".join(f"{n}={r:.3f}x" for n, r in failures)
        assert False, f"Performance not above threshold ({_MIN_TORCH_SPEED_RATIO}): {fail_str}"

    overall_ratio = sum(ratios) / len(ratios)
    print(f"\nOverall arithmetic mean ratio: {overall_ratio:.3f}x")
    assert overall_ratio > _TARGET_OVERALL_TORCH_SPEED_RATIO, (
        f"Overall ratio not above target ({_TARGET_OVERALL_TORCH_SPEED_RATIO}): "
        f"{overall_ratio:.3f}x"
    )

    print(f"All {len(_PERF_CASES)} cases passed (ratio > {_MIN_TORCH_SPEED_RATIO})")
