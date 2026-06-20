import torch

import ntops
from tests.skippers import skip_if_cuda_not_available


WARMUP = 30
ITERATIONS = 100
PERF_THRESHOLD = 0.95


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

    torch.cuda.synchronize()
    return start.elapsed_time(end) / iterations


_PERF_CASES = (
    ("2D_f32", (4096, 4096), 0, 1, torch.float32),
    ("2D_f16", (4096, 4096), 0, 1, torch.float16),
    ("2D_int32", (4096, 4096), 0, 1, torch.int32),
    ("2D_8k_f32", (8192, 8192), 0, 1, torch.float32),
    ("2D_8k_f16", (8192, 8192), 0, 1, torch.float16),
    ("3D_f32", (512, 512, 256), 0, 2, torch.float32),
    ("3D_f16", (512, 512, 256), 0, 2, torch.float16),
    ("3D_int32", (512, 512, 256), 0, 2, torch.int32),
    ("3D_wide_f32", (256, 512, 512), 1, 0, torch.float32),
    ("3D_roll_f32", (512, 512, 256), 1, 2, torch.float32),
    ("3D_reverse_f32", (512, 512, 256), (0, 2), (2, 0), torch.float32),
    ("3D_512_f32", (512, 512, 128), 0, 2, torch.float32),
    ("4D_f32", (256, 128, 128, 32), 0, 3, torch.float32),
    ("4D_f16", (256, 128, 128, 32), 0, 3, torch.float16),
    ("4D_swap_f32", (256, 128, 128, 32), 1, 2, torch.float32),
    ("4D_wide_f32", (128, 128, 128, 128), 0, 3, torch.float32),
    ("4D_multi_f32", (128, 128, 64, 64), (0, 1), (2, 3), torch.float32),
    ("4D_batchswap_f32", (128, 256, 64, 64), 1, 2, torch.float32),
    ("3D_reverse_f16", (512, 512, 128), 0, 2, torch.float16),
    ("3D_roll_int32", (512, 512, 128), 1, 2, torch.int32),
)


@skip_if_cuda_not_available
def test_moveaxis_performance_all():
    failures = []
    ratios = []

    for case in _PERF_CASES:
        name = case[0]
        shape = case[1]
        src = case[2]
        dst = case[3]
        dtype = case[4]

        if dtype == torch.bool:
            input_data = torch.randint(0, 2, shape, dtype=torch.bool, device="cuda")
        elif dtype.is_floating_point:
            input_data = torch.randn(shape, dtype=dtype, device="cuda")
        else:
            input_data = torch.randint(0, 100, shape, dtype=dtype, device="cuda")

        def ntops_fn():
            return ntops.torch.moveaxis(input_data, src, dst)

        def torch_fn():
            return torch.moveaxis(input_data, src, dst).contiguous()

        output = ntops_fn()
        reference = torch_fn()
        assert output.shape == reference.shape
        assert output.dtype == reference.dtype

        ms_ntops = _profile(ntops_fn)
        ms_torch = _profile(torch_fn)
        ratio = ms_torch / ms_ntops if ms_ntops > 0 else float("inf")
        ratios.append(ratio)

        print(
            f"moveaxis/{name}: ntops={ms_ntops:.4f} ms, "
            f"torch={ms_torch:.4f} ms, torch/ntops={ratio:.3f}x"
        )

        if ratio < PERF_THRESHOLD:
            failures.append((name, ratio))

    geomean = torch.tensor(ratios).log().mean().exp().item()
    print(
        f"\nResults: {len(_PERF_CASES) - len(failures)}/{len(_PERF_CASES)} passed, "
        f"geomean={geomean:.3f}x"
    )

    if failures:
        fail_str = "; ".join(f"{n}={r:.3f}x" for n, r in failures)
        assert False, f"Performance below threshold ({PERF_THRESHOLD}): {fail_str}"
