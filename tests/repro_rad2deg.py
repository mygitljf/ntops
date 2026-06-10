import torch

import ntops


def bench(fn, warmup=20, iterations=100):
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


shape = (1 << 22,)
dtype = torch.float32
input = torch.randn(shape, dtype=dtype, device="cuda")
nt_out = torch.empty_like(input)
th_out = torch.empty_like(input)

ntops_ms = bench(lambda: ntops.torch.rad2deg(input, out=nt_out))
torch_ms = bench(lambda: torch.rad2deg(input, out=th_out))
print(f"rad2deg f32_mid_1d: ntops={ntops_ms:.4f} ms, torch={torch_ms:.4f} ms, torch/ntops={torch_ms / ntops_ms:.3f}x")

ntops.torch.rad2deg(input, out=nt_out)
torch.cuda.synchronize()
