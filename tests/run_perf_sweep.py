"""Standalone perf sweep driver for T1-1-8.

Runs every PerfCase in tests/t1_1_8_performance_utils.py, measures
ratio = torch_ms / ntops_ms, checks correctness, and emits a JSON artifact
matching docs/性能测试/T1-1-8/artifacts/*.json.

Usage:
    PYTHONPATH=src python tests/run_perf_sweep.py --out /tmp/iluvatar_t1_1_8.json
    PYTHONPATH=src python tests/run_perf_sweep.py --op count_nonzero --out /tmp/cn.json
"""

import argparse
import json
import math
import sys

import torch

from tests.t1_1_8_performance_utils import (
    PERF_THRESHOLD,
    _PERF_CASES,
    _assert_outputs_match,
    _assert_shapes_match,
    _time_cuda,
)


def _run_case(case):
    ntops_call, torch_call = case.make_pair()
    correct = True
    err = None
    try:
        ntops_output = ntops_call()
        reference = torch_call()
        if case.compare:
            _assert_outputs_match(ntops_output, reference, rtol=case.rtol, atol=case.atol)
        else:
            _assert_shapes_match(ntops_output, reference)
    except Exception as exc:  # noqa: BLE001 - record correctness failures as data
        correct = False
        err = f"{type(exc).__name__}: {exc}"

    record = {
        "op": case.op_name,
        "case": case.case_name,
        "correct": correct,
    }
    if not correct:
        record.update({"ntops_ms": None, "torch_ms": None, "ratio": None, "status": "error", "error": err})
        return record

    try:
        ntops_ms = _time_cuda(ntops_call)
        torch_ms = _time_cuda(torch_call)
    except Exception as exc:  # noqa: BLE001
        record.update({"ntops_ms": None, "torch_ms": None, "ratio": None, "status": "error", "error": f"timing: {exc}"})
        return record

    ratio = torch_ms / ntops_ms if ntops_ms > 0 else float("inf")
    record.update(
        {
            "ntops_ms": round(ntops_ms, 6),
            "torch_ms": round(torch_ms, 6),
            "ratio": round(ratio, 4),
            "status": "pass" if ratio >= PERF_THRESHOLD else "below",
        }
    )
    return record


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", required=True)
    parser.add_argument("--op", default=None, help="filter by op name (repeatable, comma separated)")
    parser.add_argument("--case", default=None, help="substring filter on case name")
    args = parser.parse_args()

    if not torch.cuda.is_available():
        print("CUDA not available", file=sys.stderr)
        sys.exit(1)

    dev_name = torch.cuda.get_device_name(0)
    print(f"device: {dev_name}")

    ops = set(args.op.split(",")) if args.op else None
    cases = [
        c
        for c in _PERF_CASES
        if (ops is None or c.op_name in ops) and (args.case is None or args.case in c.case_name)
    ]

    results = []
    for case in cases:
        rec = _run_case(case)
        results.append(rec)
        r = rec["ratio"]
        rstr = f"{r:.3f}x" if isinstance(r, (int, float)) else "ERR"
        print(f"{rec['op']}/{rec['case']}: {rec['status']:5s} ratio={rstr} correct={rec['correct']}")

    with open(args.out, "w") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    # per-op summary
    print("\n=== summary ===")
    by_op = {}
    for rec in results:
        by_op.setdefault(rec["op"], []).append(rec)
    total_bad = 0
    for op, recs in by_op.items():
        ratios = [r["ratio"] for r in recs if isinstance(r["ratio"], (int, float)) and r["ratio"] > 0]
        bad = [r for r in recs if r["status"] != "pass"]
        total_bad += len(bad)
        geomean = math.prod(ratios) ** (1.0 / len(ratios)) if ratios else 0.0
        mn = min(ratios) if ratios else 0.0
        print(f"{op}: {len(recs)} cases, {len(bad)} bad, min={mn:.3f}, geomean={geomean:.3f}")
        for b in bad:
            print(f"    BAD {b['case']}: status={b['status']} ratio={b['ratio']}")
    print(f"TOTAL: {len(results)} cases, {total_bad} below/error")
    print(f"artifact: {args.out}")


if __name__ == "__main__":
    main()
