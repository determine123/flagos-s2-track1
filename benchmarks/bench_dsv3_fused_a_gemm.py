"""Benchmark dsv3_fused_a_gemm: torch baseline vs. the Triton kernel.

Run on a CUDA box:

    python benchmarks/bench_dsv3_fused_a_gemm.py
    python benchmarks/bench_dsv3_fused_a_gemm.py --json results.json

Three implementations are timed per configuration:

    torch_noncontig   torch.matmul with mat_b as the transposed view (baseline)
    torch_contig      torch.matmul after materialising a contiguous mat_b
    triton            the integrated kernel, once KERNEL_AVAILABLE is True

``torch_contig`` uses a pre-materialised weight: its timing excludes the
contiguous copy. It is a layout control, never a copy-inclusive baseline.

Every implementation is timed through the **returning** interface --
``dsv3_fused_a_gemm(mat_a, mat_b)`` and ``torch.matmul(mat_a, mat_b)`` -- because
that is the competition contract. The caller does not supply an output buffer, so
the output allocation is part of the measured cost for all three, and the
comparison stays apples-to-apples.

The headline configuration is the DeepSeek-V3 shape (K=7168, N=2112) with the
focus token counts M in {1, 2, 4, 8, 16}. That shape is a specialisation target,
not a correctness limit -- pass ``--sweep`` to measure the wider aligned range.

No CPU timings are produced. If no CUDA device is present the script says so and
exits rather than emitting numbers that would not mean anything.
"""

from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

import torch  # noqa: E402

from dsv3_fused_a_gemm import kernel as kernel_mod  # noqa: E402
from dsv3_fused_a_gemm.reference import (  # noqa: E402
    DSV3_FUSED_A_N,
    DSV3_HIDDEN,
    DSV3_SHAPES,
    K_ALIGNMENT,
    N_ALIGNMENT,
)

sys.path.insert(0, os.path.dirname(__file__))
from performance_utils import (  # noqa: E402
    BenchResult,
    bench,
    format_table,
    has_triton,
)

#: Wider sweep over the aligned correctness range, used by ``--sweep``.
SWEEP_SHAPES = [
    (1, K_ALIGNMENT, N_ALIGNMENT),
    (16, K_ALIGNMENT, N_ALIGNMENT),
    (1, K_ALIGNMENT * 4, N_ALIGNMENT * 8),
    (16, K_ALIGNMENT * 4, N_ALIGNMENT * 8),
    (1, DSV3_HIDDEN, DSV3_FUSED_A_N),
    (16, DSV3_HIDDEN, DSV3_FUSED_A_N),
]


def build_operands(m: int, k: int, n: int, dtype: torch.dtype, device: str, seed: int = 0):
    """Build ``(mat_a, mat_b_strided, mat_b_contig)`` on ``device``."""
    gen = torch.Generator(device=device).manual_seed(seed)
    mat_a = torch.randn(m, k, dtype=dtype, device=device, generator=gen)
    weight = torch.randn(n, k, dtype=dtype, device=device, generator=gen)
    mat_b_strided = weight.T  # [K, N] with strides (1, K) -- non-contiguous
    mat_b_contig = weight.T.contiguous()
    return mat_a, mat_b_strided, mat_b_contig


def run(
    dtype: torch.dtype,
    flush_l2: bool,
    warmup: int,
    rep: int,
    configs: list[tuple[int, int, int]],
) -> list[BenchResult]:
    device = "cuda"
    results: list[BenchResult] = []

    for m, k, n in configs:
        mat_a, mat_b_strided, mat_b_contig = build_operands(m, k, n, dtype, device)
        assert not mat_b_strided.is_contiguous(), "baseline must be the strided path"

        meta = {"k": k, "n": n}

        # --- torch baseline: non-contiguous mat_b (the real contract) -------
        # Returning interface, so the output allocation is included -- exactly
        # as it is for the kernel below.
        ms = bench(
            lambda: torch.matmul(mat_a, mat_b_strided),
            warmup=warmup,
            rep=rep,
            flush_l2=flush_l2,
        )
        results.append(
            BenchResult("torch_noncontig", m, dtype, ms, "torch_noncontig", dict(meta))
        )

        # --- torch with a materialised contiguous weight --------------------
        ms = bench(
            lambda: torch.matmul(mat_a, mat_b_contig),
            warmup=warmup,
            rep=rep,
            flush_l2=flush_l2,
        )
        results.append(
            BenchResult("torch_contig", m, dtype, ms, "torch_contig", dict(meta))
        )

        # --- Triton kernel ---------------------------------------------------
        if kernel_mod.KERNEL_AVAILABLE:
            ms = bench(
                lambda: kernel_mod.dsv3_fused_a_gemm(mat_a, mat_b_strided),
                warmup=warmup,
                rep=rep,
                flush_l2=flush_l2,
            )
            results.append(BenchResult("triton", m, dtype, ms, "triton", dict(meta)))

    return results


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dtype", default="bfloat16", choices=["bfloat16"])
    parser.add_argument("--json", default=None, help="write raw results to this path")
    parser.add_argument("--warmup", type=int, default=25)
    parser.add_argument("--rep", type=int, default=100)
    parser.add_argument(
        "--sweep",
        action="store_true",
        help=(
            "also measure the wider aligned shape range, not just the DSV3 "
            "headline shape"
        ),
    )
    parser.add_argument(
        "--no-flush-l2",
        action="store_true",
        help="leave L2 warm; measures the cache-resident case instead",
    )
    args = parser.parse_args()

    if not torch.cuda.is_available():
        print(
            "No CUDA device detected.\n"
            "This benchmark deliberately reports no CPU numbers: the operator is a\n"
            "latency-critical decode-path GEMM, so a CPU timing would not be\n"
            "meaningful. Re-run on the cloud GPU box.",
            file=sys.stderr,
        )
        return 2

    if not has_triton():
        print(
            "warning: Triton is not importable; only PyTorch controls will run.",
            file=sys.stderr,
        )

    dtype = getattr(torch, args.dtype)
    flush_l2 = not args.no_flush_l2

    configs = [(m, DSV3_HIDDEN, DSV3_FUSED_A_N) for m in DSV3_SHAPES]
    if args.sweep:
        configs += [c for c in SWEEP_SHAPES if c not in configs]

    results = run(
        dtype, flush_l2=flush_l2, warmup=args.warmup, rep=args.rep, configs=configs
    )

    print(f"\n=== dsv3_fused_a_gemm benchmark ({args.dtype}) ===")
    print("interface: dsv3_fused_a_gemm(mat_a, mat_b) -> Tensor (returning form)")
    print(f"l2 flush: {flush_l2}   warmup={args.warmup} rep={args.rep}")
    if not kernel_mod.KERNEL_AVAILABLE:
        print("NOTE: Triton kernel not integrated -- only torch paths measured.")
    print()
    print(format_table(results))

    if args.json:
        payload = [
            {
                "impl": r.impl,
                "m": r.m,
                "k": r.meta.get("k"),
                "n": r.meta.get("n"),
                "dtype": str(r.dtype),
                "ms": r.ms,
            }
            for r in results
        ]
        with open(args.json, "w", encoding="utf-8") as fh:
            json.dump(payload, fh, indent=2)
        print(f"\nwrote {args.json}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
