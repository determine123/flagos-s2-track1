"""Minimal benchmark harness for the dsv3_fused_a_gemm project.

Kept dependency-light and device-aware: it uses CUDA events when
a CUDA device is present and refuses to fabricate numbers when one is not. Every
timing produced here is a *measured* value; nothing is estimated or carried over
from documentation.

Note on small-M GEMMs: the kernel under test targets the decode regime (M=1..16),
where total runtime is tens of microseconds. Launch overhead is therefore a
first-class part of the measurement, not noise to be averaged away, which is why
the harness reports the median and keeps any
L2-cache flushing decisions explicit in the caller.
"""

from __future__ import annotations

import statistics
from dataclasses import dataclass, field
from typing import Callable, Sequence

import torch

__all__ = ["BenchResult", "bench", "cuda_sync", "format_table", "speedup"]

#: (median, low, high) quantiles reported for every measurement.
QUANTILES: tuple[float, ...] = (0.5, 0.2, 0.8)


def cuda_sync() -> None:
    """Synchronise the default CUDA stream, if one exists."""
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def has_triton() -> bool:
    try:
        import triton  # noqa: F401

        return True
    except ImportError:
        return False


def bench(
    fn: Callable[[], object],
    warmup: int = 25,
    rep: int = 100,
    flush_l2: bool = True,
) -> float:
    """Return median CUDA-event milliseconds per call of ``fn``.

    Args:
        fn: zero-argument callable performing one invocation.
        warmup: warmup iterations, excluded from the measurement.
        rep: measurement iterations.
        flush_l2: clear L2 between reps so repeated calls do not benefit from a
            warm cache. The DSV3 fused-A weights are ~30 MB in bf16, which fits
            in L2 on the large parts, so leaving the cache warm would overstate
            throughput. Must be False to measure the cache-resident case.

    Returns:
        Median milliseconds per call.

    Raises:
        RuntimeError: if called without a CUDA device. CPU timings are not
            reported, because the number would be meaningless for a kernel whose
            entire purpose is latency on an accelerator.
    """
    if not torch.cuda.is_available():
        raise RuntimeError(
            "bench() requires a CUDA device. This project does not report CPU "
            "timings -- run the benchmark on the cloud GPU box instead."
        )

    if warmup < 1 or rep < 1:
        raise ValueError("warmup and rep must be positive iteration counts")
    # One timing path makes warmup/rep units and cache policy consistent.
    # Compilation/autotuning occurs in warmup; these are iterations, not ms.
    for _ in range(warmup):
        fn()
    cuda_sync()

    samples: list[float] = []
    for _ in range(rep):
        if flush_l2:
            _flush_l2()
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        fn()
        end.record()
        cuda_sync()
        samples.append(start.elapsed_time(end))

    return statistics.median(samples)


_L2_FLUSH_BUF: torch.Tensor | None = None


def _flush_l2() -> None:
    """Write a large buffer to evict L2 (classic do_bench trick)."""
    global _L2_FLUSH_BUF
    if _L2_FLUSH_BUF is None:
        # Fixed eviction attempt; not a guaranteed cold cache on all devices.
        _L2_FLUSH_BUF = torch.empty(256 * 1024 * 1024, dtype=torch.int8, device="cuda")
    _L2_FLUSH_BUF.zero_()


def speedup(baseline_ms: float, candidate_ms: float) -> float:
    """``baseline / candidate``; > 1.0 means the candidate is faster."""
    if candidate_ms <= 0:
        return float("inf")
    return baseline_ms / candidate_ms


@dataclass
class BenchResult:
    """One measured configuration."""

    label: str
    m: int
    dtype: torch.dtype
    ms: float
    impl: str
    meta: dict = field(default_factory=dict)


def format_table(
    results: Sequence[BenchResult],
    baseline_impl: str = "torch_noncontig",
) -> str:
    """Render results as a plain-text table with speedups vs. the baseline.

    Rows are keyed on ``(M, K, N, dtype)`` taken from ``BenchResult.meta``, so a
    sweep over several shapes stays unambiguous.

    The baseline is the torch path *with the non-contiguous mat_b*, because that
    is the honest comparison: the real caller never materialises a contiguous
    weight. The cost of that choice is made visible by the separate
    ``torch_contig`` row rather than being hidden inside the baseline.
    """
    if not results:
        return "(no results)"

    def key(r: BenchResult):
        return (r.m, r.meta.get("k"), r.meta.get("n"), r.dtype)

    baseline = {key(r): r.ms for r in results if r.impl == baseline_impl}

    header = (
        f"{'M':>4}  {'K':>6}  {'N':>6}  {'dtype':>10}  {'impl':<18}  "
        f"{'median ms':>10}  {'speedup':>8}"
    )
    lines = [header, "-" * len(header)]

    def sort_key(r: BenchResult):
        return (r.m, r.meta.get("k") or 0, r.meta.get("n") or 0, str(r.dtype), r.impl)

    for r in sorted(results, key=sort_key):
        base = baseline.get(key(r))
        sp = f"{speedup(base, r.ms):.2f}x" if base else "--"
        dtype_name = str(r.dtype).replace("torch.", "")
        k = r.meta.get("k", "?")
        n = r.meta.get("n", "?")
        lines.append(
            f"{r.m:>4}  {k!s:>6}  {n!s:>6}  {dtype_name:>10}  {r.impl:<18}  "
            f"{r.ms:>10.4f}  {sp:>8}"
        )

    return "\n".join(lines)
