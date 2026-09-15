"""Test helpers for the dsv3_fused_a_gemm project.

Deliberately standalone: this project does not import ``flag_gems``, so the test
suite runs on any machine with PyTorch -- including GPU-less ones. That matters
because the reference oracle and the whole harness can then be validated locally,
leaving only the actual kernel numerics for the cloud GPU run.
"""

from __future__ import annotations

import torch

from dsv3_fused_a_gemm.reference import (
    DSV3_FUSED_A_N,
    DSV3_HIDDEN,
)

__all__ = [
    "get_device",
    "has_gpu",
    "to_reference",
    "assert_close",
    "make_operands",
    "assert_is_non_contiguous",
    "TOLERANCES",
]


def get_device() -> str:
    """Return ``"cuda"`` when a GPU is present, else ``"cpu"``."""
    return "cuda" if torch.cuda.is_available() else "cpu"


def has_gpu() -> bool:
    """True when a CUDA device is usable."""
    return torch.cuda.is_available()


def to_reference(t: torch.Tensor) -> torch.Tensor:
    """Upcast a tensor to float32 for use as a reference operand.

    bfloat16 has ~8 bits of mantissa, so a bf16-precision oracle would be too
    coarse to distinguish a correct kernel from a subtly wrong one. Upcasting
    the *inputs* (not the result) keeps the oracle accurate while still testing
    the kernel's own bf16 path.
    """
    return t.to(torch.float32)


# Tolerances are keyed by the *output* dtype.
#
# Rationale for bfloat16: the accumulation runs over K=7168 terms of a standard
# normal product, so the pre-rounding result has magnitude ~sqrt(7168) ~ 85.
# bfloat16 carries ~2^-8 relative precision, so a tolerance looser than a single
# bf16 ULP is required to avoid flagging correct kernels; rtol=2e-2 covers
# accumulation-order differences without being so loose that a real indexing
# bug (which typically produces O(1) relative errors) slips through.
TOLERANCES = {
    torch.bfloat16: {"rtol": 2e-2, "atol": 1e-2},
    torch.float16: {"rtol": 2e-2, "atol": 1e-2},
    torch.float32: {"rtol": 1e-4, "atol": 1e-4},
}


def assert_close(
    actual: torch.Tensor,
    expected: torch.Tensor,
    dtype: torch.dtype | None = None,
    msg: str = "",
) -> None:
    """Assert two tensors match within the tolerance for ``dtype``.

    On mismatch, reports the max absolute and relative deviation so a failing
    kernel can be diagnosed without re-running with extra prints.
    """
    key = dtype or actual.dtype
    tol = TOLERANCES.get(key, {"rtol": 1e-3, "atol": 1e-3})

    if actual.shape != expected.shape:
        raise AssertionError(
            f"shape mismatch: actual {tuple(actual.shape)} vs "
            f"expected {tuple(expected.shape)}. {msg}"
        )

    a = actual.to(torch.float32)
    e = expected.to(torch.float32)
    diff = (a - e).abs()
    denom = e.abs().clamp_min(1e-6)
    max_abs = diff.max().item() if diff.numel() else 0.0
    max_rel = (diff / denom).max().item() if diff.numel() else 0.0

    ok = torch.allclose(a, e, rtol=tol["rtol"], atol=tol["atol"])
    if not ok:
        raise AssertionError(
            f"tensors not close (rtol={tol['rtol']}, atol={tol['atol']}): "
            f"max_abs_diff={max_abs:.6e}, max_rel_diff={max_rel:.6e}. {msg}"
        )


def make_operands(
    m: int,
    k: int = DSV3_HIDDEN,
    n: int = DSV3_FUSED_A_N,
    dtype: torch.dtype = torch.bfloat16,
    device: str | torch.device = "cpu",
    seed: int = 0,
    non_contiguous_b: bool = True,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Build one ``(mat_a, mat_b)`` operand pair for a fused-A GEMM case.

    The operator takes two arguments and returns a new tensor, so this helper
    returns exactly those two arguments -- there is no preallocated output.

    Args:
        m: token count (the competition range is 1..16).
        k: reduction dimension; must satisfy ``k % 256 == 0`` for the case to be
            inside the competition contract.
        n: output width; must satisfy ``n % 16 == 0`` for the case to be inside
            the competition contract.
        dtype: operand dtype.
        device: where to allocate.
        seed: RNG seed, for reproducible comparisons.
        non_contiguous_b: when True (the default), ``mat_b`` is the column-major
            transposed ``weight.T`` view with strides ``(1, k)`` and is therefore
            non-contiguous. **This is the competition's main path** -- the layout
            the kernel is optimised and benchmarked for. Set False only to build
            a contiguous operand, which is an optional project extension
            (``docs/SPEC.md`` §2.2) rather than a requirement.

    Returns:
        ``(mat_a, mat_b)`` with ``mat_a`` ``[M, K]`` and ``mat_b`` ``[K, N]``.
    """
    # A generator is device-bound, so match it to the target device or CUDA
    # allocation will reject it.
    gen_device = "cuda" if str(device).startswith("cuda") else "cpu"
    gen = torch.Generator(device=gen_device).manual_seed(seed)

    mat_a = torch.randn(m, k, dtype=dtype, device=device, generator=gen)

    # A contiguous [N, K] weight read as weight.T is a [K, N] view with strides
    # (1, K) -- the layout the real caller hands over.
    weight = torch.randn(n, k, dtype=dtype, device=device, generator=gen)
    mat_b = weight.T
    if not non_contiguous_b:
        # `.contiguous()` on a [K, N] transposed view materialises it; this is
        # the control case that a contiguous-only kernel would pass.
        mat_b = mat_b.contiguous()

    return mat_a, mat_b


def assert_is_non_contiguous(mat_b: torch.Tensor) -> None:
    """Fail loudly if ``mat_b`` is actually contiguous.

    Guards the strided tests: a test that *claims* to exercise the transposed
    view but silently receives a contiguous tensor is worse than no test, because
    it reports success for the wrong reason.
    """
    if mat_b.is_contiguous():
        raise AssertionError(
            "expected mat_b to be a non-contiguous transposed view, but it is "
            "contiguous -- the strided path is NOT being exercised. Check that "
            "you did not call .contiguous() on it before the kernel call."
        )
