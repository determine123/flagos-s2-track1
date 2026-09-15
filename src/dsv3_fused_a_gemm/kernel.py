"""Triton kernel for ``dsv3_fused_a_gemm`` -- public competition entry point.

The kernel body itself is generated on the KernelGen Web platform and lives
verbatim in :mod:`._kernelgen` (see that module for provenance and for the one
deletion made to the generated output). This module is the thin public surface
around it: the competition signature, the contest gate, and the availability
flag.

=========================== HOW TO REGENERATE ===========================
1. Generate on KernelGen Web with the request recorded in
   ``docs/SPEC.md`` section "KernelGen Web request".
2. Replace ``src/dsv3_fused_a_gemm/_kernelgen.py`` with the generated
   ``triton_code``. Keep the public entry point named ``dsv3_fused_a_gemm``
   with the competition signature ``(mat_a, mat_b) -> Tensor``.
3. Drop any generated ``reference`` function -- if it calls the kernel it is
   circular and would make every correctness check pass vacuously. The oracle
   is :func:`.reference.reference`.
4. Nothing else needs editing; ``KERNEL_AVAILABLE`` is derived automatically.
5. Run ``pytest tests/ -v``.
=======================================================================

Contract the kernel must satisfy (identical to :func:`.reference.reference`):

    dsv3_fused_a_gemm(mat_a, mat_b) -> Tensor
        out = mat_a @ mat_b

    mat_a  : [M, K] row-major, bf16, K % 256 == 0
    mat_b  : [K, N] strides (1, K)  <-- NON-CONTIGUOUS column-major weight
             (N % 16 == 0). This is the REQUIRED main path.
    out    : [M, N] freshly allocated, row-major, bf16
    M      : 1..16

K = 7168 and N = 2112 are the DeepSeek-V3 shape and the headline benchmark
target. They are a *specialisation* opportunity, not a correctness restriction:
every (M, K, N) satisfying the constraints above must be handled correctly.

The ``mat_b`` layout is the single most important detail in this package. The
column-major view has strides ``(1, K)`` -- the K axis is the contiguous one --
so a kernel written against a row-major ``[K, N]`` operand with strides
``(1, N)`` reads the wrong elements. The generated kernel addresses ``mat_b``
through its real ``stride_bk`` / ``stride_bn`` arguments, which is correct for
both layouts. ``tests/test_kernel.py`` asserts equality against the reference on
a genuinely non-contiguous ``mat_b``, so a contiguous-only assumption cannot
pass.

Optimise for the column-major view, which is what the competition always passes.
A contiguous ``[K, N]`` operand is an optional project extension (``docs/SPEC.md``
§2.2), reachable only through internal stride parameters or a separate dispatch
branch -- never by adding a parameter to the public signature -- and it must not
cost the column-major path any performance.
"""

from __future__ import annotations

import torch

from .reference import validate_contest_contract

__all__ = ["KERNEL_AVAILABLE", "dsv3_fused_a_gemm"]

# The generated kernel needs Triton, which ``pyproject.toml`` declares as an
# optional extra. Import it behind a guard so the package -- and the CPU test
# suite that exercises the reference -- still imports on a machine without it.
try:
    from ._kernelgen import dsv3_fused_a_gemm as _kernelgen_impl
except ImportError:  # pragma: no cover - depends on the environment
    _kernelgen_impl = None


#: True when the generated kernel is present and its dependencies are importable.
KERNEL_AVAILABLE = _kernelgen_impl is not None


def dsv3_fused_a_gemm(mat_a: torch.Tensor, mat_b: torch.Tensor) -> torch.Tensor:
    """Triton implementation of the fused-A GEMM. Returns a fresh ``[M, N]`` tensor.

    Args:
        mat_a: ``[M, K]`` row-major activations, bfloat16.
        mat_b: ``[K, N]`` column-major transposed weight, strides ``(1, K)``.

    Returns:
        A newly allocated ``[M, N]`` row-major tensor of ``mat_a.dtype``.

    Raises:
        NotImplementedError: if the kernel body or Triton is unavailable.
        ValueError: if the operands violate the competition contract.
    """
    if not KERNEL_AVAILABLE:
        raise NotImplementedError(
            "dsv3_fused_a_gemm Triton kernel is not available. Either the "
            "generated body is missing from src/dsv3_fused_a_gemm/_kernelgen.py, "
            "or Triton is not installed (pip install 'dsv3-fused-a-gemm[kernel]'). "
            "See docs/SPEC.md for the KernelGen Web request parameters. Use the "
            "reference implementation (dsv3_fused_a_gemm.reference.reference) for "
            "CPU correctness checks."
        )

    validate_contest_contract(mat_a, mat_b)
    return _kernelgen_impl(mat_a, mat_b)
