"""vLLM out-variant adapter for ``dsv3_fused_a_gemm``.

**This module is not the competition interface.** It exists so that a caller
written against vLLM's upstream CUDA op --

    torch.ops._C.dsv3_fused_a_gemm(output, mat_a, mat_b, enable_pdl=False) -> None
    vllm._custom_ops.dsv3_fused_a_gemm(...)

-- can keep working against this repository's operator, which is defined as::

    dsv3_fused_a_gemm(mat_a, mat_b) -> Tensor

The out-variant is a thin adapter over that contract: it calls the operator and
copies the result into the caller's buffer. It is not the reference, it is not
the default test target, and ``pytest`` does not collect it (the test suite lives
under ``tests/``, and nothing here is on the competition path).

Note the deliberate asymmetry: ``enable_pdl`` is accepted and ignored. It is a
launch-time CUDA feature (programmatic dependent launch) with no bearing on the
result, and the competition contract has no such flag.
"""

from __future__ import annotations

import os
import sys

import torch

# Keep the adapter runnable straight from a checkout, matching benchmarks/.
_SRC = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "src"))
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

from dsv3_fused_a_gemm import dsv3_fused_a_gemm  # noqa: E402

__all__ = ["dsv3_fused_a_gemm_out"]


def dsv3_fused_a_gemm_out(
    output: torch.Tensor,
    mat_a: torch.Tensor,
    mat_b: torch.Tensor,
    enable_pdl: bool = False,
) -> None:
    """Write ``mat_a @ mat_b`` into ``output``, mirroring the upstream out-variant.

    Args:
        output: ``[M, N]`` row-major destination, written in place.
        mat_a: ``[M, K]`` row-major activations.
        mat_b: ``[K, N]`` weight; typically the non-contiguous ``weight.T`` view.
        enable_pdl: accepted for upstream signature compatibility and ignored
            (see the module docstring).

    Returns:
        ``None``, matching upstream -- the result lands in ``output``.

    Raises:
        TypeError: if any argument is not a tensor.
        ValueError: if ``output`` is not row-major or its shape disagrees with
            ``(mat_a.shape[0], mat_b.shape[1])``.
    """
    for name, t in (("output", output), ("mat_a", mat_a), ("mat_b", mat_b)):
        if not isinstance(t, torch.Tensor):
            raise TypeError(f"{name} must be a torch.Tensor, got {type(t).__name__}")

    expected = (mat_a.shape[0], mat_b.shape[1])
    if tuple(output.shape) != expected:
        raise ValueError(
            f"output shape {tuple(output.shape)} does not match (M, N) = {expected}"
        )
    if not output.is_contiguous():
        raise ValueError(
            "output must be row-major (contiguous); the upstream kernel writes "
            "it with row-major strides"
        )
    if output.dtype != mat_a.dtype:
        raise ValueError(
            f"dtype mismatch: output={output.dtype}, mat_a={mat_a.dtype}"
        )

    result = dsv3_fused_a_gemm(mat_a, mat_b)
    output.copy_(result)
    return None
