"""dsv3_fused_a_gemm -- DeepSeek-V3 fused MLA "A" projection GEMM.

A Triton kernel reimplementation of the fused ``q_a_proj`` + ``kv_a_proj_with_mqa``
projection, evaluated as a standalone engineering project (FlagOS Season 2,
Track 1).

The public operator contract is two arguments in, one new tensor out::

    dsv3_fused_a_gemm(mat_a, mat_b) -> Tensor
        out = mat_a @ mat_b

    mat_a : [M, K]  row-major, bfloat16
    mat_b : [K, N]  transposed weight view, strides (1, K)
    out   : [M, N]  freshly allocated, row-major, bfloat16
    M     : 1..16        K % 256 == 0        N % 16 == 0

There is no ``output`` parameter, no ``enable_pdl`` flag, and no ``None`` return
on this surface. vLLM's out-variant is preserved as an adapter in
``integrations/vllm/adapter.py``; it is not the competition interface.

Public surface:

    dsv3_fused_a_gemm          Triton implementation (KERNEL_AVAILABLE flag)
    reference                  PyTorch oracle, runs anywhere (incl. CPU)
    make_mat_b                 build the canonical (weight, weight.T) pair
    validate_contest_contract  enforce the published M/K/N/dtype constraints

See ``docs/SPEC.md`` for the full specification, including the non-contiguous
``mat_b`` requirement, and ``docs/OPTIMIZATION_LOG.md`` for measured results.
"""

from __future__ import annotations

from .kernel import KERNEL_AVAILABLE, dsv3_fused_a_gemm
from .reference import (
    CONTEST_DTYPE,
    DSV3_FUSED_A_N,
    DSV3_HIDDEN,
    DSV3_KV_LORA_RANK,
    DSV3_QK_ROPE_HEAD_DIM,
    DSV3_Q_LORA_RANK,
    DSV3_SHAPES,
    K_ALIGNMENT,
    M_MAX,
    M_MIN,
    M_RANGE,
    N_ALIGNMENT,
    make_mat_b,
    reference,
    validate_contest_contract,
)

__version__ = "0.2.0"

__all__ = [
    "DSV3_HIDDEN",
    "DSV3_Q_LORA_RANK",
    "DSV3_KV_LORA_RANK",
    "DSV3_QK_ROPE_HEAD_DIM",
    "DSV3_FUSED_A_N",
    "DSV3_SHAPES",
    "M_MIN",
    "M_MAX",
    "M_RANGE",
    "K_ALIGNMENT",
    "N_ALIGNMENT",
    "CONTEST_DTYPE",
    "KERNEL_AVAILABLE",
    "dsv3_fused_a_gemm",
    "reference",
    "make_mat_b",
    "validate_contest_contract",
    "__version__",
]
