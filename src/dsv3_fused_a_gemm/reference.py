"""Reference implementation of ``dsv3_fused_a_gemm``.

The operator is the fused "A" projection of the DeepSeek-V3 MLA attention block.
In DeepSeek-V3 the query and key/value low-rank projections both consume the same
hidden state, so they can be merged into a single GEMM:

    q_a_proj:             [hidden] -> [q_lora_rank]
    kv_a_proj_with_mqa:   [hidden] -> [kv_lora_rank + qk_rope_head_dim]

Concatenating their weights along the output dimension yields one GEMM whose
N dimension is ``q_lora_rank + kv_lora_rank + qk_rope_head_dim``. This saves a
kernel launch on the decode path, which matters because decode is latency-bound.

The public contract -- defined by the FlagOS Season 2 competition, not by the
vLLM upstream CUDA op -- is::

    dsv3_fused_a_gemm(mat_a, mat_b) -> Tensor

    out = mat_a @ mat_b

    mat_a : [M, K]  row-major
    mat_b : [K, N]  "column-major transposed weight", i.e. weight.T
    out   : [M, N]  freshly allocated, row-major, input dtype
    M     : 1..16 tokens (low-latency / decode regime)
    K     : multiple of 256
    N     : multiple of 16
    dtype : bfloat16

Two arguments in, one new tensor out. There is no ``output`` parameter and no
``enable_pdl`` flag on the competition surface. vLLM's upstream out-of-place
variant (``dsv3_fused_a_gemm(output, mat_a, mat_b, enable_pdl=False) -> None``)
is preserved separately in ``integrations/vllm/adapter.py``; it is an adapter
over this contract, never a replacement for it.

Why ``mat_b`` is the interesting part
-------------------------------------
The weight is materialised row-major as ``[N, K]``, so ``weight.T`` is ``[K, N]``
with strides ``(1, K)`` -- a *transposed view*, not a contiguous tensor. This is
not an accident of the API; it is what the caller naturally has in hand after
fusing the two projection weights, and avoiding a ``.contiguous()`` copy is part
of the point of the fusion.

A kernel that assumes contiguous ``mat_b`` will silently read the wrong elements
or fall back to a slow path, so non-contiguity has to be handled explicitly and
tested for. ``torch.matmul`` handles arbitrary strides natively, which makes this
reference a valid oracle for the strided case -- see
``tests/test_reference.py::test_mat_b_transposed_view_is_non_contiguous``, which
asserts the view really is non-contiguous so the test cannot silently degrade
into testing the easy path.
"""

from __future__ import annotations

import torch

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
    "reference",
    "validate_contest_contract",
    "make_mat_b",
]

# --- DeepSeek-V3 MLA "A" projection geometry -------------------------------
# Source: DeepSeek-V3 config (hidden_size=7168, q_lora_rank=1536,
# kv_lora_rank=512, qk_rope_head_dim=64).
DSV3_HIDDEN = 7168
DSV3_Q_LORA_RANK = 1536
DSV3_KV_LORA_RANK = 512
DSV3_QK_ROPE_HEAD_DIM = 64

#: Fused output width: q_a_proj || kv_a_proj_with_mqa.
DSV3_FUSED_A_N = DSV3_Q_LORA_RANK + DSV3_KV_LORA_RANK + DSV3_QK_ROPE_HEAD_DIM  # 2112

#: The DSV3 shape is the headline benchmark target. It is a *specialisation*
#: target, not a restriction: the operator must be correct for every shape that
#: satisfies the contest constraint below, not just for this one.
DSV3_SHAPES = (1, 2, 4, 8, 16)

# --- Competition correctness range -----------------------------------------
# These are the published FlagOS Season 2 constraints. They bound what the
# operator is required to handle; the reference stays generic so that it remains
# a usable oracle, and `validate_contest_contract` is the gate the competition
# surface applies.
M_MIN = 1
M_MAX = 16

#: Number of tokens M covers 1..16 inclusive.
M_RANGE = tuple(range(M_MIN, M_MAX + 1))

#: K must be a multiple of this (K = hidden = 7168 = 28 * 256).
K_ALIGNMENT = 256

#: N must be a multiple of this (N = 2112 = 132 * 16).
N_ALIGNMENT = 16

#: The competition is bfloat16-only.
CONTEST_DTYPE = torch.bfloat16


def _check_operands(mat_a: torch.Tensor, mat_b: torch.Tensor) -> None:
    """Check that the two operands can be multiplied.

    Deliberately *permissive* about M/K/N: this is the oracle, and it should
    compute an answer for any well-formed pair so it can be used to debug shapes
    the contest gate rejects. Range enforcement lives in
    :func:`validate_contest_contract`.

    Raises:
        TypeError: if either operand is not a tensor.
        ValueError: if either operand is not 2-D, if the K dimensions disagree,
            or if the dtypes differ.
    """
    for name, t in (("mat_a", mat_a), ("mat_b", mat_b)):
        if not isinstance(t, torch.Tensor):
            raise TypeError(f"{name} must be a torch.Tensor, got {type(t).__name__}")
        if t.dim() != 2:
            raise ValueError(
                f"{name} must be 2-D, got shape {tuple(t.shape)} "
                f"(the fused-A GEMM is a batched-free 2-D GEMM)"
            )

    if mat_a.shape[1] != mat_b.shape[0]:
        raise ValueError(
            f"K dimension mismatch: mat_a is {tuple(mat_a.shape)} but mat_b is "
            f"{tuple(mat_b.shape)}; expected mat_a.shape[1] == mat_b.shape[0]"
        )

    if mat_a.dtype != mat_b.dtype:
        raise ValueError(
            f"dtype mismatch: mat_a={mat_a.dtype}, mat_b={mat_b.dtype}; both "
            f"operands must share a dtype"
        )


def validate_contest_contract(mat_a: torch.Tensor, mat_b: torch.Tensor) -> None:
    """Enforce the published competition constraints on a pair of operands.

    The competition surface accepts:

    * ``M = mat_a.shape[0]`` in ``[1, 16]``
    * ``K = mat_a.shape[1] == mat_b.shape[0]``, a multiple of 256
    * ``N = mat_b.shape[1]``, a multiple of 16
    * both operands ``bfloat16``

    ``mat_b`` may be the non-contiguous ``weight.T`` view (strides ``(1, K)``);
    that is the common case, not a violation.

    Raises:
        TypeError: if either operand is not a tensor.
        ValueError: if any constraint above is violated.
    """
    _check_operands(mat_a, mat_b)

    m, k = mat_a.shape
    n = mat_b.shape[1]

    if not (M_MIN <= m <= M_MAX):
        raise ValueError(
            f"M = {m} is outside the supported range [{M_MIN}, {M_MAX}] tokens "
            f"(the fused-A GEMM targets the low-latency decode regime)"
        )

    if k % K_ALIGNMENT != 0:
        raise ValueError(
            f"K = {k} is not a multiple of {K_ALIGNMENT} "
            f"(the competition requires K % {K_ALIGNMENT} == 0)"
        )

    if n % N_ALIGNMENT != 0:
        raise ValueError(
            f"N = {n} is not a multiple of {N_ALIGNMENT} "
            f"(the competition requires N % {N_ALIGNMENT} == 0)"
        )

    if mat_a.dtype != CONTEST_DTYPE or mat_b.dtype != CONTEST_DTYPE:
        raise ValueError(
            f"dtype mismatch: mat_a={mat_a.dtype}, mat_b={mat_b.dtype}; the "
            f"competition is {CONTEST_DTYPE} only"
        )


def reference(mat_a: torch.Tensor, mat_b: torch.Tensor) -> torch.Tensor:
    """Compute ``mat_a @ mat_b`` in float32 and return it in the input dtype.

    This is the oracle every kernel is compared against. The multiply is
    performed with both operands upcast to ``float32`` -- a bfloat16 oracle would
    be too coarse to separate a correct kernel from a subtly wrong one -- and the
    result is then cast back to the input dtype, matching what the kernel is
    required to return.

    ``mat_b`` is deliberately **not** made contiguous: ``torch.matmul`` honours
    arbitrary strides, and the strided view is precisely the layout the real
    kernel must handle. Materialising it here would make the oracle pass for the
    wrong reason.

    Args:
        mat_a: ``[M, K]`` row-major activations.
        mat_b: ``[K, N]`` weight; typically the non-contiguous ``weight.T`` view
            with strides ``(1, K)``.

    Returns:
        A freshly allocated ``[M, N]`` tensor of ``mat_a.dtype``.

    Raises:
        TypeError: if either operand is not a tensor.
        ValueError: if the operand shapes or dtypes are incompatible.
    """
    _check_operands(mat_a, mat_b)

    # Explicit float32 upcast for the multiply, then back to the input dtype.
    return (mat_a.float() @ mat_b.float()).to(mat_a.dtype)


def make_mat_b(
    dtype: torch.dtype = torch.bfloat16,
    device: torch.device | str = "cpu",
    generator: torch.Generator | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Build the canonical DSV3 operand pair.

    Returns:
        ``(weight, mat_b)`` where ``weight`` is the contiguous ``[N, K]`` fused
        weight and ``mat_b`` is ``weight.T`` -- the non-contiguous ``[K, N]``
        view the kernel actually receives, with strides ``(1, K)``.
    """
    weight = torch.randn(
        DSV3_FUSED_A_N, DSV3_HIDDEN, dtype=dtype, device=device, generator=generator
    )
    return weight, weight.T
