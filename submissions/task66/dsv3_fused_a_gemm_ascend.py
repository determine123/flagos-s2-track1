"""dsv3_fused_a_gemm -- Task 66 performance variant for Ascend (华为).

This is the fourth Task 66 experiment (perf_test_01), built as the Ascend twin of
the Enflame variant: the code below is byte-identical to
``dsv3_fused_a_gemm_enflame.py`` and differs only in this note.

Prior result: Ascend reached correctness 8/8 but only 0.01x -- roughly two orders
of magnitude slower than the torch baseline. That is a throughput collapse, not a
compile failure, and it is the problem this file is testing against.

PINNED CONFIGURATION -- identical to this round's Enflame variant
        BLOCK_M = 16, BLOCK_N = 64, BLOCK_K = 64, num_warps = 4, num_stages = 2

    The previous round pinned BLOCK_N = 32, BLOCK_K = 32. Note that BLOCK_N and
    BLOCK_K are two independent knobs, so this is strictly a two-variable move
    even though it is one tile-size step.

    The lever tested: 64x64 halves the K loop trip count (224 -> 112) and
    doubles the arithmetic per program. The autotune decorator is NOT restored.

    The grid is computed from the pinned ``BLOCK_N`` so that the N dimension is
    still covered exactly:

        grid = (triton.cdiv(N, BLOCK_N),)

    K % 256 == 0 and BLOCK_K = 64 divides 256, so the K loop has no tail and
    needs no K mask. N = 2112 = 33 x 64 exactly, so the headline shape tiles N
    with no partial block.

    DELIBERATELY NOT CHANGED: the kernel body, the stride addressing, the masks,
    the store, the public interface, and the ``reference`` entry point. No
    64-bit offset cast is added -- with K=7168 and N=2112 the largest offset is
    ~15.1M, far inside 32-bit range, so there is no overflow to fix.

Contract
--------
    dsv3_fused_a_gemm(mat_a, mat_b) -> Tensor

    mat_a : [M, K]  row-major, bf16, M in 1..16, K % 256 == 0
    mat_b : [K, N]  column-major transposed weight, actual strides (1, K)
    out   : [M, N]  freshly allocated, row-major, bf16, N % 16 == 0

    ``mat_b`` is the ``.T`` view of a contiguous ``[N, K]`` weight, so its
    contiguous axis is K. Reading it as a row-major ``[K, N]`` tensor with
    strides ``(1, N)`` is wrong. Because ``K % 256 == 0`` and ``BLOCK_K`` divides
    256, the K loop has no tail and needs no K mask.

Note on ``reference``
---------------------
``reference`` is the platform entry point. It calls the Triton kernel, so it is
NOT a correctness oracle -- comparing the kernel against it passes
unconditionally, for any kernel. The oracle is
``(mat_a.float() @ mat_b.float()).to(mat_a.dtype)``.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl

#: The pinned configuration -- the single experimental variable.
BLOCK_M = 16
BLOCK_N = 64
BLOCK_K = 64
NUM_WARPS = 4
NUM_STAGES = 2


@triton.jit
def _dsv3_fused_a_gemm_kernel(
    a_ptr,
    b_ptr,
    c_ptr,
    M,
    N,
    K,
    stride_am,
    stride_ak,
    stride_bk,
    stride_bn,
    stride_cm,
    stride_cn,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    pid_n = tl.program_id(0)

    offs_m = tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    mask_m = offs_m < M
    mask_n = offs_n < N

    a_ptrs = a_ptr + offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak
    b_ptrs = b_ptr + offs_k[:, None] * stride_bk + offs_n[None, :] * stride_bn

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for _ in range(0, K, BLOCK_K):
        a = tl.load(
            a_ptrs,
            mask=mask_m[:, None],
            other=0.0,
        )
        b = tl.load(
            b_ptrs,
            mask=mask_n[None, :],
            other=0.0,
        )
        acc = tl.dot(a, b, acc)
        a_ptrs += BLOCK_K * stride_ak
        b_ptrs += BLOCK_K * stride_bk

    c_ptrs = c_ptr + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn
    c_mask = mask_m[:, None] & mask_n[None, :]
    tl.store(c_ptrs, acc.to(c_ptr.dtype.element_ty), mask=c_mask)


@torch.no_grad()
def dsv3_fused_a_gemm(mat_a: torch.Tensor, mat_b: torch.Tensor) -> torch.Tensor:
    M, K = mat_a.shape
    _, N = mat_b.shape

    out = torch.empty((M, N), dtype=mat_a.dtype, device=mat_a.device)

    grid = (triton.cdiv(N, BLOCK_N),)

    _dsv3_fused_a_gemm_kernel[grid](
        mat_a,
        mat_b,
        out,
        M,
        N,
        K,
        mat_a.stride(0),
        mat_a.stride(1),
        mat_b.stride(0),
        mat_b.stride(1),
        out.stride(0),
        out.stride(1),
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        BLOCK_K=BLOCK_K,
        num_warps=NUM_WARPS,
        num_stages=NUM_STAGES,
    )

    return out


def reference(mat_a, mat_b):
    return dsv3_fused_a_gemm(mat_a, mat_b)
