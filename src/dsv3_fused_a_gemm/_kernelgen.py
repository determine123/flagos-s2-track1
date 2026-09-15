"""KernelGen-generated kernel body for ``dsv3_fused_a_gemm``.

PROVENANCE
    Platform : KernelGen Web (https://kernelgen.flagos.io)
    Request  : ``docs/SPEC.md`` §6 -- kernel_name=dsv3_fused_a_gemm,
               func_type=other, device=nvidia, arg_names="mat_a, mat_b"
    Shape    : M=1..16, K=7168 (multiple of 256), N=2112 (multiple of 16), bf16

This file holds the generated code unmodified apart from the one deletion noted
below, so a future regeneration can be diffed against it directly. Everything
around it -- the public signature, the contest gate, the ``KERNEL_AVAILABLE``
flag -- lives in :mod:`.kernel`, which imports this module behind a guard so the
package still imports when Triton is absent (see the ``kernel`` extra in
``pyproject.toml``).

DELETION FROM THE GENERATED OUTPUT
    The generated bundle also defined::

        def reference(mat_a, mat_b):
            return dsv3_fused_a_gemm(mat_a, mat_b)

    That is circular -- it calls the Triton kernel, so comparing the kernel
    against it would always pass. It has been dropped. The oracle is
    :func:`dsv3_fused_a_gemm.reference.reference`, which upcasts both operands to
    float32 and casts back to the input dtype.

Layout note
-----------
``mat_b`` is addressed through its real ``stride_bk`` / ``stride_bn`` arguments.
For the competition's column-major view those are ``(1, K)``, so the load walks
the contiguous K axis; for a contiguous ``[K, N]`` tensor they are ``(N, 1)``.
Both work from the same code path, so the contiguous case costs the main path
nothing -- no branch, no extra argument on the public signature.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        triton.Config({"BLOCK_N": 16, "BLOCK_K": 32}, num_warps=2, num_stages=2),
        triton.Config({"BLOCK_N": 16, "BLOCK_K": 64}, num_warps=2, num_stages=3),
        triton.Config({"BLOCK_N": 16, "BLOCK_K": 128}, num_warps=4, num_stages=3),
        triton.Config({"BLOCK_N": 32, "BLOCK_K": 32}, num_warps=2, num_stages=3),
        triton.Config({"BLOCK_N": 32, "BLOCK_K": 64}, num_warps=2, num_stages=3),
        triton.Config({"BLOCK_N": 32, "BLOCK_K": 128}, num_warps=4, num_stages=3),
        triton.Config({"BLOCK_N": 64, "BLOCK_K": 32}, num_warps=4, num_stages=3),
        triton.Config({"BLOCK_N": 64, "BLOCK_K": 64}, num_warps=4, num_stages=3),
        triton.Config({"BLOCK_N": 64, "BLOCK_K": 128}, num_warps=4, num_stages=3),
        triton.Config({"BLOCK_N": 128, "BLOCK_K": 32}, num_warps=4, num_stages=3),
        triton.Config({"BLOCK_N": 128, "BLOCK_K": 64}, num_warps=8, num_stages=3),
        triton.Config({"BLOCK_N": 128, "BLOCK_K": 128}, num_warps=8, num_stages=2),
    ],
    key=["M", "N", "K"],
)
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

    grid = lambda META: (triton.cdiv(N, META["BLOCK_N"]),)

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
        BLOCK_M=16,
    )

    return out
