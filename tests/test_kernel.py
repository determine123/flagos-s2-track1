"""Tests for the Triton kernel.

Every test here is skipped until the KernelGen Web kernel is integrated (see
``kernel.KERNEL_AVAILABLE``) and a CUDA device is present. The skip is explicit
rather than silent so a local run reports *why* nothing executed, and so these
tests light up automatically the moment the kernel lands -- no edit required.

These tests are the ones that must be run on the cloud GPU box. The primary gate
is written against the competition contract literally::

    actual   = dsv3_fused_a_gemm(mat_a, mat_b)
    expected = reference(mat_a, mat_b)

Two arguments in, one new tensor out, with ``mat_b`` as the column-major
transposed view (strides ``(1, K)``). **That is the main path and the only one
the competition requires** -- it is what every performance claim is measured
against, and it is what the unmarked tests here cover:

    pytest -m "not extension"     # main path only

Tests marked ``@pytest.mark.extension`` additionally cover a contiguous
``[K, N]`` operand. That case is an optional project extension, not a
requirement (``docs/SPEC.md`` §2.2), and must never be bought with main-path
performance.
"""

from __future__ import annotations

import pytest
import torch

from dsv3_fused_a_gemm import KERNEL_AVAILABLE, dsv3_fused_a_gemm
from dsv3_fused_a_gemm.reference import (
    DSV3_FUSED_A_N,
    DSV3_HIDDEN,
    K_ALIGNMENT,
    M_MAX,
    M_MIN,
    M_RANGE,
    N_ALIGNMENT,
    reference,
)

from accuracy_utils import (
    assert_close,
    assert_is_non_contiguous,
    get_device,
    make_operands,
)

requires_kernel = pytest.mark.skipif(
    not KERNEL_AVAILABLE,
    reason=(
        "Triton kernel not integrated yet -- generate on KernelGen Web and paste "
        "into src/dsv3_fused_a_gemm/kernel.py (see docs/SPEC.md)"
    ),
)

requires_gpu = pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="no CUDA device available; run this suite on the cloud GPU box",
)

#: Aligned (K, N) pairs that are inside the contract but are not the DSV3 shape.
#: K=7168/N=2112 is the headline target; it is not the only legal shape.
GENERIC_SHAPES = [
    (K_ALIGNMENT, N_ALIGNMENT),
    (K_ALIGNMENT * 2, N_ALIGNMENT * 2),
    (K_ALIGNMENT * 4, N_ALIGNMENT * 8),
    (DSV3_HIDDEN, DSV3_FUSED_A_N),  # 7168 x 2112 -- the DSV3 target
]


def _case(m, k=DSV3_HIDDEN, n=DSV3_FUSED_A_N, seed=0, non_contiguous_b=True):
    """Build a case on the GPU, falling back to CPU when none is present."""
    return make_operands(
        m,
        k=k,
        n=n,
        dtype=torch.bfloat16,
        device=get_device(),
        seed=seed,
        non_contiguous_b=non_contiguous_b,
    )


@requires_kernel
@requires_gpu
@pytest.mark.parametrize("m", M_RANGE)
def test_kernel_matches_reference_non_contiguous(m):
    """Primary correctness gate: strided mat_b, kernel vs reference, all of M."""
    mat_a, mat_b = _case(m, seed=7)
    assert_is_non_contiguous(mat_b)

    actual = dsv3_fused_a_gemm(mat_a, mat_b)
    expected = reference(mat_a, mat_b)

    assert_close(actual, expected, dtype=torch.bfloat16, msg=f"M={m} (strided)")


@requires_kernel
@requires_gpu
@pytest.mark.extension
@pytest.mark.parametrize("m", [1, 16])
def test_kernel_matches_reference_contiguous_control(m):
    """Extension, optional: a contiguous ``mat_b`` gives the same answer.

    A contiguous ``[K, N]`` operand is **not** a competition requirement
    (``docs/SPEC.md`` §2.2) -- the competition always passes the column-major
    view. This test is here to keep the kernel from resting on a layout
    assumption it never has to make, and it is excluded from the main-path suite
    run with ``pytest -m "not extension"``.

    Supporting it is optional; if the dispatch branch that handles it slows the
    column-major path down, drop the extension rather than the performance.
    """
    mat_a, mat_b = _case(m, seed=13, non_contiguous_b=False)
    assert mat_b.is_contiguous()

    actual = dsv3_fused_a_gemm(mat_a, mat_b)
    expected = reference(mat_a, mat_b)

    assert_close(actual, expected, dtype=torch.bfloat16, msg=f"M={m} (contig)")


@requires_kernel
@requires_gpu
@pytest.mark.parametrize("k,n", GENERIC_SHAPES)
def test_kernel_handles_generic_aligned_shapes(k, n):
    """The DSV3 shape is a specialisation target, not a correctness limit."""
    for m in (M_MIN, M_MAX):
        mat_a, mat_b = _case(m, k=k, n=n, seed=19)

        actual = dsv3_fused_a_gemm(mat_a, mat_b)
        expected = reference(mat_a, mat_b)

        assert actual.shape == (m, n)
        assert_close(
            actual, expected, dtype=torch.bfloat16, msg=f"M={m} K={k} N={n}"
        )


@requires_kernel
@requires_gpu
def test_kernel_returns_a_fresh_tensor():
    """The interface returns a new tensor -- not None, and not an input alias."""
    m = 8
    mat_a, mat_b = _case(m, seed=2)

    actual = dsv3_fused_a_gemm(mat_a, mat_b)

    assert isinstance(actual, torch.Tensor)
    assert actual.shape == (m, DSV3_FUSED_A_N)
    assert actual.dtype == torch.bfloat16
    assert actual.is_contiguous()
    assert actual.data_ptr() not in (mat_a.data_ptr(), mat_b.data_ptr())


@requires_kernel
@requires_gpu
def test_kernel_does_not_mutate_its_inputs():
    mat_a, mat_b = _case(4, seed=5)
    mat_a_before = mat_a.clone()
    mat_b_before = mat_b.clone()

    dsv3_fused_a_gemm(mat_a, mat_b)

    assert torch.equal(mat_a, mat_a_before), "kernel mutated mat_a"
    assert torch.equal(mat_b, mat_b_before), "kernel mutated mat_b"


@requires_kernel
@requires_gpu
def test_kernel_writes_every_output_element():
    """No output element may be left uninitialised."""
    mat_a, mat_b = _case(4, seed=5)
    actual = dsv3_fused_a_gemm(mat_a, mat_b)
    assert not torch.isnan(actual).any(), "kernel left output elements unset"


@requires_kernel
@requires_gpu
def test_kernel_is_deterministic_across_calls():
    """Two calls on identical inputs must produce identical output."""
    mat_a, mat_b = _case(2, seed=4)
    first = dsv3_fused_a_gemm(mat_a, mat_b)
    second = dsv3_fused_a_gemm(mat_a, mat_b)
    assert torch.equal(first, second)


@requires_kernel
@requires_gpu
@pytest.mark.parametrize(
    "m,k,n",
    [
        (M_MAX + 1, 7168, DSV3_FUSED_A_N),   # M out of range
        (2, K_ALIGNMENT - 1, DSV3_FUSED_A_N),  # K not a multiple of 256
        (2, 7168, N_ALIGNMENT - 1),          # N not a multiple of 16
    ],
)
def test_kernel_rejects_out_of_contract_shapes(m, k, n):
    """The competition surface validates its published constraints."""
    mat_a = torch.randn(m, k, dtype=torch.bfloat16, device=get_device())
    weight = torch.randn(n, k, dtype=torch.bfloat16, device=get_device())
    with pytest.raises(ValueError):
        dsv3_fused_a_gemm(mat_a, weight.T)


def test_kernel_stub_raises_before_integration():
    """While the slot is empty the module must fail loudly, not approximate."""
    if KERNEL_AVAILABLE:
        pytest.skip("kernel has been integrated; the stub guard no longer applies")

    mat_a, mat_b = make_operands(2, dtype=torch.bfloat16, seed=1)
    with pytest.raises(NotImplementedError, match="KernelGen Web"):
        dsv3_fused_a_gemm(mat_a, mat_b)


def test_kernel_has_no_out_variant_parameter():
    """The competition entry point takes exactly ``(mat_a, mat_b)``.

    Guards against an out-variant or an ``enable_pdl`` flag leaking back onto the
    competition surface; both belong in ``integrations/vllm/``.
    """
    import inspect

    params = list(inspect.signature(dsv3_fused_a_gemm).parameters)
    assert params == ["mat_a", "mat_b"], (
        f"dsv3_fused_a_gemm must take (mat_a, mat_b), found {params}"
    )
