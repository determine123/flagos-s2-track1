"""Tests for the PyTorch reference oracle.

These run anywhere PyTorch runs, including a GPU-less machine, which is the
point: the oracle and the harness get validated locally so that the eventual
cloud GPU run only has to establish the *kernel's* numerics.

The contract under test is the competition one -- two arguments in, one new
tensor out::

    out = dsv3_fused_a_gemm(mat_a, mat_b)
    out = reference(mat_a, mat_b)

The column-major ``mat_b`` view -- strides ``(1, K)`` -- is the **main path**: it
is what the competition passes and what the kernel is optimised for. A reference
that silently made ``mat_b`` contiguous would pass every test here while
exercising nothing the real kernel must handle, so
``test_mat_b_transposed_view_is_non_contiguous`` pins the layout down explicitly.

Tests marked ``@pytest.mark.extension`` cover a contiguous ``[K, N]`` operand,
which is an optional project extension rather than a requirement
(``docs/SPEC.md`` §2.2). Run the main path alone with ``pytest -m "not extension"``.
"""

from __future__ import annotations

import pytest
import torch

from dsv3_fused_a_gemm.reference import (
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

from accuracy_utils import (
    assert_close,
    assert_is_non_contiguous,
    make_operands,
    to_reference,
)

DTYPES = [torch.float32, torch.bfloat16]

#: Aligned (K, N) pairs that satisfy the competition constraints but are *not*
#: the DeepSeek-V3 shape. The DSV3 shape is the headline target, not the only
#: shape that has to be correct.
ALIGNED_SHAPES = [
    (K_ALIGNMENT, N_ALIGNMENT),          # smallest legal case: 256 x 16
    (K_ALIGNMENT * 2, N_ALIGNMENT * 2),  # 512 x 32
    (K_ALIGNMENT * 4, N_ALIGNMENT * 8),  # 1024 x 128
    (DSV3_HIDDEN, DSV3_FUSED_A_N),       # 7168 x 2112 -- the DSV3 target
]


# --- geometry sanity -------------------------------------------------------


def test_fused_output_width_matches_dsv3_config():
    """2112 == q_lora_rank + kv_lora_rank + qk_rope_head_dim."""
    assert DSV3_FUSED_A_N == 2112
    assert DSV3_Q_LORA_RANK == 1536
    assert DSV3_KV_LORA_RANK == 512
    assert DSV3_QK_ROPE_HEAD_DIM == 64
    assert (
        DSV3_Q_LORA_RANK + DSV3_KV_LORA_RANK + DSV3_QK_ROPE_HEAD_DIM
        == DSV3_FUSED_A_N
    )


def test_hidden_size_matches_dsv3_config():
    assert DSV3_HIDDEN == 7168


def test_dsv3_shape_satisfies_the_contest_alignment_rules():
    """The headline shape is inside the contract, and so are the focus M values."""
    assert DSV3_HIDDEN % K_ALIGNMENT == 0
    assert DSV3_FUSED_A_N % N_ALIGNMENT == 0
    assert all(M_MIN <= m <= M_MAX for m in DSV3_SHAPES)
    assert set(DSV3_SHAPES) <= set(M_RANGE)


# --- the layout that makes this operator interesting ----------------------


def test_mat_b_transposed_view_is_non_contiguous():
    """``mat_b`` must be a genuinely non-contiguous transposed view.

    The weight is row-major ``[N, K]``, so ``weight.T`` is ``[K, N]`` with
    strides ``(1, K)`` -- note the *inner* stride is the one that is 1, because
    the K axis is the contiguous axis of the weight.

    If this ever fails, every strided test below has stopped testing the strided
    path and would pass for the wrong reason.
    """
    weight, mat_b = make_mat_b(dtype=torch.bfloat16, device="cpu")

    assert weight.shape == (DSV3_FUSED_A_N, DSV3_HIDDEN)
    assert mat_b.shape == (DSV3_HIDDEN, DSV3_FUSED_A_N)

    assert weight.is_contiguous()
    assert weight.stride() == (DSV3_HIDDEN, 1)
    assert mat_b.stride() == (1, DSV3_HIDDEN)

    assert not mat_b.is_contiguous()
    assert_is_non_contiguous(mat_b)


def test_hidden_is_the_contiguous_axis_of_mat_b():
    """A strided ``mat_b`` is read as ``[N, K]`` column-wise, not row-wise.

    Guards the stride convention from the other direction: if someone "fixes"
    the view to strides ``(1, N)`` -- i.e. treats ``mat_b`` as a row-major
    ``[K, N]`` laid out in N-major order -- the values read are different, and
    the reference would disagree with the fp32 oracle below.
    """
    mat_a, mat_b = make_operands(4, seed=17)
    assert mat_b.stride() == (1, DSV3_HIDDEN)

    wrong = torch.as_strided(
        mat_b, size=(DSV3_HIDDEN, DSV3_FUSED_A_N), stride=(1, DSV3_FUSED_A_N)
    )
    assert wrong.stride() == (1, DSV3_FUSED_A_N)
    # The two layouts must not coincide, otherwise this test proves nothing.
    assert not torch.equal(reference(mat_a, mat_b), reference(mat_a, wrong))


# --- correctness against a float32 matmul ----------------------------------


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("m", M_RANGE)
def test_reference_matches_fp32_matmul_non_contiguous(m, dtype):
    """The default (strided) path must match a float32 matmul oracle."""
    mat_a, mat_b = make_operands(m, dtype=dtype, device="cpu", seed=7)
    assert_is_non_contiguous(mat_b)

    out = reference(mat_a, mat_b)

    expected = torch.matmul(to_reference(mat_a), to_reference(mat_b)).to(dtype)
    assert_close(out, expected, dtype=dtype, msg=f"M={m} dtype={dtype}")


@pytest.mark.extension
@pytest.mark.parametrize("m", DSV3_SHAPES)
def test_reference_matches_fp32_matmul_contiguous_control(m):
    """Extension, optional: a contiguous ``mat_b`` agrees with the column-major view.

    A contiguous ``[K, N]`` operand is not a competition requirement (``docs/SPEC.md``
    §2.2); the competition always passes the column-major view. Together with the
    test above this pins down that layout is the only difference between the two
    paths.
    """
    mat_a, mat_b_strided = make_operands(m, dtype=torch.bfloat16, seed=11)
    _, mat_b_contig = make_operands(
        m, dtype=torch.bfloat16, seed=11, non_contiguous_b=False
    )
    assert_is_non_contiguous(mat_b_strided)
    assert mat_b_contig.is_contiguous()

    out_strided = reference(mat_a, mat_b_strided)
    out_contig = reference(mat_a, mat_b_contig)

    # Compared with the bf16 tolerance rather than `torch.equal`: the fp32
    # reduction inside torch.matmul may sum the K axis in a different order for
    # the two layouts, and a value sitting on a bf16 rounding boundary can then
    # round the other way. That is a ULP-level artefact of the oracle, not a
    # layout bug -- and a layout bug produces O(1) differences, which this still
    # catches.
    assert_close(
        out_strided,
        out_contig,
        dtype=torch.bfloat16,
        msg=(
            "strided and contiguous mat_b disagree -- the reference is not "
            "reading the transposed view correctly"
        ),
    )


@pytest.mark.parametrize("k,n", ALIGNED_SHAPES)
def test_reference_generic_aligned_shapes(k, n):
    """Correctness holds for every legal shape, not just the DSV3 one."""
    for m in (M_MIN, M_MAX):
        mat_a, mat_b = make_operands(m, k=k, n=n, seed=23)
        assert_is_non_contiguous(mat_b)

        out = reference(mat_a, mat_b)
        expected = torch.matmul(
            to_reference(mat_a), to_reference(mat_b)
        ).to(torch.bfloat16)

        assert out.shape == (m, n)
        assert_close(out, expected, dtype=torch.bfloat16, msg=f"M={m} K={k} N={n}")


def test_reference_multiplies_in_float32_and_casts_back():
    """The oracle must upcast both operands, multiply, then cast to input dtype.

    This is the contract, spelled out literally: a bf16-by-bf16 product would
    accumulate differently and cannot be used to judge a kernel that accumulates
    in fp32.
    """
    mat_a, mat_b = make_operands(4, dtype=torch.bfloat16, seed=31)

    actual = reference(mat_a, mat_b)
    expected = (mat_a.float() @ mat_b.float()).to(torch.bfloat16)

    assert actual.dtype == torch.bfloat16
    assert torch.equal(actual, expected), (
        "reference() is not computing (mat_a.float() @ mat_b.float()).to(dtype)"
    )


@pytest.mark.parametrize("dtype", DTYPES)
def test_reference_output_dtype_follows_the_input(dtype):
    mat_a, mat_b = make_operands(8, dtype=dtype, seed=3)
    out = reference(mat_a, mat_b)

    assert out.shape == (8, DSV3_FUSED_A_N)
    assert out.dtype == dtype
    assert out.is_contiguous()


def test_reference_returns_a_fresh_tensor_and_leaves_inputs_untouched():
    mat_a, mat_b = make_operands(4, dtype=torch.bfloat16, seed=5)
    mat_a_before = mat_a.clone()
    mat_b_before = mat_b.clone()

    out = reference(mat_a, mat_b)

    assert out.data_ptr() not in (mat_a.data_ptr(), mat_b.data_ptr())
    assert torch.equal(mat_a, mat_a_before), "reference() mutated mat_a"
    assert torch.equal(mat_b, mat_b_before), "reference() mutated mat_b"


def test_m_one_single_token_is_supported():
    """M=1 is the latency-critical decode case."""
    mat_a, mat_b = make_operands(1, dtype=torch.bfloat16, seed=9)
    out = reference(mat_a, mat_b)

    expected = torch.matmul(to_reference(mat_a), to_reference(mat_b)).to(
        torch.bfloat16
    )
    assert out.shape == (1, DSV3_FUSED_A_N)
    assert_close(out, expected, dtype=torch.bfloat16)


# --- the competition gate ---------------------------------------------------


@pytest.mark.parametrize("m", M_RANGE)
def test_contest_contract_accepts_the_whole_m_range(m):
    mat_a, mat_b = make_operands(m, seed=0)
    validate_contest_contract(mat_a, mat_b)  # must not raise


@pytest.mark.extension
def test_contest_contract_does_not_gate_on_layout():
    """The gate checks shapes and dtype, not strides.

    The *requirement* is the column-major view with strides ``(1, K)`` -- that is
    the main path the kernel is optimised and benchmarked for. But a caller that
    happens to hand over a contiguous tensor is still asking a legitimate
    question, so the gate does not reject it; the contiguous case is simply an
    optional project extension (``docs/SPEC.md`` §2.2), not a contest guarantee.
    """
    mat_a, mat_b = make_operands(1, non_contiguous_b=False)
    assert mat_b.is_contiguous()
    validate_contest_contract(mat_a, mat_b)  # must not raise


@pytest.mark.parametrize("m", [0, M_MAX + 1, 32])
def test_contest_contract_rejects_m_outside_1_to_16(m):
    mat_a = torch.randn(m, DSV3_HIDDEN, dtype=CONTEST_DTYPE)
    mat_b = torch.randn(DSV3_FUSED_A_N, DSV3_HIDDEN, dtype=CONTEST_DTYPE).T
    with pytest.raises(ValueError, match="outside the supported range"):
        validate_contest_contract(mat_a, mat_b)


@pytest.mark.parametrize("k", [128, K_ALIGNMENT + 1, DSV3_HIDDEN - 1])
def test_contest_contract_rejects_k_not_a_multiple_of_256(k):
    mat_a = torch.randn(2, k, dtype=CONTEST_DTYPE)
    mat_b = torch.randn(DSV3_FUSED_A_N, k, dtype=CONTEST_DTYPE).T
    with pytest.raises(ValueError, match="not a multiple of 256"):
        validate_contest_contract(mat_a, mat_b)


@pytest.mark.parametrize("n", [1, N_ALIGNMENT - 1, DSV3_FUSED_A_N + 8])
def test_contest_contract_rejects_n_not_a_multiple_of_16(n):
    mat_a = torch.randn(2, DSV3_HIDDEN, dtype=CONTEST_DTYPE)
    mat_b = torch.randn(n, DSV3_HIDDEN, dtype=CONTEST_DTYPE).T
    with pytest.raises(ValueError, match="not a multiple of 16"):
        validate_contest_contract(mat_a, mat_b)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float16])
def test_contest_contract_rejects_non_bf16(dtype):
    mat_a = torch.randn(2, DSV3_HIDDEN, dtype=dtype)
    mat_b = torch.randn(DSV3_FUSED_A_N, DSV3_HIDDEN, dtype=dtype).T
    with pytest.raises(ValueError, match="bfloat16 only"):
        validate_contest_contract(mat_a, mat_b)


def test_contest_contract_rejects_m_zero():
    """M=0 is out of contract even though an empty GEMM would 'work'."""
    mat_a = torch.randn(0, DSV3_HIDDEN, dtype=CONTEST_DTYPE)
    mat_b = torch.randn(DSV3_FUSED_A_N, DSV3_HIDDEN, dtype=CONTEST_DTYPE).T
    assert mat_b.shape == (DSV3_HIDDEN, DSV3_FUSED_A_N)
    with pytest.raises(ValueError, match="outside the supported range"):
        validate_contest_contract(mat_a, mat_b)


# --- operand validation -----------------------------------------------------


def test_k_dimension_mismatch_raises():
    mat_a = torch.randn(4, DSV3_HIDDEN, dtype=torch.bfloat16)
    bad_b = torch.randn(DSV3_HIDDEN - 1, 8, dtype=torch.bfloat16)
    with pytest.raises(ValueError, match="K dimension mismatch"):
        reference(mat_a, bad_b)


def test_dtype_mismatch_raises():
    mat_a = torch.randn(4, DSV3_HIDDEN, dtype=torch.bfloat16)
    bad_b = torch.randn(DSV3_HIDDEN, 8, dtype=torch.float32)
    with pytest.raises(ValueError, match="dtype mismatch"):
        reference(mat_a, bad_b)


@pytest.mark.parametrize("bad_dim", [1, 3])
def test_non_2d_operands_raise(bad_dim):
    if bad_dim == 1:
        mat_a = torch.randn(DSV3_HIDDEN, dtype=torch.bfloat16)
    else:
        mat_a = torch.randn(2, 2, DSV3_HIDDEN, dtype=torch.bfloat16)
    mat_b = torch.randn(DSV3_HIDDEN, 8, dtype=torch.bfloat16)
    with pytest.raises(ValueError, match="must be 2-D"):
        reference(mat_a, mat_b)


@pytest.mark.parametrize("bad_operand", ["mat_a", "mat_b"])
def test_non_tensor_operand_raises_type_error(bad_operand):
    mat_a = torch.zeros(1, DSV3_HIDDEN)
    mat_b = torch.zeros(DSV3_HIDDEN, 8)
    if bad_operand == "mat_a":
        mat_a = [[0.0] * DSV3_HIDDEN]
    else:
        mat_b = [[0.0] * 8 for _ in range(DSV3_HIDDEN)]

    with pytest.raises(TypeError, match="must be a torch.Tensor"):
        reference(mat_a, mat_b)


def test_the_reference_is_permissive_outside_the_contest_range():
    """The oracle still computes an answer for shapes the gate rejects.

    Keeping the oracle generic is deliberate: it is the tool used to debug a
    kernel, so it must not refuse to run on the shapes being debugged.
    """
    mat_a, mat_b = make_operands(M_MAX + 1, k=DSV3_HIDDEN, n=DSV3_FUSED_A_N, seed=2)
    out = reference(mat_a, mat_b)
    assert out.shape == (M_MAX + 1, DSV3_FUSED_A_N)

    with pytest.raises(ValueError, match="outside the supported range"):
        validate_contest_contract(mat_a, mat_b)
