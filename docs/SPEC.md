# `dsv3_fused_a_gemm` — Operator Specification

> Historical design document retained from local development. Current platform
> outcomes and limitations are in EXPERIMENTS.md and REPRODUCIBILITY.md.
> Local tolerance choices below are not a verified copy of the official checker.

## 1. What the operator is

The fused "A" projection of the DeepSeek-V3 MLA (Multi-head Latent Attention)
block. DeepSeek-V3's attention compresses queries and keys/values into low-rank
latents before expanding them per-head:

| Projection | Shape | Meaning |
|---|---|---|
| `q_a_proj` | `[hidden] → [q_lora_rank]` | query compression |
| `kv_a_proj_with_mqa` | `[hidden] → [kv_lora_rank + qk_rope_head_dim]` | KV compression + RoPE key slice |

Both consume **the same input hidden state**, so their weight matrices can be
concatenated along the output dimension and the two GEMMs collapsed into one:

```
N = q_lora_rank + kv_lora_rank + qk_rope_head_dim = 1536 + 512 + 64 = 2112
K = hidden = 7168
```

This removes one kernel launch from the decode path. Decode is latency-bound
(one token at a time, tiny M), so launch overhead and kernel count dominate the
end-to-end cost — which is why the fusion is worth doing even though the
arithmetic is unchanged. SGLang measured ~3.8–4.2% end-to-end improvement for
DeepSeek models at `bs=1..32` from exactly this fusion.

## 2. Contract

**The contract is defined by the FlagOS Season 2 competition.** It is two
arguments in, one new tensor out:

```python
def dsv3_fused_a_gemm(mat_a, mat_b):
    out = torch.empty(
        (mat_a.shape[0], mat_b.shape[1]),
        dtype=mat_a.dtype,
        device=mat_a.device,
    )
    # must actually launch a Triton / Triton-TLE kernel
    return out
```

| Aspect | Requirement |
|---|---|
| Semantics | `out = mat_a @ mat_b` |
| Signature | `dsv3_fused_a_gemm(mat_a, mat_b)` — exactly two positional arguments |
| Return | a **freshly allocated** `[M, N]` row-major tensor of the input dtype |
| `mat_a` | `[M, K]`, row-major |
| `mat_b` | `[K, N]` **column-major transposed view**, strides `(1, K)` — `weight.T` of a contiguous `[N, K]` weight |
| `mat_b` layout | the column-major view is the **required main path**; a contiguous `[K, N]` tensor is an optional project extension (§2.2) |
| `M` | `1..16` (low-latency decode regime) |
| `K` | a multiple of `256` |
| `N` | a multiple of `16` |
| dtype | `bfloat16` |
| Mutates | nothing — neither operand may be written |

There is **no `output` parameter, no `enable_pdl` flag, and no `None` return** on
this surface. A caller supplies both operands and receives a tensor.

### 2.1 The column-major `mat_b` — the defining constraint

The weight is materialised row-major as `[N, K]`. Therefore:

```
weight      : [2112, 7168]  strides (7168, 1)   contiguous
mat_b = w.T : [7168, 2112]  strides (1, 7168)   NON-contiguous
```

Note which stride is 1: **the contiguous axis of `mat_b` is K**, because K is the
contiguous axis of the weight it views. A `[K, N]` operand with strides `(1, N)`
is a *different layout*, not this one — see
`test_hidden_is_the_contiguous_axis_of_mat_b`, which pins the difference down.

This is not an accident of the API. It is the natural form after concatenating
the two projection weights, and **avoiding the `.contiguous()` copy is part of
the point of the fusion** — materialising a 30 MB bf16 weight per call would
cost more than the kernel saves.

Consequences for the kernel:

1. Addressing `mat_b` requires its real strides. Assuming row-major `[K, N]`
   silently reads the wrong elements.
2. Equivalently, `mat_b` is a row-major `[N, K]` matrix read transposed. Loading
   it as the contiguous `[N, K]` while transposing in-kernel is usually the
   faster strategy, because global loads stay coalesced along the contiguous
   axis; the transpose then happens in shared memory.

This layout is the **main path**. It is what the competition passes, it is what
the kernel must be optimised and validated for, and it is the configuration every
headline benchmark number is measured on.

`tests/test_reference.py::test_mat_b_transposed_view_is_non_contiguous` asserts
the stride layout explicitly, so strided tests cannot silently degrade into
testing the easy path.

### 2.2 A contiguous `mat_b` is an optional project extension

A genuinely contiguous `[K, N]` operand is **not** a competition requirement. It
is carried as a project extension compatibility test, selectable with
`pytest -m extension`, so the kernel does not rest on a layout assumption it
never has to make.

Where the extension is supported, the public function still takes exactly two
arguments — `(mat_a, mat_b)`. The extra case must be reached through **internal
stride parameters or a separate dispatch branch**, never by adding a parameter to
the competition signature.

**The extension must not cost the main path anything.** The column-major view is
the measured configuration; a dispatch branch, a stride generalisation, or a tile
shape chosen for contiguous inputs are acceptable only if the column-major path
keeps its performance. If supporting the extension degrades the main path, drop
the extension — the competition does not require it.

## 3. Reference implementation

`src/dsv3_fused_a_gemm/reference.py` is the oracle the kernel is compared
against:

```python
def reference(mat_a, mat_b):
    return (mat_a.float() @ mat_b.float()).to(mat_a.dtype)
```

Two properties are deliberate and tested:

- **Both operands are upcast to float32 for the multiply**, and only the result
  is cast back to the input dtype. A bfloat16 oracle would be too coarse to
  separate a correct kernel from a subtly wrong one.
- **`mat_b` is never made contiguous.** `torch.matmul` honours arbitrary strides,
  and the strided view is exactly the layout the real kernel must handle;
  materialising it would make the oracle pass for the wrong reason.

The reference stays *permissive* about M/K/N — it will compute an answer for
shapes the competition rejects, because it is also the tool used to debug them.
Enforcing the range is a separate gate, `validate_contest_contract`, which the
competition surface applies.

### 3.1 vLLM's out-variant is an adapter, not the contract

vLLM vendors the upstream CUDA op as `vllm._custom_ops.dsv3_fused_a_gemm`, whose
signature is `(output, mat_a, mat_b, enable_pdl=False) -> None`. That is **not**
this operator's contract and must not leak onto the competition surface. It is
preserved separately in `integrations/vllm/adapter.py`, which calls
`dsv3_fused_a_gemm(mat_a, mat_b)` and copies the result into the caller's buffer.
Nothing under `integrations/` is the reference, the competition interface, or a
default test target.

## 4. Correctness range

The published competition constraints, enforced by
`reference.validate_contest_contract`:

| Constraint | Value |
|---|---|
| `M` | `1 ≤ M ≤ 16` |
| `K` | `K % 256 == 0` |
| `N` | `N % 16 == 0` |
| dtype | `bfloat16` only |
| `mat_b` layout | column-major transposed view, strides `(1, K)` — the required main path |

`K = 7168`, `N = 2112` is the **DeepSeek-V3 headline benchmark shape**, and the
focus token counts are `M ∈ {1, 2, 4, 8, 16}` (`DSV3_SHAPES`). It is a
*specialisation* target — worth tuning tile sizes and launch geometry for — and
**not** a correctness restriction. Every shape satisfying the table above must be
correct, which the generic-shape tests sweep explicitly
(`K ∈ {256, 512, 1024, 7168}`, `N ∈ {16, 32, 128, 2112}`).

The `mat_b` layout row is the one exception to "the table is the whole gate":
`validate_contest_contract` checks shapes and dtype, not strides, because a
caller that happens to pass a contiguous tensor is still asking a legitimate
question. The *requirement* is nonetheless the column-major view — it is the
layout the competition presents and the one every performance claim is made
against. `pytest -m "not extension"` runs exactly that main-path suite.

## 5. Numerical tolerances

| Output dtype | rtol | atol |
|---|---|---|
| `bfloat16` | 2e-2 | 1e-2 |
| `float16` | 2e-2 | 1e-2 |
| `float32` | 1e-4 | 1e-4 |

Rationale for bf16: accumulation over `K=7168` standard-normal products yields a
pre-rounding magnitude of ~√7168 ≈ 85, and bf16 carries ~2⁻⁸ relative precision.
The tolerance must exceed one bf16 ULP to avoid flagging correct kernels, yet stay
tight enough that an indexing bug — which produces O(1) relative error — cannot
sneak through.

Worth noting: split-K reduction makes the *last few* ULPs of near-zero outputs
order-dependent, so elements whose true value is near zero can show unbounded
relative error while absolute error stays tiny. This is expected and is why the
harness uses `torch.allclose` (which combines rtol and atol) rather than a pure
relative-error test.

## 6. KernelGen Web request

The initial kernel is generated on the **KernelGen Web platform**
(https://kernelgen.flagos.io), not via the Claude Code MCP integration — see
`docs/OPTIMIZATION_LOG.md` for why.

Integration is a single-file swap:

1. Generate with the parameters below.
2. Replace `src/dsv3_fused_a_gemm/_kernelgen.py` with the generated
   `triton_code`, keeping the entry point named `dsv3_fused_a_gemm` with the
   signature `(mat_a, mat_b) -> Tensor`.
3. **Delete any generated `reference` function if it calls the kernel.** The
   first Web generation shipped exactly that —
   `def reference(mat_a, mat_b): return dsv3_fused_a_gemm(mat_a, mat_b)` — which
   compares the kernel to itself and would make every correctness check pass
   vacuously. The oracle is `dsv3_fused_a_gemm.reference.reference`.
4. Nothing else needs editing: `KERNEL_AVAILABLE` is derived in `kernel.py` from
   whether `_kernelgen` imports, so it flips automatically. The generated body
   sits behind a guard because Triton is an optional extra
   (`pip install 'dsv3-fused-a-gemm[kernel]'`), which is what lets the CPU test
   suite import the package on a machine with no Triton.

**Status: one Web generation has been integrated** (12 autotune configs, generic
`stride_bk` / `stride_bn` addressing). It has **not** been run — the development
machine has no CUDA device — so no correctness or performance claim is attached
to it yet.

Request parameters to use:

| Field | Value |
|---|---|
| `kernel_name` | `dsv3_fused_a_gemm` |
| `func_type` | `other` |
| `device` | `nvidia` |

**`func_desc`:**

> Low-latency fused-A GEMM for the DeepSeek-V3 MLA projection. Computes
> `out = mat_a @ mat_b` and **returns a newly allocated output tensor** — there is
> no output argument in the signature, and nothing is written in place.
>
> `mat_a` is `[M, K]` row-major. `mat_b` is `[K, N]` and is the *column-major
> transposed weight*: it is the `.T` view of a contiguous `[N, K]` weight, so its
> strides are `(1, K)` — the K axis is the contiguous axis — and it is NOT
> contiguous. **This column-major view is the required main path and the
> performance-critical one**: it is the layout the caller always passes, and the
> kernel must be optimised and validated for it. Read `mat_b` through its actual
> strides, or load it as the contiguous `[N, K]` weight and transpose in-kernel,
> whichever is faster. Assuming a row-major `[K, N]` operand with strides `(1, N)`
> is incorrect and will read the wrong elements.
>
> A genuinely contiguous `[K, N]` operand is an optional extension, not a
> requirement. Support it only through internal stride parameters or a separate
> dispatch branch, and only if it does not slow down the column-major path.
>
> dtype bfloat16; M in 1..16; K a multiple of 256; N a multiple of 16. The
> headline target is M in {1,2,4,8,16}, K=7168, N=2112, but the kernel must be
> correct for any shape meeting those constraints.

**`arg_names`:** `mat_a, mat_b`

**`arg_type`:** `Tensor, Tensor`

**`arg_descs`:**

> mat_a: [M, K] row-major activations, bfloat16, M in 1..16, K % 256 == 0.
> mat_b: [K, N] column-major transposed weight with strides (1, K), derived from
> a contiguous [N, K] weight via .T. This is the required main path and must NOT
> be assumed contiguous. N % 16 == 0.

**`output_arg_desc`:**

> A newly allocated [M, N] row-major bfloat16 tensor is returned. There is no
> output argument and nothing is written in place.

**Reference notes to pass along:**

- Reference oracle: `(mat_a.float() @ mat_b.float()).to(mat_a.dtype)`, with
  `mat_b` left as the non-contiguous column-major view.
- Acceptance tolerance vs. a float32 oracle: rtol 2e-2, atol 1e-2 for bf16.
- The fused N dimension is `q_lora_rank + kv_lora_rank + qk_rope_head_dim`
  = 1536 + 512 + 64 = 2112; K = hidden = 7168.
- Both `q_a_proj` and `kv_a_proj_with_mqa` take the same input, which is why they
  fuse into one GEMM.

## 7. Out of scope

- A contiguous `[K, N]` `mat_b`. Optional project extension only (§2.2); the
  competition always passes the column-major view.
- Dropout, bias, and activation epilogues — the upstream op has none.
- The `q_b_proj` / `kv_b_proj` expansion GEMMs and the attention itself.
- Quantised (fp8/int8) weights — the competition is gated to bf16.
- Programmatic dependent launch. `enable_pdl` belongs to vLLM's out-variant and
  is accepted-and-ignored by the adapter in `integrations/vllm/`; the competition
  signature has no such flag.
- Batch sizes beyond 16: vLLM falls back to `F.linear` there, and matching that
  behaviour is the caller's concern, not this kernel's.
