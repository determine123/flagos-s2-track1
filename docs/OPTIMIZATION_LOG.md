# Optimization Log — `dsv3_fused_a_gemm`

> Historical pre-submission notes. The subsequent five platform submissions and
> best 8/8 result are recorded in EXPERIMENTS.md and ../results/platform_submissions.csv.
> Statements below about pending GPU measurements describe the local environment.

Running record of design decisions and measured results. **Every performance
number in this file must come from an actual run on a CUDA device.** Nothing is
estimated, copied from a paper, or extrapolated. Rows without measurements say so
explicitly rather than being left blank or filled with a guess.

Environment note: the development machine has **no GPU** (`torch.cuda.is_available()
== False`, torch `2.8.0+cpu`). All local work is therefore reference-level and
static; every kernel measurement is pending a cloud GPU run.

---

## Phase 0 — Scope and contract (2026-09-14)

Pinned the operator contract against the upstream implementation rather than
working from memory. Sources: `vllm/_custom_ops.py` (`dsv3_fused_a_gemm`) and
`vllm/model_executor/models/deepseek_v2.py` (`DeepSeekV2FusedQkvAProjLinear`).

Decisions:

- ~~**Signature fixed to upstream exactly** — `(output, mat_a, mat_b, enable_pdl=False)`,
  `output` first, returns `None`. Divergence here would break drop-in use.~~
  **Superseded — see Phase 0.1. The competition defines the contract, not the
  vLLM upstream op.**
- **`mat_b` non-contiguity treated as a first-class requirement**, not an
  implementation detail. It falls out of fusing the two projection weights, and
  eliminating the `.contiguous()` copy is part of the fusion's value.
- **Reference deliberately never materialises a contiguous `mat_b`**, so the
  oracle exercises the same layout the kernel must handle.
- **Two operand layouts tested** (strided and contiguous). A contiguous-only
  kernel fails the first; a strided-only kernel fails the second.
- Tolerances derived from K and bf16 precision (see `SPEC.md` §4) rather than
  picked by feel.

## Phase 0.1 — Contract corrected (2026-09-15)

The Phase 0 decision above was wrong. It pinned the operator to vLLM's upstream
CUDA signature; the **FlagOS Season 2 competition defines the contract**, and the
two are not the same. Upstream is a *consumer* of this operator, not its
definition.

Corrected public contract:

```python
def dsv3_fused_a_gemm(mat_a, mat_b):
    out = torch.empty(
        (mat_a.shape[0], mat_b.shape[1]), dtype=mat_a.dtype, device=mat_a.device
    )
    # must actually launch a Triton/Triton-TLE kernel
    return out
```

Reference oracle:

```python
def reference(mat_a, mat_b):
    return (mat_a.float() @ mat_b.float()).to(mat_a.dtype)
```

What changed, and why:

- **`output`, `enable_pdl`, and the `None` return are gone from the competition
  path.** The operator takes exactly `(mat_a, mat_b)` and returns a fresh tensor.
  `tests/test_kernel.py::test_kernel_has_no_out_variant_parameter` asserts the
  parameter list so an out-variant cannot creep back in.
- **vLLM's out-variant moved to `integrations/vllm/adapter.py`.** It wraps the
  competition operator and copies into the caller's buffer. It is not the
  reference, not the competition interface, and not a default test target.
- **The `mat_b` stride convention was stated inconsistently.** `[K, N]` with
  strides `(1, K)` is correct; an earlier draft of `SPEC.md` §5 said `(1, N)`,
  which is a different — and wrong — layout. The contiguous axis of `mat_b` is K.
  `test_hidden_is_the_contiguous_axis_of_mat_b` now pins the difference down.
- **The correctness range is the published competition one**, not the DSV3 shape:
  `M ∈ [1, 16]`, `K % 256 == 0`, `N % 16 == 0`, bf16. `K=7168, N=2112` is the
  headline benchmark and specialisation target, not a correctness restriction;
  the generic-shape tests sweep `K ∈ {256, 512, 1024, 7168}`,
  `N ∈ {16, 32, 128, 2112}`.
- **The reference upcasts both operands to fp32 explicitly** and casts the result
  back to the input dtype. The previous oracle used `torch.matmul(..., out=...)`
  into a bf16 buffer.
- **The benchmark times the returning interface** for every implementation
  (torch and Triton alike), so the output allocation is inside the measurement
  for all of them. Timing an `out=` variant would not have measured the contract.

`KERNEL_AVAILABLE` stays `False`: no KernelGen-generated code has been obtained
yet, and the stub must fail loudly rather than approximate.

### Layout framing settled (2026-09-15, follow-up)

An earlier draft of this correction treated "strided *and* contiguous" as two
co-equal requirements. That overstates it, and the overstatement is expensive: a
kernel generalised for both layouts can pay for it on the one that matters.

Settled framing:

- The **column-major `[K, N]` view with strides `(1, K)` is the main path**. It is
  what the competition always passes, it is what the kernel is optimised and
  validated for, and it is the configuration every headline benchmark number is
  measured on.
- A **contiguous `[K, N]` operand is an optional project extension**, not a
  competition requirement. It is kept as a compatibility test (`pytest -m
  extension`) so the kernel does not rest on a layout assumption it never has to
  make — but it is not a gate.
- **The extension must not cost the main path anything.** It may be supported
  through internal stride parameters or a separate dispatch branch, never by
  adding a parameter to the public signature (which stays `(mat_a, mat_b)`). If
  supporting it degrades the column-major path, drop the extension.
- `validate_contest_contract` deliberately does **not** gate on strides: a caller
  passing a contiguous tensor is asking a legitimate question. The requirement is
  documented, not enforced by rejecting a valid GEMM.

Consequence for kernel generation: `SPEC.md` §6 asks for a kernel tuned for the
column-major view first, with the contiguous case explicitly optional.

### Stale archive (2026-09-15)

`D:\Projects\flagos-s2-track1.zip` (2026-09-14 22:11) is a snapshot taken
**before** the contract correction. It contains the old contract in 9 of its
files, including `dsv3_fused_a_gemm_out` / `enable_pdl` / the `None` return
throughout its tests.

**Status: stale. Do not commit, do not distribute.** It sits outside this
repository's root, so it is not tracked by git and cannot be committed from here
by accident — but it must not be mistaken for a deliverable either.

**Do not repackage until the real kernel is integrated and has passed the GPU
test suite.** A fresh archive is only meaningful once `KERNEL_AVAILABLE = True`
and `pytest tests/ -v` has run green on CUDA hardware.

## Phase 1 — Project skeleton (2026-09-14)

Delivered: package layout, reference oracle, test suite, benchmark harness, spec,
this log. No kernel yet.

**Status: reference-only. No performance data exists yet — by design.**

| What | State |
|---|---|
| Reference oracle | implemented; validated on CPU |
| Test suite | implemented; kernel tests auto-skip until integration |
| Benchmark harness | implemented; reports no CPU numbers |
| Triton kernel | **integrated, not yet run** — one Web generation in `_kernelgen.py` |

## Phase 2 — Kernel generation via KernelGen Web (pending)

### Why the Web platform and not the MCP integration

The Claude Code MCP route was attempted first and **failed three times**, each
terminating at *exactly* 300 s of silence:

```
MCP server "kernelgen-server" tool "generate_kernel" sent no response or
progress for 300s; aborting.
```

Ruled out during diagnosis:

- **Token expiry** — not it; the JWT `exp` decodes to ~Dec 2026.
- **Wrong config file** — not it. `CLAUDE_CONFIG_DIR` points at
  `D:\ClaudeCode\config\.claude`, and that `.claude.json` shares an inode with
  `C:\Users\Administrator\.claude\.claude.json` (same device:inode), i.e. one
  file, not a stale duplicate.
- **Per-server `timeout`** — `"timeout": 1800000` was written into
  `mcpServers.kernelgen-server` and verified present in the live file **after a
  full restart**; the abort still fired at 300 s. The field appears to govern
  connection/handshake, not the tool-call idle window.

Not conclusively established: whether the exact-300 s cutoff is an undocumented
silent-output monitor, and whether the third-party `deepseek-flash[1m]` endpoint
(`ANTHROPIC_BASE_URL=https://api.deepseek.com/anthropic`) contributes. Treated as
an **infrastructure blocker, deferred** — not a code problem, and not worth
further budget.

**Consequence:** initial kernel generation moves to the KernelGen Web platform,
which has no such silent-output limit. The request parameters are recorded in
`SPEC.md` §5 so the run is reproducible by anyone.

## Phase 3 — Integration (2026-09-15)

One KernelGen Web generation obtained and integrated. **It has not been run** —
this machine has no CUDA device — so nothing in this section is a correctness or
performance claim, only a record of what was wired in.

### What landed

- `src/dsv3_fused_a_gemm/_kernelgen.py` — the generated body, kept verbatim
  apart from one deletion (below). 12 autotune configs (`BLOCK_N` ∈ {16,32,64,128}
  × `BLOCK_K` ∈ {32,64,128}), `BLOCK_M` fixed at 16, `key=["M","N","K"]`.
- `src/dsv3_fused_a_gemm/kernel.py` — thin public surface: the competition
  signature, the contest gate, and `KERNEL_AVAILABLE`.

### Design decisions taken during integration

- **The generated `reference` was deleted.** The bundle shipped::

      def reference(mat_a, mat_b):
          return dsv3_fused_a_gemm(mat_a, mat_b)

  which calls the Triton kernel. Any comparison against it would have passed
  unconditionally. This is the single most dangerous thing the generator
  produced, because a harness that trusts the file's own `reference` reports
  perfect accuracy for any kernel at all. Recorded in `SPEC.md` §6 as a step to
  check on every regeneration.
- **The generated body was placed in its own module rather than pasted into
  `kernel.py`.** `pyproject.toml` declares Triton an optional extra
  (`kernel = ["triton>=3.0"]`), so the package must import without it —
  otherwise the CPU reference suite could not run on a machine that has no
  Triton, which defeats the point of having a GPU-independent oracle.
  `kernel.py` imports `_kernelgen` behind a guard and derives
  `KERNEL_AVAILABLE` from it, so nothing needs editing after a regeneration.
- **`validate_contest_contract` runs in the wrapper, before dispatch** — the
  generated code has no gate of its own. Verified: `M=17` raises `ValueError`
  before the kernel is reached.

### The `mat_b` layout

The generated kernel addresses `mat_b` through its real `stride_bk` / `stride_bn`
arguments, which is the spec's strategy (a). For the competition's column-major
view those are `(1, K)`; for a contiguous `[K, N]` tensor they are `(N, 1)`. Both
resolve correctly from one code path, so the contiguous extension costs the main
path nothing — no branch, no extra public argument.

Performance note for Phase 4: strategy (a) walks the K axis, which is contiguous,
but the tile is addressed column-major relative to the load's fast axis. Strategy
(b) — load the contiguous `(BLOCK_N, BLOCK_K)` tile and `tl.trans` in-kernel — is
usually faster. Worth measuring before assuming, and only after a baseline exists.

### Verification done locally (CPU, no GPU)

| Check | Result |
|---|---|
| Full suite | 88 passed, 30 skipped |
| `pytest -m "not extension"` | main path only, no extension cases |
| Public signature | exactly `(mat_a, mat_b)` |
| `KERNEL_AVAILABLE` | `True` |
| Gate ordering | `M=17` → `ValueError` before dispatch |
| Import without Triton (simulated) | package imports, `KERNEL_AVAILABLE=False`, clear `NotImplementedError`, reference unaffected |

Still outstanding, and only a CUDA box can do it:

```
pytest tests/ -v
python benchmarks/bench_dsv3_fused_a_gemm.py --json results.json
```

## Phase 4 — Optimization iterations (pending a GPU run)

Empty until Phase 3 produces a baseline. Each entry will record: change made,
rationale, measured before/after median ms, and the shape/dtype config — with
raw JSON cited so numbers are traceable.

| Iter | Change | M | dtype | before (ms) | after (ms) | speedup |
|---|---|---|---|---|---|---|
| — | *no measurements yet* | — | — | — | — | — |

---

## Results summary

**No measured results exist.** The kernel has not been run. The table below is
the template that will be filled from Phase 3 onward; it is intentionally left
empty rather than populated with placeholder or indicative figures.

| M | dtype | torch_noncontig (ms) | torch_contig (ms) | triton (ms) | speedup vs baseline |
|---|---|---|---|---|---|
| 1 | bf16 | *pending* | *pending* | *pending* | *pending* |
| 2 | bf16 | *pending* | *pending* | *pending* | *pending* |
| 4 | bf16 | *pending* | *pending* | *pending* | *pending* |
| 8 | bf16 | *pending* | *pending* | *pending* | *pending* |
| 16 | bf16 | *pending* | *pending* | *pending* | *pending* |

### Measurement methodology (fixed in advance, to avoid post-hoc rationalisation)

- `triton.testing.do_bench`, median, quantiles `(0.5, 0.2, 0.8)`.
- Warmup 25, rep 100.
- **L2 flushed between reps by default.** The fused weight is ~30 MB in bf16,
  which fits in the L2 of large parts; a warm cache would overstate throughput.
  The cache-resident case is reported separately via `--no-flush-l2`, never
  mixed into the headline figure.
- Baseline is `torch_noncontig` — the honest comparison, since a real caller
  never materialises a contiguous weight. `torch_contig` is reported to price the
  transpose the fusion avoids.
- First-sample outliers: `do_bench` warms up, so JIT compile cost is excluded; if
  a sample set is still skewed the median resists it better than the mean.
