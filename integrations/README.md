# `integrations/`

Adapters for frameworks that expect a *different* signature from the one the
competition defines. Nothing in this directory is part of the operator contract,
the reference, or the default test suite.

| Adapter | Signature it serves | Status |
|---|---|---|
| `vllm/adapter.py` | `dsv3_fused_a_gemm(output, mat_a, mat_b, enable_pdl=False) -> None` | thin wrapper over the public operator |

## `vllm/`

vLLM vendors the upstream CUDA op as `vllm._custom_ops.dsv3_fused_a_gemm`, which
takes a preallocated `output` first and returns `None`. This repository's
operator does not: per the FlagOS Season 2 competition contract it is

```python
dsv3_fused_a_gemm(mat_a, mat_b) -> torch.Tensor
```

The adapter bridges the two by calling the operator and copying the result into
the caller's buffer, so a vLLM-shaped call site keeps working without the
out-variant leaking back into the competition surface. `enable_pdl` is accepted
and ignored — it is a launch-time CUDA feature that does not affect the result.

It is not tested by `pytest tests/` (the default target) and must not be used as
the oracle for anything.
