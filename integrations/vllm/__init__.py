"""vLLM integration layer for ``dsv3_fused_a_gemm``.

Holds the upstream-compatible out-variant adapter. Nothing in this package is
part of the competition operator contract or of the default test suite -- see
``integrations/vllm/adapter.py``.
"""

from __future__ import annotations

from .adapter import dsv3_fused_a_gemm_out

__all__ = ["dsv3_fused_a_gemm_out"]
