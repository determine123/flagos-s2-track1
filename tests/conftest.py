"""Pytest configuration: make ``dsv3_fused_a_gemm`` importable from ``src/``.

Keeps the project runnable without an editable install, which is convenient on
machines where the package is not (or cannot be) pip-installed.
"""

from __future__ import annotations

import os
import sys

_SRC = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src"))
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)
