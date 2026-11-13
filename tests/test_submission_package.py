"""CPU-only checks for immutable submission bytes and archive reproducibility."""
import importlib.util
import shutil
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("packager", ROOT / "scripts/package_submission.py")
packager = importlib.util.module_from_spec(spec)
spec.loader.exec_module(packager)

def test_best_source_hashes_and_reproducible_zip(tmp_path):
    source = ROOT / "submissions/task66_perf_test_02"
    first = packager.build(source, tmp_path / "one.zip")
    second = packager.build(source, tmp_path / "two.zip")
    assert first == second
    assert first["files"]["dsv3_fused_a_gemm.py"] == "7a352710d114ea693f54591ecc09fc7bf62b08346fd5d2544767e9c584a6da25"
    assert first["files"]["dsv3_fused_a_gemm_ascend.py"] == "9a3fd60bdab7a1dfe2367222a927dcf2e71b8363ba53829857afa49a4966a5cc"
    with pytest.raises(FileExistsError):
        packager.build(source, tmp_path / "one.zip")

def test_missing_default_rejected(tmp_path):
    with pytest.raises(ValueError):
        packager.validate(tmp_path)

def _copy_best_source(tmp_path):
    target = tmp_path / "source"
    shutil.copytree(ROOT / "submissions/task66_perf_test_02", target)
    return target


@pytest.mark.parametrize(
    "snippet",
    [
        "def hidden_matmul(mat_a, mat_b):\n    return mat_a @ mat_b",
        "def hidden_matmul(mat_a, mat_b):\n    return mat_a.matmul(mat_b)",
        'def hidden_matmul(mat_a, mat_b):\n    return torch.einsum("ij,jk->ik", mat_a, mat_b)',
    ],
    ids=["operator", "method", "torch-einsum"],
)
def test_packager_rejects_torch_matrix_computation_forms(tmp_path, snippet):
    source = _copy_best_source(tmp_path)
    target = source / "dsv3_fused_a_gemm.py"
    target.write_text(
        target.read_text(encoding="utf-8") + "\n\n" + snippet + "\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="Torch matrix computation"):
        packager.validate(source)


def test_packager_reports_missing_required_function(tmp_path):
    source = _copy_best_source(tmp_path)
    target = source / "dsv3_fused_a_gemm.py"
    target.write_text(
        "def reference(mat_a, mat_b):\n"
        "    return mat_a\n\n"
        "@triton.jit\n"
        "def kernel():\n"
        "    pass\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="missing required function dsv3_fused_a_gemm"):
        packager.validate(source)
