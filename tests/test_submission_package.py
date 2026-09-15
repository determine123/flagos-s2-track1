"""CPU-only checks for immutable submission bytes and archive reproducibility."""
import importlib.util
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
