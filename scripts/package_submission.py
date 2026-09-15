"""Validate standalone sources and build a deterministic ZIP; never execute kernels."""
import argparse
import ast
import hashlib
import json
from pathlib import Path
import zipfile

ROOT = Path(__file__).resolve().parents[1]
EXPECTED = {
    "dsv3_fused_a_gemm.py",
    "dsv3_fused_a_gemm_metax.py",
    "dsv3_fused_a_gemm_enflame.py",
    "dsv3_fused_a_gemm_hygon.py",
    "dsv3_fused_a_gemm_ascend.py",
}

def validate(source):
    paths = sorted(source.iterdir())
    if {p.name for p in paths if p.is_file()} != EXPECTED:
        raise ValueError("Expected exactly five named source files")
    payload = {}
    for path in paths:
        if path.name not in EXPECTED:
            continue
        raw = path.read_bytes()
        tree = ast.parse(raw.decode("utf-8"), filename=path.name)
        functions = {n.name: n for n in tree.body if isinstance(n, ast.FunctionDef)}
        for name in ("dsv3_fused_a_gemm", "reference"):
            args = functions[name].args
            if ([a.arg for a in args.args] != ["mat_a", "mat_b"]
                    or args.posonlyargs or args.kwonlyargs or args.vararg
                    or args.kwarg or args.defaults):
                raise ValueError(f"{path.name}: invalid {name} signature")
        if not any(ast.unparse(d) == "triton.jit"
                   for f in functions.values() for d in f.decorator_list):
            raise ValueError("Missing Triton JIT kernel")
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom) and node.level:
                raise ValueError("Relative import in standalone source")
            if isinstance(node, ast.Try):
                raise ValueError("Unexpected exception handler")
            if isinstance(node, ast.Call) and ast.unparse(node.func) in {
                "torch.matmul", "torch.mm", "torch.bmm", "torch.addmm"
            }:
                raise ValueError("Torch computation in submission")
        payload[path.name] = raw
    return payload

def build(source, output):
    payload = validate(source)
    if output.exists() or output.with_suffix(".manifest.json").exists():
        raise FileExistsError("Choose a new output path to preserve existing artifacts")
    output.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(output, "w") as archive:
        for name, raw in payload.items():
            info = zipfile.ZipInfo(name, date_time=(2026, 9, 15, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = 0o100644 << 16
            archive.writestr(info, raw)
    with zipfile.ZipFile(output) as archive:
        if archive.testzip() is not None:
            raise ValueError("ZIP CRC failure")
        for name, raw in payload.items():
            if archive.read(name) != raw:
                raise ValueError("ZIP payload mismatch")
    manifest = {
        "files": {n: hashlib.sha256(v).hexdigest() for n, v in payload.items()},
        "zip_sha256": hashlib.sha256(output.read_bytes()).hexdigest(),
        "validation": "static only; GPU execution not performed",
    }
    output.with_suffix(".manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return manifest

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=ROOT / "submissions/task66_perf_test_02")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(build(args.source, args.output), indent=2))
