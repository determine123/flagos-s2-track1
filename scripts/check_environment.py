"""Print reproducibility metadata without reading credentials."""
import importlib.metadata
import json
import platform
import sys

def main():
    result = {"python": sys.version, "os": platform.platform()}
    for name in ("torch", "triton", "triton-windows", "pytest"):
        try:
            result[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            result[name] = None
    try:
        import torch
        result["cuda_available"] = torch.cuda.is_available()
        result["cuda_runtime"] = torch.version.cuda
        result["devices"] = [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]
    except ImportError:
        result["cuda_available"] = False
    print(json.dumps(result, indent=2))

if __name__ == "__main__":
    main()
