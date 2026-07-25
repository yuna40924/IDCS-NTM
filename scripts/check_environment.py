from __future__ import annotations

import argparse
import platform
import subprocess
import sys


def _version(module_name: str) -> str:
    module = __import__(module_name)
    return str(getattr(module, "__version__", "unknown"))


def main() -> None:
    parser = argparse.ArgumentParser(description="Check the IDCS-NTM server runtime")
    parser.add_argument("--require-cuda", action="store_true")
    options = parser.parse_args()

    import torch
    import torchvision
    from torch.func import functional_call  # noqa: F401

    print(f"python={platform.python_version()} executable={sys.executable}")
    print(f"platform={platform.platform()}")
    print(f"torch={torch.__version__} torchvision={torchvision.__version__}")
    print(f"torch_cuda_runtime={torch.version.cuda}")
    print(
        "numpy=" + _version("numpy")
        + " scipy=" + _version("scipy")
        + " yaml=" + _version("yaml")
    )

    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=name,driver_version,memory.total",
                "--format=csv,noheader",
            ],
            check=False,
            capture_output=True,
            text=True,
        )
        if result.stdout.strip():
            print("nvidia_smi=" + result.stdout.strip().replace("\n", " | "))
    except FileNotFoundError:
        print("nvidia_smi=not found")

    available = torch.cuda.is_available()
    print(f"cuda_available={available} device_count={torch.cuda.device_count()}")
    if available:
        for index in range(torch.cuda.device_count()):
            properties = torch.cuda.get_device_properties(index)
            memory_gib = properties.total_memory / (1024**3)
            print(
                f"cuda:{index} name={properties.name} "
                f"capability={properties.major}.{properties.minor} "
                f"memory_gib={memory_gib:.1f}"
            )
    if options.require_cuda and not available:
        raise SystemExit("CUDA is required but torch.cuda.is_available() is false")


if __name__ == "__main__":
    main()
