"""Print the isolated experiment environment status."""

import importlib.util

import torch

print(
    {
        "torch": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "cuda_version": torch.version.cuda,
        "device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "open_clip_installed": importlib.util.find_spec("open_clip") is not None,
    }
)
