import json
import os
import sys

import numpy as np
import torch
import monai


def log_environment(
    out_dir: str,
    seed: int,
    patch_size,
    target_spacing,
    speed_mode: bool,
):
    info = {
        "python": sys.version,
        "torch": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "device_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
        "monai": monai.__version__,
        "numpy": np.__version__,
        "seed": seed,
        "target_spacing_mm": target_spacing,
        "patch_size": patch_size,
        "speed_mode": speed_mode,
    }
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "environment.json"), "w") as f:
        json.dump(info, f, indent=2)
